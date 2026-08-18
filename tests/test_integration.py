"""End-to-end integration test: the whole app through its real API surface.

The only mocked pieces are the LLM (``FakeListChatModel``) and embeddings
(fake vectors, from the ``fake_embeddings`` fixture), so no Ollama daemon is
required. Everything else runs for real: the LangGraph agent, the vault
ingester, the unified vector store, the watcher lifecycle, SQLite memory,
and SSE streaming.
"""
import json

import chromadb
import pytest
import rag.graph as graph_mod
import rag.vectorstore as vs_mod
from config import settings
from fastapi.testclient import TestClient
from langchain_core.language_models import FakeListChatModel


@pytest.fixture
def app_client(tmp_path, monkeypatch, fake_embeddings):
    """A TestClient whose graph uses a deterministic fake LLM.

    Note: separate ``chromadb.Client()`` instances in this chroma version share
    one in-process data store, so the shared ``localbrain`` collection is
    deleted on setup for test isolation.
    """
    from main import app

    original_store = vs_mod.vectorstore
    vs = vs_mod.LocalBrainVectorStore(client=chromadb.Client())
    try:
        vs._client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    except Exception:  # noqa: BLE001, S110
        pass
    vs_mod.vectorstore = vs

    original_persist = settings.CHROMA_PERSIST_DIR
    # Sub-dir inside tmp_path: the memory DB is derived from the *parent* of
    # CHROMA_PERSIST_DIR, so tmp_path/chroma gives each test its own DB file.
    settings.CHROMA_PERSIST_DIR = str(tmp_path / "chroma")

    # Force the agent graph to rebuild with the fake LLM on first query.
    monkeypatch.setattr(graph_mod, "_agent_graph", None)
    monkeypatch.setattr(
        graph_mod,
        "get_llm",
        lambda **kwargs: FakeListChatModel(
            responses=[
                "retrieve",                        # router -> retrieve
                "yes", "yes", "yes", "yes", "yes",  # grade up to 5 docs
                "The decisions agreed.",           # generate
                "yes",                             # reflect -> grounded
            ]
        ),
    )

    with TestClient(app) as c:
        yield c

    settings.CHROMA_PERSIST_DIR = original_persist
    vs_mod.vectorstore = original_store


def _sse_tokens(body: str) -> str:
    """Concatenate all ``{type: token}`` payloads from an SSE response body."""
    parts = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            msg = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "token":
            parts.append(msg.get("data", ""))
    return "".join(parts)


class TestWholeApp:
    def test_ingest_and_query_flow(self, app_client):
        # 1) Ingest the sample vault through the API route
        res = app_client.post("/api/vault/ingest")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "indexed"
        assert data["chunks"] > 0

        # 2) Workspace listing reflects the frontmatter labels
        ws = app_client.get("/api/workspaces")
        assert ws.status_code == 200
        assert "work" in ws.json()["workspaces"]
        assert "personal" in ws.json()["workspaces"]

        # 3) Notes are scoped by workspace
        notes = app_client.get("/api/notes", params={"workspace": "work"})
        assert notes.status_code == 200
        assert notes.json()["count"] >= 2

        # 4) Agent query — full graph loop through the sync endpoint
        q = app_client.post(
            "/api/query",
            json={"question": "What decisions were made?", "workspace": "work"},
        )
        assert q.status_code == 200
        body = q.json()
        assert body["answer"] == "The decisions agreed."
        assert isinstance(body["sources"], list) and body["sources"]
        assert body["latency_seconds"] >= 0

    def test_stream_endpoint_emits_real_tokens(self, app_client):
        app_client.post("/api/vault/ingest")

        res = app_client.post(
            "/api/query/stream",
            json={"question": "What decisions were made?", "workspace": "work"},
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")

        body = res.text
        # The generate node streams the LLM output as {type: token} events
        streamed = _sse_tokens(body)
        assert streamed, "expected at least one token event"
        # ...and reports which notes were used (order follows retrieval)
        assert '"type": "source_documents"' in body
        assert '"workspace/note1.md"' in body
        assert '"workspace/note2.md"' in body
        # ...then terminates
        assert '"type": "done"' in body

        # The accumulated streamed tokens are exactly what lands in history
        h = app_client.get("/api/history", params={"workspace": "work"}).json()
        assert h["count"] == 2
        assert h["messages"][0]["role"] == "human"
        assert h["messages"][1]["role"] == "ai"
        assert h["messages"][1]["content"] == streamed

    def test_conversation_history_persists(self, app_client):
        app_client.post("/api/vault/ingest")
        res = app_client.post(
            "/api/query",
            json={"question": "What decisions were made?", "workspace": "work"},
        )
        answer = res.json()["answer"]

        h = app_client.get("/api/history", params={"workspace": "work"}).json()
        assert h["count"] == 2
        assert [m["role"] for m in h["messages"]] == ["human", "ai"]
        # The saved reply matches what the endpoint returned
        assert h["messages"][1]["content"] == answer

        # Clearing removes the thread
        clr = app_client.delete("/api/history", params={"workspace": "work"})
        assert clr.json()["status"] == "cleared"
        assert app_client.get("/api/history", params={"workspace": "work"}).json()["count"] == 0

    def test_health_and_frontend(self, app_client):
        res = app_client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert isinstance(data["ollama_connected"], bool)
        assert "vault_path" in data
        assert data["notes_indexed"] == 3  # vault index rebuilt on startup from sample vault

        page = app_client.get("/")
        assert page.status_code == 200
        assert "LocalBrain" in page.text