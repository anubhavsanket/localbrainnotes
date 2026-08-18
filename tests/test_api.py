"""Tests for the FastAPI API layer.

Uses an in-memory ChromaDB and monkeypatches the module-level ``vectorstore``
singleton so no Ollama daemon is required to run these tests.
"""
import pytest
from fastapi.testclient import TestClient

import rag.vectorstore as vs_mod


@pytest.fixture
def client(tmp_path):
    """Build a TestClient backed by an in-memory vector store and a temp db."""
    import chromadb as _cdb

    original_store = vs_mod.vectorstore
    vs = vs_mod.LocalBrainVectorStore(client=_cdb.Client())
    vs_mod.vectorstore = vs

    # Patch settings for test context. Sub-dir inside tmp_path so the memory
    # DB (derived from the parent of CHROMA_PERSIST_DIR) is unique per test.
    from config import settings

    original_persist = settings.CHROMA_PERSIST_DIR
    settings.CHROMA_PERSIST_DIR = str(tmp_path / "chroma")

    from main import app

    with TestClient(app) as c:
        yield c

    settings.CHROMA_PERSIST_DIR = original_persist
    vs_mod.vectorstore = original_store


@pytest.fixture
def notes_client(tmp_path, fake_embeddings):
    """TestClient with VAULT_PATH redirected to a throwaway dir seeded with
    two notes, plus an in-memory Chroma so no Ollama/disk is touched."""
    import chromadb as _cdb
    import db.store as store_mod

    original_store = vs_mod.vectorstore
    vs_mod.vectorstore = vs_mod.LocalBrainVectorStore(client=_cdb.Client())

    from config import settings

    original_vault = settings.VAULT_PATH
    vault = tmp_path / "vault"
    (vault / "work").mkdir(parents=True)
    (vault / "work" / "note1.md").write_text(
        "---\ntitle: Note One\nworkspace: work\n---\n# Note One\n\nContent.",
        encoding="utf-8",
    )
    settings.VAULT_PATH = str(vault)

    # Isolate the JSON vault index so the temp-vault rebuild can't overwrite
    # the real backend/db/vault_index.json (or leak state between tests).
    original_index = store_mod._index
    original_index_path = store_mod._INDEX_PATH
    store_mod._INDEX_PATH = tmp_path / "vault_index.json"
    store_mod._index = {}

    from main import app

    with TestClient(app) as c:
        yield c, vault

    settings.VAULT_PATH = original_vault
    vs_mod.vectorstore = original_store
    store_mod._INDEX_PATH = original_index_path
    store_mod._index = original_index


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "notes_indexed" in data


class TestVaultIngest:
    def test_ingest_full_vault(self, client, fake_embeddings):
        res = client.post("/api/vault/ingest")
        assert res.status_code == 200
        data = res.json()
        assert data["chunks"] > 0

    def test_ingest_single_file(self, client, fake_embeddings):
        # note1 should exist from sample vault
        res = client.post(
            "/api/vault/ingest/file",
            params={"path": "vaults/sample/workspace/note1.md"},
        )
        assert res.status_code == 200

    def test_ingest_single_file_rejects_path_escape(self, client, fake_embeddings):
        """Regression: the vault/ingest/file route must refuse files outside
        the vault root (was: arbitrary filesystem paths accepted)."""
        res = client.post(
            "/api/vault/ingest/file",
            params={"path": "../backend/main.py"},
        )
        assert res.status_code == 400
        assert "escapes vault" in res.json()["detail"]

    def test_ingest_single_file_404_for_missing(self, client, fake_embeddings):
        res = client.post("/api/vault/ingest/file", params={"path": "vaults/sample/nope.md"})
        assert res.status_code == 404


class TestWorkspaces:
    def test_list_workspaces(self, client, fake_embeddings):
        # First ingest so there are notes to index
        client.post("/api/vault/ingest")
        res = client.get("/api/workspaces")
        assert res.status_code == 200
        data = res.json()
        assert "workspaces" in data
        assert isinstance(data["workspaces"], list)


class TestHistory:
    def test_clear_history(self, client):
        res = client.delete("/api/history", params={"workspace": "default"})
        assert res.status_code == 200
        assert res.json()["status"] == "cleared"

    def test_get_empty_history(self, client):
        res = client.get("/api/history", params={"workspace": "nonexistent"})
        assert res.status_code == 200
        assert res.json()["count"] == 0


# ---------------------------------------------------------------------------
# SSE streaming (agent event loop → browser)
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal stand-in for the compiled LangGraph — yields custom stream parts."""

    def __init__(self):
        self.input_state = None
        self.config = None

    async def astream(self, input_state, stream_mode=None, config=None):
        self.input_state = input_state
        self.config = config
        yield ("custom", {"kind": "token", "data": "Hello from"})
        yield ("custom", {"kind": "token", "data": " the agent."})
        yield ("custom", {"kind": "sources", "data": ["note1.md"]})
        yield ("values", {"answer": "Hello from the agent."})  # must be ignored


class TestOfflineGuard:
    def test_query_returns_503_with_clear_message(self, client, monkeypatch):
        import rag.graph as graph_mod

        monkeypatch.setattr(graph_mod, "_agent_graph", None)

        def _boom(*args, **kwargs):
            raise ConnectionError(
                "Ollama server unreachable at http://localhost:11434. Start the "
                "Ollama daemon or switch LLM_PROVIDER to a cloud provider."
            )

        monkeypatch.setattr(graph_mod, "get_llm", _boom)
        res = client.post("/api/query", json={"question": "hi", "workspace": "work"})
        assert res.status_code == 503
        assert "Ollama server unreachable" in res.json()["detail"]


class TestQueryStream:
    def test_stream_emits_tokens_sources_and_done(self, client, monkeypatch):
        import rag.graph as graph_mod

        fake = _FakeGraph()
        monkeypatch.setattr(graph_mod, "get_agent_graph", lambda llm=None: fake)

        res = client.post(
            "/api/query/stream",
            json={"question": "hi", "workspace": "work"},
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")

        body = res.text
        # Token chunks are forwarded as {type: token}
        assert '"type": "token"' in body
        assert "Hello from" in body and " the agent." in body
        # Sources are forwarded as {type: source_documents}
        assert '"type": "source_documents"' in body
        assert '"data": ["note1.md"]' in body
        # Generator terminates with a done event
        assert '"type": "done"' in body

        # The graph received the workspace scoped state + thread_id config
        assert fake.input_state["workspace"] == "work"
        assert fake.input_state["question"] == "hi"
        assert fake.config == {"configurable": {"thread_id": "work"}}

    def test_stream_persists_conversation(self, client, monkeypatch):
        import rag.graph as graph_mod

        monkeypatch.setattr(graph_mod, "get_agent_graph", lambda llm=None: _FakeGraph())
        client.post("/api/query/stream", json={"question": "hi", "workspace": "streamtest"})

        res = client.get("/api/history", params={"workspace": "streamtest"})
        assert res.status_code == 200
        data = res.json()
        # The streamed answer ("Hello from the agent.") is saved as the ai turn
        assert data["count"] == 2
        roles = [m["role"] for m in data["messages"]]
        assert roles == ["human", "ai"]
        assert "Hello from" in data["messages"][1]["content"]


class TestNotesCRUD:
    """File-level note CRUD against a throwaway vault (see `notes_client`)."""

    def test_read_note_returns_content(self, notes_client):
        client, _ = notes_client
        res = client.get("/api/notes/work/note1.md")
        assert res.status_code == 200
        data = res.json()
        assert data["title"] == "Note One"
        assert data["workspace"] == "work"
        assert "# Note One" in data["content"]

    def test_read_missing_note_returns_404(self, notes_client):
        client, _ = notes_client
        res = client.get("/api/notes/does/not-exist.md")
        assert res.status_code == 404

    def test_path_escape_is_rejected(self, notes_client):
        client, _ = notes_client
        # `..` traversal must never resolve outside the vault root.
        res = client.get("/api/notes/..%2F..%2Fbackend%2Fmain.py")
        assert res.status_code == 400

    def test_non_utf8_note_returns_400(self, notes_client, tmp_path):
        # Seed a binary/non-UTF-8 file directly inside the throwaway vault.
        client, vault = notes_client
        bad = vault / "bad.md"
        bad.write_bytes(b"\xff\xfe\x00# not utf8")
        res = client.get("/api/notes/bad.md")
        assert res.status_code == 400
        assert "UTF-8" in res.json()["detail"]

    def test_create_roundtrip(self, notes_client):
        client, vault = notes_client
        res = client.post(
            "/api/notes",
            json={"path": "work/new.md", "content": "# New\n\nHello.", "workspace": "work"},
        )
        assert res.status_code == 200
        assert res.json()["path"] == "work/new.md"

        # File physically exists and reads back.
        assert (vault / "work" / "new.md").exists()
        read = client.get("/api/notes/work/new.md")
        assert read.status_code == 200
        assert "# New" in read.json()["content"]

        # Listed among notes.
        listed = client.get("/api/notes").json()
        assert any(n["path"] == "work/new.md" for n in listed["notes"])

    def test_create_existing_returns_409(self, notes_client):
        client, _ = notes_client
        res = client.post(
            "/api/notes",
            json={"path": "work/note1.md", "content": "# dup", "workspace": "work"},
        )
        assert res.status_code == 409

    def test_update_overwrites_and_reindexes(self, notes_client):
        client, vault = notes_client
        upd = client.put(
            "/api/notes/work/note1.md",
            json={"path": "work/note1.md", "content": "# Renamed\n\nChanged.", "workspace": "work"},
        )
        assert upd.status_code == 200
        assert "# Renamed" in upd.json()["content"]
        assert "# Renamed" in (vault / "work" / "note1.md").read_text(encoding="utf-8")

    def test_update_missing_returns_404(self, notes_client):
        client, _ = notes_client
        upd = client.put(
            "/api/notes/work/nope.md",
            json={"path": "work/nope.md", "content": "x", "workspace": "work"},
        )
        assert upd.status_code == 404

    def test_delete_removes_file_and_listing(self, notes_client):
        client, vault = notes_client
        # Seed a note first.
        client.post("/api/notes", json={"path": "work/todelete.md", "content": "# Bye", "workspace": "work"})

        res = client.delete("/api/notes/work/todelete.md")
        assert res.status_code == 200
        assert res.json()["status"] == "deleted"
        assert not (vault / "work" / "todelete.md").exists()

        listed = client.get("/api/notes").json()
        assert not any(n["path"] == "work/todelete.md" for n in listed["notes"])
