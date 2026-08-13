"""Tests for the FastAPI API layer.

Uses an in-memory ChromaDB and monkeypatches the module-level ``vectorstore``
singleton so no Ollama daemon is required to run these tests.
"""
import pytest
from fastapi.testclient import TestClient

import chromadb
import rag.vectorstore as vs_mod


@pytest.fixture
def client(tmp_path):
    """Build a TestClient backed by an in-memory vector store and a temp db."""
    client = chromadb.Client()
    vs = vs_mod.LocalBrainVectorStore(client=client)
    vs_mod.vectorstore = vs

    # Patch settings for test context
    from config import settings

    original_persist = settings.CHROMA_PERSIST_DIR
    settings.CHROMA_PERSIST_DIR = str(tmp_path)

    from main import app

    with TestClient(app) as c:
        yield c

    settings.CHROMA_PERSIST_DIR = original_persist
    vs_mod.vectorstore = vs  # restore original singleton


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
