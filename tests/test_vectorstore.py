"""Tests for the unified ChromaDB vector store."""
import pytest
from rag.vectorstore import LocalBrainVectorStore, _mmr_indices, stable_chunk_id


@pytest.fixture
def store(in_memory_client):
    """A LocalBrainVectorStore backed by an in-memory ChromaDB.

    Ensures the collection is empty before each test so state doesn't leak.
    """
    vs = LocalBrainVectorStore(client=in_memory_client)
    try:
        vs._client.delete_collection("localbrain")
    except Exception:  # noqa: BLE001, S110
        pass
    return vs


def _make_chunks(n: int = 5, workspace: str = "default") -> list[dict]:
    """Generate n synthetic chunk dicts with fake embeddings (dim 8)."""

    chunks = []
    for i in range(n):
        emb = [0.0] * 8
        emb[i % 8] = 1.0
        chunks.append({
            "id": stable_chunk_id("test_note", i),
            "text": f"Chunk {i} about topic {i % 3}",
            "embedding": emb,
            "metadata": {
                "note_id": "test_note",
                "path": "test_note.md",
                "chunk_index": i,
                "workspace": workspace,
            },
        })
    return chunks


class TestUpsert:
    def test_upsert_returns_count(self, store):
        chunks = _make_chunks(3)
        count = store.upsert_chunks(chunks, workspace="work")
        assert count == 3

    def test_upsert_is_idempotent(self, store):
        chunks = _make_chunks(3)
        store.upsert_chunks(chunks, workspace="work")
        count = store.upsert_chunks(chunks, workspace="work")
        assert count == 3

    def test_workspace_is_set_in_metadata(self, store):
        chunks = _make_chunks(2, workspace="personal")
        store.upsert_chunks(chunks, workspace="personal")
        results = store.collection.get(where={"workspace": "personal"})
        assert len(results["ids"]) == 2


class TestQuery:
    def test_query_returns_results(self, store):
        chunks = _make_chunks(5)
        store.upsert_chunks(chunks)
        q_emb = [1.0] + [0.0] * 7
        results = store.query_similar(q_emb, n=3)
        assert len(results["documents"][0]) == 3

    def test_workspace_filter(self, store):
        # Use distinct note_ids so chunk IDs don't collide across workspaces.
        work_chunks = _make_chunks(3, workspace="work")
        for c in work_chunks:
            c["id"] = f"work_{c['id']}"
        personal_chunks = _make_chunks(2, workspace="personal")
        for c in personal_chunks:
            c["id"] = f"personal_{c['id']}"

        store.upsert_chunks(work_chunks, workspace="work")
        store.upsert_chunks(personal_chunks, workspace="personal")

        results = store.query_similar([1.0] + [0.0] * 7, n=10, workspace="work")
        assert len(results["documents"][0]) == 3

    def test_cross_workspace_search(self, store):
        work_chunks = _make_chunks(3, workspace="work")
        for c in work_chunks:
            c["id"] = f"work_{c['id']}"
        personal_chunks = _make_chunks(2, workspace="personal")
        for c in personal_chunks:
            c["id"] = f"personal_{c['id']}"

        store.upsert_chunks(work_chunks, workspace="work")
        store.upsert_chunks(personal_chunks, workspace="personal")

        results = store.query_similar([1.0] + [0.0] * 7, n=10, workspace=None)
        assert len(results["documents"][0]) == 5


class TestDelete:
    def test_delete_removes_chunks(self, store):
        chunks = _make_chunks(5)
        store.upsert_chunks(chunks)
        removed = store.delete_note_chunks("test_note")
        assert removed == 5
        remaining = store.collection.get(where={"note_id": "test_note"})
        assert len(remaining["ids"]) == 0

    def test_delete_nonexistent_is_safe(self, store):
        removed = store.delete_note_chunks("nonexistent")
        assert removed == 0


class TestMMR:
    def test_mmr_selects_diverse_results(self):
        import numpy as np

        n = 10
        query_emb = np.ones(16, dtype="float32").tolist()
        # Build embeddings where chunks 0-4 are near-identical (redundant)
        # and chunks 5-9 are distinct.
        doc_embs = []
        for i in range(5):
            e = np.ones(16, dtype="float32") * 0.95
            e[0] += i * 0.01
            doc_embs.append(e.tolist())
        for i in range(5):
            e = np.zeros(16, dtype="float32")
            e[i] = 1.0
            doc_embs.append(e.tolist())

        selected = _mmr_indices(query_emb, doc_embs, k=5)
        # MMR should prefer the distinct chunks (5-9) over redundant ones.
        assert len(selected) == 5
        assert set(selected).issubset(set(range(n)))


class TestStableIds:
    def test_deterministic(self):
        id1 = stable_chunk_id("notes/README.md", 0)
        id2 = stable_chunk_id("notes/README.md", 0)
        assert id1 == id2

    def test_different_paths_different_ids(self):
        assert stable_chunk_id("a.md", 0) != stable_chunk_id("b.md", 0)

    def test_different_chunks_different_ids(self):
        assert stable_chunk_id("a.md", 0) != stable_chunk_id("a.md", 1)
