"""Pytest root setup — ensures backend/ is importable and provides shared fixtures."""
import sys
from pathlib import Path

_BACKEND = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import pytest
import chromadb


@pytest.fixture
def in_memory_client():
    """Return an in-memory ChromaDB client (no disk persistence)."""
    return chromadb.Client()


@pytest.fixture
def sample_vault_path():
    """Path to the sample vault shipped with the project."""
    return str(Path(__file__).resolve().parent.parent / "vaults" / "sample")


@pytest.fixture(autouse=False)
def fake_embeddings(monkeypatch):
    """Monkeypatch embed_documents and embed to return deterministic fake
    vectors so vault-ingest tests don't need a running Ollama."""
    import numpy as np

    _DIM = 16  # small dimension for speed

    def _fake_embed_query(text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % 2**32)
        vec = rng.randn(_DIM).astype("float32")
        return (vec / (np.linalg.norm(vec) + 1e-12)).tolist()

    def _fake_embed_documents(texts: list[str]) -> list[list[float]]:
        return [_fake_embed_query(t) for t in texts]

    import rag.embedder as embedder_mod
    monkeypatch.setattr(embedder_mod, "embed", _fake_embed_query)
    monkeypatch.setattr(embedder_mod, "embed_documents", _fake_embed_documents)
