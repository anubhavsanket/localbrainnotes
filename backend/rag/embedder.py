"""Unified embedding layer with a lightweight local-file cache.

Default is Ollama's ``nomic-embed-text`` (offline-first). The file-backed
cache means identical text is only embedded once per cache on disk, which
drops re-embedding cost on repeated ingestion sessions.

The cache is a simple JSON file mapping text → serialized embedding vector,
avoiding any dependency on langchain's internal cache module (which was
removed in langchain 1.3.x).
"""
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_core.embeddings import Embeddings

from config import settings, PROJECT_ROOT

# Cache lives under the repo root so it is found regardless of the CWD the
# server is launched from (README quick-start runs `python main.py` from backend/).
EMBEDDING_CACHE_DIR = str(PROJECT_ROOT / "cache" / "embeddings")
EMBED_CACHE_FILE = str(PROJECT_ROOT / "cache" / "embeddings.json")


# ---------------------------------------------------------------------------
# Self-contained embedding cache
# ---------------------------------------------------------------------------

class CachedEmbeddings(Embeddings):
    """Thin wrapper that checks a JSON file cache before calling the
    underlying embedding model."""

    def __init__(self, underlying: Embeddings, cache_path: str = EMBED_CACHE_FILE):
        self.underlying = underlying
        self._cache_path = Path(cache_path)
        self._cache: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _save(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, separators=(",", ":")),
            encoding="utf-8",
        )

    def embed_query(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]
        emb = self.underlying.embed_query(text)
        self._cache[text] = emb
        self._save()
        return emb

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            new_embs = self.underlying.embed_documents(uncached_texts)
            for idx, emb in zip(uncached_indices, new_embs):
                self._cache[texts[idx]] = emb
                results[idx] = emb
            self._save()

        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------

def _build_underlying_embeddings() -> Embeddings:
    """Instantiate the raw (uncached) embedding model for the configured provider."""
    provider = settings.EMBEDDING_PROVIDER

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI embeddings")
        return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, openai_api_key=key)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    raise ValueError(f"Unsupported embedding provider: {provider}")


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a process-wide, cache-backed Embeddings object (one per run)."""
    underlying = _build_underlying_embeddings()
    return CachedEmbeddings(underlying, cache_path=EMBED_CACHE_FILE)


def embed(text: str) -> list[float]:
    """Embed a single query string."""
    return get_embeddings().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document strings (used during ingestion)."""
    return get_embeddings().embed_documents(texts)


def _clear_cache() -> None:
    """Test helper: drop the cached Embeddings instance so provider/config
    changes take effect in a fresh process-free way."""
    get_embeddings.cache_clear()
