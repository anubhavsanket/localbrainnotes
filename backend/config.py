"""Application configuration via pydantic-settings.

Every field maps to an environment variable of the same name (see .env.example).
Defaults are offline-first: Ollama for both the LLM and embeddings.
"""
from pathlib import Path
from typing import Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (parent of backend/) — anchors every relative file path so the app
# behaves the same regardless of the working directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),  # always the repo-root .env
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Relative file paths are resolved against PROJECT_ROOT, not the CWD.
    @field_validator("VAULT_PATH", "CHROMA_PERSIST_DIR", mode="before")
    @classmethod
    def _anchor_relative_paths(cls, v: str) -> str:
        path = Path(v)
        return str(path) if path.is_absolute() else str(PROJECT_ROOT / path)

    # --- LLM ----------------------------------------------------------------
    LLM_PROVIDER: Literal["ollama", "openai", "anthropic", "groq", "nvidia"] = "ollama"
    LLM_MODEL: str = "phi4-mini-localbrain"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 256

    # --- Embeddings ---------------------------------------------------------
    EMBEDDING_PROVIDER: Literal["ollama", "openai", "huggingface"] = "ollama"
    EMBEDDING_MODEL: str = "nomic-embed-text"

    # --- Vault --------------------------------------------------------------
    VAULT_PATH: str = "./vaults/sample"

    # --- ChromaDB -----------------------------------------------------------
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "localbrain"  # ONE collection, workspace = metadata field

    # --- Chunking -----------------------------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # --- Retrieval ----------------------------------------------------------
    SEARCH_TYPE: Literal["similarity", "mmr"] = "mmr"
    TOP_K: int = 5
    MMR_FETCH_K: int = 15
    SIMILARITY_THRESHOLD: float = 0.70

    # --- Agent loop ---------------------------------------------------------
    # When the router classifies a query as "fastpath", skip the
    # grade→rewrite→reflect cycle and go straight from retrieve→generate.
    FASTPATH_ENABLED: bool = True
    # Groundedness-guard rewrite cap (how many in-place answer repairs allowed).
    GUARD_REPAIR_MAX: int = 1
    # Context compression budget: retrieved context above this many characters
    # is trimmed to the most question-relevant sentences before generation.
    CONTEXT_MAX_CHARS: int = 6000

    # --- Web search tool ----------------------------------------------------
    # Live web search is a dependency-free DuckDuckGo HTML scrape; disable it
    # to force offline behavior ("tool" queries then answer with a clear note).
    WEB_SEARCH_ENABLED: bool = True

    # --- Memory -------------------------------------------------------------
    MEMORY_WINDOW_SIZE: int = 10

    # --- Ollama -------------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # --- API keys (optional cloud upgrade path) -----------------------------
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    NVIDIA_API_KEY: Optional[str] = None

    # --- Evaluation (RAGAS judge) -------------------------------------------
    # Use Ollama by default (offline-first, deterministic, free).
    # Set EVAL_LLM_PROVIDER=openai to use an OpenAI-compatible /v1 endpoint
    # (OpenRouter, NVIDIA NIM, ...): point EVAL_LLM_BASE_URL there and set
    # EVAL_LLM_API_KEY; OPENAI_API_KEY is the fallback.
    EVAL_JUDGE_PROVIDER: Literal["ollama", "openai"] = "ollama"
    EVAL_LLM_MODEL: str = "phi4-mini-localbrain"  # judge model
    EVAL_LLM_BASE_URL: Optional[str] = None
    EVAL_LLM_API_KEY: Optional[str] = None


settings = Settings()
