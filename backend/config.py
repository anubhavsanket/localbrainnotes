"""Application configuration via pydantic-settings.

Every field maps to an environment variable of the same name (see .env.example).
Defaults are offline-first: Ollama for both the LLM and embeddings.
"""
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ----------------------------------------------------------------
    LLM_PROVIDER: Literal["ollama", "openai", "anthropic", "groq", "nvidia"] = "ollama"
    LLM_MODEL: str = "phi4-mini-localbrain"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 512

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

    # --- Memory -------------------------------------------------------------
    MEMORY_WINDOW_SIZE: int = 10

    # --- Ollama -------------------------------------------------------------
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # --- API keys (optional cloud upgrade path) -----------------------------
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    NVIDIA_API_KEY: Optional[str] = None


settings = Settings()
