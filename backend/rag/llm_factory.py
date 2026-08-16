"""Multi-provider LLM factory.

Default is Ollama (offline-first).  Each provider returns a LangChain chat
model; the ``get_llm`` factory resolves a provider from settings (or an
explicit override) and raises a clear error if Ollama is unreachable when
that provider is selected.
"""
import os
from typing import Optional

from langchain_core.language_models import BaseChatModel

from config import settings


def _ollama_alive(base_url: str) -> bool:
    """Quick liveness probe — avoids an opaque timeout deep in the HTTP client."""
    import urllib.request

    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """Build a LangChain chat model for the requested provider.

    All parameters fall back to ``settings`` when not explicitly provided.
    """
    provider = provider or settings.LLM_PROVIDER
    model = model or settings.LLM_MODEL
    temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
    max_tokens = max_tokens or settings.LLM_MAX_TOKENS

    if provider == "ollama":
        if not _ollama_alive(settings.OLLAMA_BASE_URL):
            raise ConnectionError(
                f"Ollama server unreachable at {settings.OLLAMA_BASE_URL}. "
                "Start the Ollama daemon or switch LLM_PROVIDER to a cloud provider."
            )
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,  # cap per-call output so verbose local
            # models can't run away and tie up Ollama's single slot
            base_url=settings.OLLAMA_BASE_URL,
            **kwargs,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        key = api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAI provider")
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=key,
            **kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        key = api_key or settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY must be set for Anthropic provider")
        return ChatAnthropic(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=key,
            **kwargs,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY must be set for Groq provider")
        return ChatGroq(
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=key,
            **kwargs,
        )

    if provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        key = api_key or settings.NVIDIA_API_KEY or os.getenv("NVIDIA_API_KEY")
        if not key:
            raise ValueError("NVIDIA_API_KEY must be set for NVIDIA provider")
        return ChatNVIDIA(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=key,
            **kwargs,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
