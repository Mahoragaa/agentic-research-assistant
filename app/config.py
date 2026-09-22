"""
Agentic Research Assistant — Configuration

Type-safe settings loaded from environment variables via pydantic-settings.
Only GOOGLE_API_KEY is required; all other settings have sensible defaults.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from .env / environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Required ────────────────────────────────────────────────────────
    google_api_key: str

    # ── LLM / Embeddings ───────────────────────────────────────────────
    gemini_model: str = "gemini-3.6-flash"
    embedding_model: str = "models/text-embedding-004"

    # ── ChromaDB ───────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"

    # ── Ingestion defaults ─────────────────────────────────────────────
    default_max_papers: int = 25
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── LangSmith (optional) ───────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str | None = None
    langchain_project: str = "agentic-research-assistant"

    # ── Derived helpers ────────────────────────────────────────────────
    @property
    def chroma_path(self) -> Path:
        """Resolved, absolute path for ChromaDB persistence."""
        return Path(self.chroma_persist_dir).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Lazy singleton for application settings.

    Defers validation until first access so imports don't fail
    when GOOGLE_API_KEY isn't set yet (e.g., during tests).
    """
    return Settings()  # type: ignore[call-arg]


class _SettingsProxy:
    """Transparent proxy that lazily initializes Settings on first attribute access."""

    def __getattr__(self, name: str):
        return getattr(get_settings(), name)


# Module-level proxy — import `settings` everywhere; validation
# is deferred until the first attribute is actually read.
settings = _SettingsProxy()  # type: ignore[assignment]
