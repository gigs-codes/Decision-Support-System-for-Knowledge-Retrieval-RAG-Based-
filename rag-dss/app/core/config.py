"""
Centralised application configuration.
All values can be overridden via environment variables or a .env file.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "RAG Decision Support System"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field("development", pattern="^(development|staging|production)$")
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Security ──────────────────────────────────────────────────────────────
    REQUIRE_API_KEY: bool = False
    API_KEY: str = "change-me-in-production"
    SECRET_KEY: str = "super-secret-jwt-signing-key"

    # ── Rate limiting ─────────────────────────────────────────────────────────
    ENABLE_RATE_LIMIT: bool = True
    RATE_LIMIT_RPM: int = 60  # requests per minute per IP

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # ── LLM (OpenAI-compatible) ───────────────────────────────────────────────
    LLM_PROVIDER: str = Field("openai", pattern="^(openai|anthropic|local)$")
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-5-haiku-20241022"
    LOCAL_LLM_URL: str = "http://localhost:11434/api/generate"
    LOCAL_LLM_MODEL: str = "mistral"

    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT_SECONDS: int = 60

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # sentence-transformers model
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 64

    # ── Vector store (ChromaDB) ───────────────────────────────────────────────
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    CHROMA_COLLECTION: str = "rag_knowledge_base"
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    USE_CHROMA_SERVER: bool = False   # False = embedded mode (great for dev)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35   # cosine similarity floor
    RERANKER_ENABLED: bool = False
    HYBRID_SEARCH_ENABLED: bool = True        # BM25 + dense vector fusion
    HYBRID_ALPHA: float = 0.7                 # weight for dense vs sparse

    # ── Document ingestion ────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512          # tokens per chunk
    CHUNK_OVERLAP: int = 64        # overlap between chunks
    MAX_DOCUMENT_SIZE_MB: int = 20
    SUPPORTED_MIME_TYPES: List[str] = [
        "text/plain",
        "application/pdf",
        "application/json",
        "text/markdown",
        "text/csv",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    # ── Sessions ──────────────────────────────────────────────────────────────
    SESSION_TTL_MINUTES: int = 60
    MAX_HISTORY_TURNS: int = 10

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
