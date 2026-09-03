"""Application configuration via environment variables.

All settings are typed and validated at startup. Fail fast if required
secrets are missing in non-local environments.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "agentic-rag-assistant"
    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # --- Gemini (Google AI) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    gemini_max_tokens: int = 4096
    # Reasoning depth for Gemini 3: minimal | low | medium | high.
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"

    # --- Embeddings (local, fastembed) ---
    # Runs on CPU in-process: no API key, no rate limit. Weights download once
    # on first use (~67 MB for the default) into embedding_cache_dir.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # MUST match embedding_model's output size and the Pinecone index dimension.
    embedding_dimension: int = 384
    # Kept small on purpose: BGE pads each batch to its longest sequence, so
    # attention memory grows with batch_size x seq_len^2. Large batches raise
    # onnxruntime "bad allocation" on modest machines. Raise only if RAM allows.
    embedding_batch_size: int = 8
    # None lets fastembed pick its default cache location.
    embedding_cache_dir: str | None = None

    # --- Vector store (Pinecone) ---
    pinecone_api_key: str = ""
    # 384-dim index. A 1024-dim index built for a hosted embedder cannot be
    # reused — the dimension is fixed at creation.
    pinecone_index: str = "rag-knowledge-384"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Web search (Tavily) ---
    tavily_api_key: str = ""
    web_search_max_results: int = 5

    # --- Ingestion ---
    max_upload_mb: int = 25
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- Agent ---
    # The planner cannot know whether documents were uploaded, so vector_search
    # always runs and the web is a fallback rather than a peer. Set
    # agent_web_search_is_fallback=False to let the planner pick both up front.
    agent_allow_web_search: bool = True
    agent_web_search_is_fallback: bool = True

    # --- Retrieval ---
    retrieval_top_k: int = 8
    context_token_budget: int = 6000

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
