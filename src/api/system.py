"""Liveness and diagnostics."""
from fastapi import APIRouter

from src.core.config import EMBEDDING_MODEL, GROQ_MODEL, RERANK_MODEL

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/info")
def info():
    """Which models this instance is actually running - the first thing to check when
    answers look different from what you expected."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "llm_model": GROQ_MODEL,
    }
