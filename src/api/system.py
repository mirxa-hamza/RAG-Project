"""Liveness and diagnostics."""
from fastapi import APIRouter

from src.core.config import EMBEDDING_MODEL, GROQ_MODEL, RERANK_MODEL
from src.ml import embeddings

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    """
    Liveness, plus whether the embedding model has finished loading. The port opens before
    the model is ready (it loads in a warm-up thread), so the UI needs to tell "up but
    still warming" apart from "up and able to answer".
    """
    return {"status": "ok", "embedding_model_ready": embeddings.is_ready()}


@router.get("/info")
def info():
    """Which models this instance is actually running - the first thing to check when
    answers look different from what you expected."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "llm_model": GROQ_MODEL,
    }
