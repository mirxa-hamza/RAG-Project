"""Liveness and diagnostics."""
import uuid

from fastapi import APIRouter, Response

from src.core.config import (
    COHERE_EMBED_MODEL,
    COHERE_RERANK_MODEL,
    DOCUMENT_STORE,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    JINA_EMBED_MODEL,
    RAG_MODE,
    RERANK_MODEL,
    STATE_STORE,
    VECTOR_STORE,
)
from src.ml import embeddings

router = APIRouter(tags=["system"])

# A new identity for every server start. The browser stores the one it signed in under and
# compares it on load: same id means "this is the same running server, stay signed in";
# a different id means the project was restarted, which is when a fresh sign-in is wanted.
# It is not a secret - it is a random label with nothing derived from it.
BOOT_ID = uuid.uuid4().hex


@router.get("/health")
def health():
    """
    Liveness, plus whether the embedding model has finished loading. The port opens before
    the model is ready (it loads in a warm-up thread), so the UI needs to tell "up but
    still warming" apart from "up and able to answer".
    """
    return {"status": "ok", "embedding_model_ready": embeddings.is_ready(),
            "boot_id": BOOT_ID}


@router.get("/ready")
async def ready(response: Response):
    """
    Readiness, as an orchestrator means it: can this process actually serve a question?

    /health answers as soon as the process is up - which is true but useless to a load
    balancer, because the embedding model takes ~18s to load and Mongo may be down. Routing
    traffic here before both are true produces errors that look like application bugs.
    503 while not ready, so a probe can act on it without parsing the body.
    """
    from src.services import database

    model_ready = embeddings.is_ready()
    try:
        await database.ping()
        db_ready = True
    except Exception:
        db_ready = False

    ok = model_ready and db_ready
    if not ok:
        response.status_code = 503
    return {
        "ready": ok,
        "embedding_model_ready": model_ready,
        "database_ready": db_ready,
    }


@router.get("/api/health/auth")
async def auth_health():
    """
    Whether MongoDB is reachable. The login screen calls this so a failure says "the
    database is down" rather than "incorrect username or password".
    """
    from src.services import database

    try:
        await database.ping()
        return {"database": "ok"}
    except database.DatabaseUnavailable as exc:
        return {"database": "unavailable", "detail": str(exc)}


@router.get("/info")
def info():
    """Which models this instance is actually running - the first thing to check when
    answers look different from what you expected."""
    from src.ml import reranker
    from src.services import answer_cache

    embed_provider = embeddings.provider_name()
    rerank_provider = reranker.provider_name()
    embed_model = {"local": EMBEDDING_MODEL, "cohere": COHERE_EMBED_MODEL,
                   "jina": JINA_EMBED_MODEL}.get(embed_provider, EMBEDDING_MODEL)

    return {
        # Which half of the config is live. A cloud-mode instance answering from local
        # models (or the reverse) is the explanation for most "why are the answers
        # different" questions, and it is otherwise invisible.
        "mode": RAG_MODE,
        "embeddings_provider": embed_provider,
        "reranker_provider": rerank_provider,
        "vector_store": VECTOR_STORE,
        "document_store": DOCUMENT_STORE,
        "state_store": STATE_STORE,
        "embedding_model": embed_model,
        "rerank_model": RERANK_MODEL if rerank_provider == "local" else COHERE_RERANK_MODEL,
        "llm_model": GROQ_MODEL,
        # Re-ranking is the largest single quality stage, and it degrades SILENTLY to a
        # cosine floor if the model cannot load. Without this, the only evidence is one log
        # line at startup and noticeably worse answers weeks later.
        "reranker_available": reranker.is_available(),
        "answer_cache": answer_cache.stats(),
    }
