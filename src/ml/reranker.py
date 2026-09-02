"""
Re-ranking - the precision stage of retrieval.

A bi-encoder (the embedding model) compresses a chunk into one vector *before* it has ever
seen the question, so ranking by vector distance is inherently coarse. A cross-encoder
reads the question and the chunk together and scores that specific pair, which is far more
accurate and far too slow to run over a whole corpus. Hence the standard shape: retrieve a
wide candidate set cheaply, then re-rank it properly and keep the best few.

Two backends, chosen by RERANKER_PROVIDER (which defaults from RAG_MODE):

* **local** - a cross-encoder here, loaded lazily on first use (~80MB) so that startup,
  ingestion and the test suite never pay for it when no question has been asked.
* **pinecone / cohere** - a hosted re-ranker over HTTP, for cloud mode.

The scales are NOT the same. The local model emits unbounded logits (irrelevant pairs sit
well below zero); Cohere emits 0..1 relevance. Callers must therefore compare against
score_floor() rather than a hard-coded constant - using the logit floor on a 0..1 score
keeps every candidate and quietly disables the "not in these documents" answer.

Both backends fail OPEN: if the model cannot load or the API is rate-limited, rerank()
returns None and the caller keeps its fused order. Worse ranking beats a failed question -
but it is invisible, which is why is_available() is on /info.
"""
import threading
import time
from typing import Dict, List, Optional, Tuple

from src.core.config import (
    MIN_RERANK_SCORE,
    MIN_RERANK_SCORE_API,
    RERANK_MODEL,
    RERANKER_PROVIDER,
)
from src.core.logging import get_logger, timed

log = get_logger(__name__)

# How long to wait before retrying a failed model load. A transient failure (no network on
# the very first question, a half-written cache) used to disable re-ranking - the single
# biggest quality stage - for the entire life of the process, silently.
_RETRY_AFTER_SECONDS = 300

_lock = threading.Lock()
_model = None
_next_retry_at = 0.0

_remote = None
_remote_lock = threading.Lock()


def _get_remote():
    """The HTTP re-ranker, built once."""
    global _remote
    if _remote is not None:
        return _remote
    with _remote_lock:
        if _remote is None:
            if RERANKER_PROVIDER == "pinecone":
                from src.ml.providers import PineconeReranker
                _remote = PineconeReranker()
            elif RERANKER_PROVIDER == "cohere":
                from src.ml.providers import CohereReranker
                _remote = CohereReranker()
            else:
                raise ValueError(
                    f"RERANKER_PROVIDER must be local, pinecone or cohere - got "
                    f"{RERANKER_PROVIDER!r}"
                )
    return _remote


def _get_model():
    global _model, _next_retry_at
    if _model is not None:
        return _model
    if time.monotonic() < _next_retry_at:
        return None

    with _lock:
        if _model is None and time.monotonic() >= _next_retry_at:
            try:
                from sentence_transformers import CrossEncoder
                log.info("Loading re-ranker '%s' (first use downloads it)...", RERANK_MODEL)
                _model = CrossEncoder(RERANK_MODEL)
                log.info("Re-ranker loaded.")
            except Exception as exc:
                # No model, no network, or a stubbed sentence_transformers in tests:
                # degrade to fusion-only ranking rather than failing the request, and try
                # again later instead of giving up permanently.
                _next_retry_at = time.monotonic() + _RETRY_AFTER_SECONDS
                log.warning(
                    "Re-ranker unavailable (%s) - using fused ranking; retrying in %ds.",
                    exc, _RETRY_AFTER_SECONDS,
                )
    return _model


def provider_name() -> str:
    return RERANKER_PROVIDER


def score_floor() -> float:
    """
    The minimum score a chunk must reach to be kept, on whichever scale is in use.

    Retrieval calls this instead of reading MIN_RERANK_SCORE directly, because the two
    backends' scores are not comparable.
    """
    return MIN_RERANK_SCORE if RERANKER_PROVIDER == "local" else MIN_RERANK_SCORE_API


def available() -> bool:
    if RERANKER_PROVIDER == "local":
        return _get_model() is not None
    return _get_remote().available()


def is_available() -> bool:
    """
    Whether re-ranking is working right now.

    Exposed on /info because this stage fails OPEN: if it cannot run, retrieval quietly
    falls back to a similarity floor and answers get worse with no error anywhere. That is
    the right runtime behaviour and the wrong thing to leave invisible.
    """
    if RERANKER_PROVIDER == "local":
        return _model is not None
    return _get_remote().available()


def reset_provider() -> None:
    """Drop cached backends. Only for tests that flip the configuration."""
    global _model, _remote, _next_retry_at
    with _lock:
        _model = None
        _next_retry_at = 0.0
    with _remote_lock:
        _remote = None


def rerank(question: str, chunks: List[Dict]) -> Optional[List[Tuple[Dict, float]]]:
    """
    Returns [(chunk, score), ...] sorted best first, or None if re-ranking isn't available
    (caller then keeps its existing order).

    Compare the scores against score_floor(), not against a literal - see the module
    docstring on the two scales.
    """
    if not chunks:
        return None

    if RERANKER_PROVIDER != "local":
        return _get_remote().rerank(question, chunks)

    model = _get_model()
    if model is None:
        return None

    pairs = [(question, c["text"]) for c in chunks]
    with timed(log, f"re-rank {len(pairs)} candidates"):
        scores = model.predict(pairs)

    ranked = sorted(zip(chunks, (float(s) for s in scores)), key=lambda p: p[1], reverse=True)
    return ranked
