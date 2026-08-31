"""
Cross-encoder re-ranking - the precision stage of retrieval.

A bi-encoder (the embedding model) compresses a chunk into one vector *before* it has ever
seen the question, so ranking by vector distance is inherently coarse. A cross-encoder
reads the question and the chunk together and scores that specific pair, which is far more
accurate and far too slow to run over a whole corpus. Hence the standard shape: retrieve a
wide candidate set cheaply, then re-rank it properly and keep the best few.

The model (~80MB) is loaded lazily on first use so that startup, ingestion, and the test
suite never pay for it when no question has been asked.
"""
import threading
import time
from typing import Dict, List, Optional, Tuple

from src.core.config import RERANK_MODEL
from src.core.logging import get_logger, timed

log = get_logger(__name__)

# How long to wait before retrying a failed model load. A transient failure (no network on
# the very first question, a half-written cache) used to disable re-ranking - the single
# biggest quality stage - for the entire life of the process, silently.
_RETRY_AFTER_SECONDS = 300

_lock = threading.Lock()
_model = None
_next_retry_at = 0.0


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


def available() -> bool:
    return _get_model() is not None


def rerank(question: str, chunks: List[Dict]) -> Optional[List[Tuple[Dict, float]]]:
    """
    Returns [(chunk, score), ...] sorted best first, or None if the re-ranker isn't
    available (caller then keeps its existing order).

    Scores are raw cross-encoder logits: unbounded, with clearly-irrelevant pairs well
    below zero. Compare against config.MIN_RERANK_SCORE, don't read them as probabilities.
    """
    model = _get_model()
    if model is None or not chunks:
        return None

    pairs = [(question, c["text"]) for c in chunks]
    with timed(log, f"re-rank {len(pairs)} candidates"):
        scores = model.predict(pairs)

    ranked = sorted(zip(chunks, (float(s) for s in scores)), key=lambda p: p[1], reverse=True)
    return ranked
