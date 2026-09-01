"""
A small answer cache.

Real users repeat themselves - the same question from two tabs, a refresh, a demo shown
twice - and every repeat currently costs an embedding, a BM25 pass, a cross-encoder pass
and a paid LLM call. Caching the finished answer removes all four.

Three properties make it safe rather than merely fast:

* **The user id is part of the key.** Two people asking the same words must never share an
  entry; that would defeat the whole isolation story.
* **Entries die when the user's documents change.** Each user has a generation counter that
  ingestion bumps; an entry minted under an older generation is ignored. Without it, the
  answer to "what does chapter 3 say" would survive the deletion of chapter 3.
* **Entries expire.** A TTL bounds how stale a cached answer can be even if a generation
  bump is ever missed.

In-process and single-worker, like the rate limiter - see the note in core/ratelimit.py.
"""
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from src.core.config import ANSWER_CACHE_SIZE, ANSWER_CACHE_TTL_SECONDS
from src.core.logging import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_entries: "OrderedDict[tuple, dict]" = OrderedDict()
_generation: Dict[str, int] = {}

hits = 0
misses = 0


def generation(user_id: str) -> int:
    with _lock:
        return _generation.get(user_id, 0)


def bump(user_id: Optional[str] = None) -> None:
    """
    Invalidates a user's cached answers - or everyone's, when the change is store-wide.

    Called from ingestion whenever documents are added, replaced or removed.
    """
    with _lock:
        if user_id is None:
            _generation.clear()
            _entries.clear()
            return
        _generation[user_id] = _generation.get(user_id, 0) + 1
    log.debug("Answer cache invalidated for %s", user_id)


def _key(user_id: str, question: str, source: Optional[str], top_k: int) -> tuple:
    # Normalised so "What is A*?" and "  what is a*? " are one entry. Deliberately not
    # stripping punctuation beyond the edges: "not X" and "not X?" mean the same thing,
    # "X" and "X!" might not.
    return (user_id, " ".join(question.lower().split()), source or "", top_k)


def get(user_id: str, question: str, source: Optional[str], top_k: int) -> Optional[dict]:
    """The cached {answer, sources, search_query}, or None."""
    global hits, misses
    key = _key(user_id, question, source, top_k)
    now = time.monotonic()

    with _lock:
        entry = _entries.get(key)
        if entry is None:
            misses += 1
            return None
        if entry["expires_at"] < now or entry["generation"] != _generation.get(user_id, 0):
            # Stale by time, or the user's documents changed since it was stored.
            _entries.pop(key, None)
            misses += 1
            return None
        _entries.move_to_end(key)
        hits += 1
        return entry["value"]


def put(user_id: str, question: str, source: Optional[str], top_k: int, value: dict) -> None:
    key = _key(user_id, question, source, top_k)
    with _lock:
        _entries[key] = {
            "value": value,
            "expires_at": time.monotonic() + ANSWER_CACHE_TTL_SECONDS,
            "generation": _generation.get(user_id, 0),
        }
        _entries.move_to_end(key)
        while len(_entries) > ANSWER_CACHE_SIZE:
            _entries.popitem(last=False)


def stats() -> dict:
    with _lock:
        return {"entries": len(_entries), "hits": hits, "misses": misses}


def clear() -> None:
    """For tests."""
    global hits, misses
    with _lock:
        _entries.clear()
        _generation.clear()
        hits = misses = 0
