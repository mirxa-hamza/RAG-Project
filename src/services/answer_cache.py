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

Backend chosen by STATE_STORE (see core/config.py): "memory" is in-process and
single-worker, like the rate limiter - see the note in core/ratelimit.py. "mongo" moves
entries and generation counters into MongoDB, via the same synchronous client the rate
limiter uses, so a cache built by one serverless instance is visible to the next.
"""
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from src.core.config import ANSWER_CACHE_SIZE, ANSWER_CACHE_TTL_SECONDS, STATE_STORE
from src.core.logging import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_entries: "OrderedDict[tuple, dict]" = OrderedDict()
_generation: Dict[str, int] = {}

hits = 0
misses = 0

_MONGO_INDEX_READY = False
_mongo_index_lock = threading.Lock()


def _key(user_id: str, question: str, source: Optional[str], top_k: int) -> tuple:
    # Normalised so "What is A*?" and "  what is a*? " are one entry. Deliberately not
    # stripping punctuation beyond the edges: "not X" and "not X?" mean the same thing,
    # "X" and "X!" might not.
    return (user_id, " ".join(question.lower().split()), source or "", top_k)


def _digest(key: tuple) -> str:
    """A short, stable id for the Mongo document - the key tuple itself can be long."""
    raw = "\x1f".join(str(part) for part in key)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- in-memory backend

def _generation_memory(user_id: str) -> int:
    with _lock:
        return _generation.get(user_id, 0)


def _bump_memory(user_id: Optional[str]) -> None:
    with _lock:
        if user_id is None:
            _generation.clear()
            _entries.clear()
            return
        _generation[user_id] = _generation.get(user_id, 0) + 1


def _get_memory(user_id: str, question: str, source: Optional[str], top_k: int) -> Optional[dict]:
    global hits, misses
    key = _key(user_id, question, source, top_k)
    now = time.monotonic()

    with _lock:
        entry = _entries.get(key)
        if entry is None:
            misses += 1
            return None
        if entry["expires_at"] < now or entry["generation"] != _generation.get(user_id, 0):
            _entries.pop(key, None)
            misses += 1
            return None
        _entries.move_to_end(key)
        hits += 1
        return entry["value"]


def _put_memory(user_id: str, question: str, source: Optional[str], top_k: int, value: dict) -> None:
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


# --------------------------------------------------------------------- mongo backend

def _ensure_mongo_indexes() -> None:
    global _MONGO_INDEX_READY
    if _MONGO_INDEX_READY:
        return
    with _mongo_index_lock:
        if _MONGO_INDEX_READY:
            return
        from src.services import database

        try:
            database.sync_collection("answer_cache_entries").create_index(
                "expires_at", expireAfterSeconds=0
            )
        except Exception:
            log.warning("Could not ensure answer-cache indexes", exc_info=True)
        _MONGO_INDEX_READY = True


def _generation_mongo(user_id: str) -> int:
    from src.services import database

    doc = database.sync_collection("answer_cache_generations").find_one({"_id": user_id})
    return int(doc["gen"]) if doc else 0


def _bump_mongo(user_id: Optional[str]) -> None:
    from src.services import database

    if user_id is None:
        # Store-wide change: every entry is now potentially stale. There is no cheap
        # "increment everyone's generation" in Mongo, so drop the cache outright instead -
        # this path is rare (a full /reset with no owner, the CLI) and correctness matters
        # more than the next few requests being uncached.
        database.sync_collection("answer_cache_entries").delete_many({})
        database.sync_collection("answer_cache_generations").delete_many({})
        return
    database.sync_collection("answer_cache_generations").update_one(
        {"_id": user_id}, {"$inc": {"gen": 1}}, upsert=True
    )


def _get_mongo(user_id: str, question: str, source: Optional[str], top_k: int) -> Optional[dict]:
    global hits, misses
    from datetime import datetime, timezone

    from src.services import database

    _ensure_mongo_indexes()
    key = _key(user_id, question, source, top_k)
    doc = database.sync_collection("answer_cache_entries").find_one({"_id": _digest(key)})
    if doc is None:
        misses += 1
        return None

    expires_at = doc["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc) or doc["generation"] != _generation_mongo(user_id):
        database.sync_collection("answer_cache_entries").delete_one({"_id": doc["_id"]})
        misses += 1
        return None

    hits += 1
    return doc["value"]


def _put_mongo(user_id: str, question: str, source: Optional[str], top_k: int, value: dict) -> None:
    from datetime import datetime, timedelta, timezone

    from src.services import database

    _ensure_mongo_indexes()
    key = _key(user_id, question, source, top_k)
    database.sync_collection("answer_cache_entries").update_one(
        {"_id": _digest(key)},
        {"$set": {
            "value": value,
            "expires_at": datetime.now(timezone.utc) + timedelta(seconds=ANSWER_CACHE_TTL_SECONDS),
            "generation": _generation_mongo(user_id),
            "user_id": user_id,
        }},
        upsert=True,
    )
    # ANSWER_CACHE_SIZE is an in-memory LRU bound; the TTL index is what keeps the Mongo
    # collection from growing without limit instead, which is an intentional trade rather
    # than an oversight - trimming to an exact count needs an extra query on every write.


# --------------------------------------------------------------------- public API

def generation(user_id: str) -> int:
    if STATE_STORE == "mongo":
        return _generation_mongo(user_id)
    return _generation_memory(user_id)


def bump(user_id: Optional[str] = None) -> None:
    """
    Invalidates a user's cached answers - or everyone's, when the change is store-wide.

    Called from ingestion whenever documents are added, replaced or removed.
    """
    if STATE_STORE == "mongo":
        _bump_mongo(user_id)
    else:
        _bump_memory(user_id)
    log.debug("Answer cache invalidated for %s", user_id or "everyone")


def get(user_id: str, question: str, source: Optional[str], top_k: int) -> Optional[dict]:
    """The cached {answer, sources, search_query}, or None."""
    if STATE_STORE == "mongo":
        return _get_mongo(user_id, question, source, top_k)
    return _get_memory(user_id, question, source, top_k)


def put(user_id: str, question: str, source: Optional[str], top_k: int, value: dict) -> None:
    if STATE_STORE == "mongo":
        _put_mongo(user_id, question, source, top_k, value)
    else:
        _put_memory(user_id, question, source, top_k, value)


def stats() -> dict:
    if STATE_STORE == "mongo":
        from src.services import database

        try:
            entries = database.sync_collection("answer_cache_entries").count_documents({})
        except Exception:
            entries = 0
        return {"entries": entries, "hits": hits, "misses": misses}
    with _lock:
        return {"entries": len(_entries), "hits": hits, "misses": misses}


def clear() -> None:
    """For tests."""
    global hits, misses
    with _lock:
        _entries.clear()
        _generation.clear()
        hits = misses = 0
    if STATE_STORE == "mongo":
        from src.services import database

        try:
            database.sync_collection("answer_cache_entries").delete_many({})
            database.sync_collection("answer_cache_generations").delete_many({})
        except Exception:
            pass
