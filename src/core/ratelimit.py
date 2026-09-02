"""
Rate limiting.

Two backends, chosen by STATE_STORE (see core/config.py):

* **memory** (the default locally). Deliberately not slowapi or Redis: this app is
  single-worker by design (see CLAUDE.md), so a dict and a lock give exactly the same
  guarantee as a Redis token bucket would, with no extra service to run. The moment this
  app runs multiple workers, THIS PATH STOPS BEING CORRECT - each worker would enforce its
  own limit and the effective rate would multiply by the worker count.
* **mongo** (STATE_STORE=mongo, the cloud default). A serverless host gives every cold
  start a fresh, empty process - the in-memory dict above would reset on every request,
  which is not a rate limit at all. One document per event in a Mongo collection, with a
  TTL index so old events expire themselves, gives the same sliding-window guarantee
  shared across every concurrent function instance.

What it protects against:

* **Password guessing.** `/api/login` is otherwise as fast as the network.
* **CPU exhaustion.** Argon2 is deliberately expensive (~50-100ms of dedicated CPU per
  verification), which slows an attacker down and hands them a cheap denial-of-service in
  the same breath. A few hundred concurrent login attempts will saturate the machine.
* **Runaway cost.** Every /chat call spends Groq tokens; every upload spends disk/Cloudinary
  storage and minutes of CPU; every signed upload URL is a stolen slot if handed out freely.

Limits are per (bucket, key), where key is usually the client IP or a user id.
"""
import threading
import time
from typing import Dict, Optional, Tuple

from src.core.config import STATE_STORE
from src.core.logging import get_logger

log = get_logger(__name__)


class RateLimit:
    """One named limit: `allowance` events per `per_seconds`, per key."""

    __slots__ = ("name", "allowance", "per_seconds")

    def __init__(self, name: str, allowance: int, per_seconds: float):
        self.name = name
        self.allowance = allowance
        self.per_seconds = per_seconds


# Tuned for "a person using the app", not for a load test. Login is the strictest because
# it is the one an attacker actually wants.
LOGIN = RateLimit("login", allowance=8, per_seconds=60)
SIGNUP = RateLimit("signup", allowance=5, per_seconds=3600)
UPLOAD = RateLimit("upload", allowance=30, per_seconds=3600)
CHAT = RateLimit("chat", allowance=60, per_seconds=3600)
# Cloud mode only: minting a signed Cloudinary upload URL. Separate from UPLOAD (which
# still governs /upload/complete) because a signature can be requested and abandoned
# cheaply - the real cost is in the completed upload - but an unlimited signing endpoint is
# still a way to hand out upload slots faster than intended.
UPLOAD_SIGN = RateLimit("upload_sign", allowance=60, per_seconds=3600)

_lock = threading.Lock()
# (limit name, key) -> [event timestamps], oldest first.
_events: Dict[Tuple[str, str], list] = {}
_last_sweep = 0.0

# Mongo events carry their own TTL index; kept close to per_seconds so a burst near the
# window edge still expires promptly rather than lingering.
_MONGO_INDEX_READY = False
_mongo_index_lock = threading.Lock()


def _sweep(now: float) -> None:
    """Drops empty buckets so the dict cannot grow forever on a long-running server."""
    global _last_sweep
    if now - _last_sweep < 300:
        return
    _last_sweep = now
    stale = [k for k, stamps in _events.items() if not stamps or now - stamps[-1] > 3600]
    for key in stale:
        _events.pop(key, None)


def _check_memory(limit: RateLimit, key: str) -> Optional[int]:
    """The original in-process sliding window. Unchanged - see the module docstring."""
    now = time.monotonic()
    with _lock:
        _sweep(now)
        bucket = _events.setdefault((limit.name, key), [])

        cutoff = now - limit.per_seconds
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)

        if len(bucket) >= limit.allowance:
            retry_after = int(bucket[0] + limit.per_seconds - now) + 1
            log.warning("Rate limit '%s' hit by %s (retry in %ds)", limit.name, key, retry_after)
            return retry_after

        bucket.append(now)
        return None


def _ensure_mongo_index() -> None:
    """
    Creates the TTL index once per process. Guarded by a flag rather than relying on
    create_index() being cheap to call repeatedly - it still round-trips to the server.
    """
    global _MONGO_INDEX_READY
    if _MONGO_INDEX_READY:
        return
    with _mongo_index_lock:
        if _MONGO_INDEX_READY:
            return
        from src.services import database

        collection = database.sync_collection("rate_limit_events")
        try:
            collection.create_index("expires_at", expireAfterSeconds=0)
            collection.create_index([("bucket", 1), ("key", 1), ("ts", 1)])
        except Exception:
            # A missing index degrades to a slower scan, not a broken limiter - never let
            # index creation itself take the app down.
            log.warning("Could not ensure rate-limit indexes", exc_info=True)
        _MONGO_INDEX_READY = True


def _check_mongo(limit: RateLimit, key: str) -> Optional[int]:
    """
    Sliding window over a Mongo collection: one document per event, shared by every
    process/instance that points at the same database.
    """
    from datetime import datetime, timedelta, timezone

    from src.services import database

    _ensure_mongo_index()
    collection = database.sync_collection("rate_limit_events")

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=limit.per_seconds)

    window = list(
        collection.find(
            {"bucket": limit.name, "key": key, "ts": {"$gte": cutoff}},
            sort=[("ts", 1)],
        )
    )

    if len(window) >= limit.allowance:
        oldest = window[0]["ts"]
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        retry_after = int((oldest + timedelta(seconds=limit.per_seconds) - now).total_seconds()) + 1
        log.warning("Rate limit '%s' hit by %s (retry in %ds)", limit.name, key, max(retry_after, 1))
        return max(retry_after, 1)

    collection.insert_one({
        "bucket": limit.name,
        "key": key,
        "ts": now,
        # expireAfterSeconds=0 on this field means "delete once this timestamp has
        # passed" - a little slack past the window so a borderline read never misses an
        # event that is about to be reaped out from under it.
        "expires_at": now + timedelta(seconds=limit.per_seconds + 60),
    })
    return None


def check(limit: RateLimit, key: str) -> Optional[int]:
    """
    Records an event and returns None if it is allowed, or the seconds to wait if not.

    Sliding window rather than fixed buckets: a fixed window lets an attacker fire the full
    allowance at 59.9s and again at 60.1s. Dispatches to the in-memory or Mongo
    implementation per STATE_STORE - see the module docstring for why callers never need to
    know which.
    """
    if STATE_STORE == "mongo":
        return _check_mongo(limit, key)
    return _check_memory(limit, key)


def reset() -> None:
    """Clears all buckets. For tests only. Only affects the in-memory backend."""
    with _lock:
        _events.clear()


def client_key(request) -> str:
    """
    A key for an unauthenticated caller.

    `request.client.host` is the peer address - which behind a reverse proxy is the proxy,
    making every user share one bucket. `--proxy-headers` (uvicorn) rewrites it from
    X-Forwarded-For, so run it that way if you put a proxy in front; trusting the header
    here directly would let anyone forge a fresh identity per request.
    """
    return request.client.host if request.client else "unknown"
