"""
A small record of what's been ingested: filename -> {sha256, pages, chunks, owner, ...}.

Why this exists: the obvious way to answer "which documents are in the store?" is
`collection.get(include=["metadatas"])`, which pulls the metadata of *every chunk* -
thousands of dicts - and that call sat on the /stats endpoint, on every frontend page
load, and on every ingest run. This file answers the same question in a few hundred bytes,
and carries the fingerprints needed to notice when a PDF has changed.

Backend chosen by STATE_STORE (see core/config.py):

* **memory-mode is actually disk-backed**: a JSON file inside CHROMA_DIR, written
  atomically (temp file + os.replace) under a lock, so the index and its manifest are wiped
  together and a reader never observes a half-written file. This is the original design and
  is unchanged here.
* **mongo**: one document per filename in a `document_manifest` collection. Required for
  cloud mode for the same reason job state and rate limits are - a serverless function has
  no persistent disk, so a JSON sidecar under CHROMA_DIR would be empty again on every cold
  start, and the whole "already ingested, skip it" logic would silently stop working (every
  request would look like a first ingest).
"""
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.core.config import MANIFEST_PATH, STATE_STORE
from src.core.logging import get_logger

log = get_logger(__name__)

# The ingest thread writes while request threads read. Every read-modify-write below runs
# under this lock so two updates can't clobber each other. (Mongo mode doesn't need the
# lock for correctness - each write is already one atomic document operation - but keeping
# it means the same code above never has to know which mode it's in.)
_lock = threading.RLock()

_MONGO = STATE_STORE == "mongo"


# --------------------------------------------------------------------- disk (JSON) backend

def _load() -> Dict[str, Dict]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        # A corrupt manifest must not take the server down - worst case we re-ingest.
        log.warning("Could not read manifest (%s); treating the store as empty.", exc)
        return {}
    return data if isinstance(data, dict) else {}


def _save(entries: Dict[str, Dict]) -> None:
    """
    Atomic write: serialise to a temp file in the same directory, then os.replace().

    A plain write_text() leaves a window where the file on disk is half-written. A reader
    landing in that window gets a JSONDecodeError, falls back to "{}", and the ingester
    concludes nothing has ever been stored - re-embedding the entire corpus.
    """
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entries, indent=2)

    fd, tmp_path = tempfile.mkstemp(dir=str(MANIFEST_PATH.parent), prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, MANIFEST_PATH)  # atomic on POSIX and on Windows
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------- mongo backend

def _collection():
    from src.services import database

    return database.sync_collection("document_manifest")


def _mongo_get(filename: str) -> Optional[Dict]:
    doc = _collection().find_one({"_id": filename})
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


def _mongo_put(filename: str, record: Dict) -> None:
    _collection().update_one({"_id": filename}, {"$set": record}, upsert=True)


def _mongo_remove(filename: str) -> None:
    _collection().delete_one({"_id": filename})


def _mongo_all() -> Dict[str, Dict]:
    out = {}
    for doc in _collection().find({}):
        doc = dict(doc)
        filename = doc.pop("_id")
        out[filename] = doc
    return out


# --------------------------------------------------------------------- public API

def get(filename: str) -> Optional[Dict]:
    if _MONGO:
        return _mongo_get(filename)
    with _lock:
        return _load().get(filename)


def put(filename: str, *, sha256: str, mtime: float, size: int, pages: int, chunks: int,
        user_id: Optional[str] = None) -> None:
    record = {
        "sha256": sha256,
        "mtime": mtime,
        "size": size,
        "pages": pages,
        "chunks": chunks,
        **({"user_id": user_id} if user_id else {}),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if _MONGO:
        _mongo_put(filename, record)
        return
    with _lock:
        entries = _load()
        entries[filename] = record
        _save(entries)


def set_owner(filename: str, user_id: str) -> bool:
    """Stamps an owner onto an existing entry. Returns False if there is no such entry."""
    if _MONGO:
        result = _collection().update_one({"_id": filename}, {"$set": {"user_id": user_id}})
        return result.matched_count > 0
    with _lock:
        entries = _load()
        record = entries.get(filename)
        if record is None:
            return False
        record["user_id"] = user_id
        _save(entries)
        return True


def owner_of(filename: str) -> Optional[str]:
    record = get(filename)
    return record.get("user_id") if record else None


def remove(filename: str) -> None:
    if _MONGO:
        _mongo_remove(filename)
        return
    with _lock:
        entries = _load()
        if entries.pop(filename, None) is not None:
            _save(entries)


def sources(user_id: Optional[str] = None) -> List[str]:
    """Every ingested document, or only `user_id`'s when one is given."""
    entries = _mongo_all() if _MONGO else _load_locked()
    if user_id is None:
        return sorted(entries.keys())
    return sorted(name for name, rec in entries.items() if rec.get("user_id") == user_id)


def unowned() -> List[str]:
    """Documents with no owner - i.e. ingested before accounts existed, or by the CLI."""
    entries = _mongo_all() if _MONGO else _load_locked()
    return sorted(name for name, rec in entries.items() if not rec.get("user_id"))


def summary(user_id: Optional[str] = None) -> Dict:
    """
    What /stats reports. With a user_id, only that user's documents are described - the
    counts and filenames of anyone else's must never reach a response.
    """
    entries = _mongo_all() if _MONGO else _load_locked()
    if user_id is not None:
        entries = {n: r for n, r in entries.items() if r.get("user_id") == user_id}
    return {
        "sources": sorted(entries.keys()),
        "documents": [
            {"filename": name,
             **{k: rec[k] for k in ("pages", "chunks", "size", "ingested_at") if k in rec}}
            for name, rec in sorted(entries.items())
        ],
    }


def clear() -> None:
    if _MONGO:
        _collection().delete_many({})
        return
    with _lock:
        _save({})


def _load_locked() -> Dict[str, Dict]:
    with _lock:
        return _load()
