"""
A small JSON record of what's been ingested.

Why this exists: the obvious way to answer "which documents are in the store?" is
`collection.get(include=["metadatas"])`, which pulls the metadata of *every chunk* -
thousands of dicts - and that call sat on the /stats endpoint, on every frontend page
load, and on every ingest run. This file answers the same question in a few hundred bytes,
and carries the fingerprints needed to notice when a PDF on disk has changed.

Lives inside CHROMA_DIR so the index and its manifest are wiped together.
"""
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.core.config import MANIFEST_PATH
from src.core.logging import get_logger

log = get_logger(__name__)

# The ingest thread writes while request threads read. Every read-modify-write below runs
# under this lock so two updates can't clobber each other.
_lock = threading.RLock()


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
        # Never leave the temp file behind if the replace failed.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def get(filename: str) -> Optional[Dict]:
    with _lock:
        return _load().get(filename)


def put(filename: str, *, sha256: str, mtime: float, size: int, pages: int, chunks: int,
        user_id: Optional[str] = None) -> None:
    with _lock:
        entries = _load()
        entries[filename] = {
            "sha256": sha256,
            "mtime": mtime,
            "size": size,
            "pages": pages,
            "chunks": chunks,
            # The owner, mirrored from the chunk metadata so /stats and the document list
            # can answer "whose is this?" without querying Chroma. Absent for documents
            # ingested before accounts existed, and for the CLI.
            **({"user_id": user_id} if user_id else {}),
            "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _save(entries)


def set_owner(filename: str, user_id: str) -> bool:
    """Stamps an owner onto an existing entry. Returns False if there is no such entry."""
    with _lock:
        entries = _load()
        record = entries.get(filename)
        if record is None:
            return False
        record["user_id"] = user_id
        _save(entries)
        return True


def owner_of(filename: str) -> Optional[str]:
    with _lock:
        record = _load().get(filename)
    return record.get("user_id") if record else None


def remove(filename: str) -> None:
    with _lock:
        entries = _load()
        if entries.pop(filename, None) is not None:
            _save(entries)


def sources(user_id: Optional[str] = None) -> List[str]:
    """Every ingested document, or only `user_id`'s when one is given."""
    with _lock:
        entries = _load()
    if user_id is None:
        return sorted(entries.keys())
    return sorted(name for name, rec in entries.items() if rec.get("user_id") == user_id)


def unowned() -> List[str]:
    """Documents with no owner - i.e. ingested before accounts existed, or by the CLI."""
    with _lock:
        entries = _load()
    return sorted(name for name, rec in entries.items() if not rec.get("user_id"))


def summary(user_id: Optional[str] = None) -> Dict:
    """
    What /stats reports. With a user_id, only that user's documents are described - the
    counts and filenames of anyone else's must never reach a response.
    """
    with _lock:
        entries = _load()
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
    with _lock:
        _save({})
