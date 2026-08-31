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
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.config import MANIFEST_PATH
from app.logging_setup import get_logger

log = get_logger(__name__)


def _load() -> Dict[str, Dict]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        # A corrupt manifest must not take the server down - worst case we re-ingest.
        log.warning("Could not read manifest (%s); treating the store as empty.", exc)
        return {}


def _save(entries: Dict[str, Dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def get(filename: str) -> Optional[Dict]:
    return _load().get(filename)


def put(filename: str, *, sha256: str, mtime: float, size: int, pages: int, chunks: int) -> None:
    entries = _load()
    entries[filename] = {
        "sha256": sha256,
        "mtime": mtime,
        "size": size,
        "pages": pages,
        "chunks": chunks,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save(entries)


def remove(filename: str) -> None:
    entries = _load()
    if entries.pop(filename, None) is not None:
        _save(entries)


def sources() -> List[str]:
    return sorted(_load().keys())


def summary() -> Dict:
    entries = _load()
    return {
        "sources": sorted(entries.keys()),
        "documents": [
            {"filename": name, **{k: rec[k] for k in ("pages", "chunks", "ingested_at") if k in rec}}
            for name, rec in sorted(entries.items())
        ],
    }


def clear() -> None:
    _save({})
