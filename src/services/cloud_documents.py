"""
Registry of documents whose bytes live in Cloudinary (DOCUMENT_STORE=cloudinary).

Local mode's identity for a document is its path relative to DATA_DIR
("users/<id>/book.pdf"), and every isolation, dedup and prune rule in ingestion.py is
written against that path. Rather than invent a second identity scheme for cloud mode,
this registry keeps the exact same shape - each entry's key IS that path - so
owner_from_path(), the per-owner dedupe logic, and prune_deleted() all keep working
completely unchanged. Only where the bytes physically live differs: a Cloudinary URL here,
a local open() call in local mode.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.core.logging import get_logger

log = get_logger(__name__)


def _collection():
    from src.services import database

    return database.sync_collection("cloud_documents")


def _from_doc(doc: Dict) -> Dict:
    doc = dict(doc)
    doc["filename"] = doc.pop("_id")
    return doc


def register(filename: str, *, user_id: str, public_id: str, url: str, size_bytes: int,
            sha256: str) -> None:
    """Records (or updates) one document's Cloudinary location. `filename` is its
    users/<id>/name.pdf identity, matching local mode."""
    _collection().update_one(
        {"_id": filename},
        {"$set": {
            "user_id": user_id,
            "public_id": public_id,
            "url": url,
            "bytes": size_bytes,
            "sha256": sha256,
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }},
        upsert=True,
    )


def get(filename: str) -> Optional[Dict]:
    doc = _collection().find_one({"_id": filename})
    return _from_doc(doc) if doc is not None else None


def remove(filename: str) -> Optional[Dict]:
    """Deletes the registry entry and returns what it was, or None if there wasn't one."""
    doc = get(filename)
    if doc is not None:
        _collection().delete_one({"_id": filename})
    return doc


def list_for_user(user_id: str) -> List[Dict]:
    return sorted((_from_doc(d) for d in _collection().find({"user_id": user_id})),
                 key=lambda d: d["filename"])


def list_all() -> List[Dict]:
    return sorted((_from_doc(d) for d in _collection().find({})), key=lambda d: d["filename"])


def used_bytes(user_id: str) -> int:
    return sum(d.get("bytes", 0) for d in list_for_user(user_id))


def unique_filename(safe_name: str, user_id: str) -> str:
    """
    Same collision-avoidance as local mode's uploads.unique_path(): two uploads named
    "notes.pdf" become "notes.pdf" and "notes (2).pdf" instead of one silently shadowing
    the other in the registry (Cloudinary itself would happily store both under different
    public_ids - this is about the human-readable name staying unique per account).
    """
    existing = {d["filename"].rsplit("/", 1)[-1] for d in list_for_user(user_id)}
    if safe_name not in existing:
        return safe_name
    stem = safe_name[:-4]
    counter = 2
    while f"{stem} ({counter}).pdf" in existing:
        counter += 1
    return f"{stem} ({counter}).pdf"


def clear() -> None:
    """For tests."""
    _collection().delete_many({})
