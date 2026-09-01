"""
Ownership of documents: the one-off adoption of everything indexed before accounts existed.

The corpus predates authentication. Those chunks carry no `user_id`, which means every
isolation filter excludes them - they would be searchable by nobody and deletable by
nobody, while still occupying the store. Rather than stranding or re-embedding them, the
first account created adopts them.

This runs exactly once, on the first signup, and never again: after it, `manifest.unowned()`
is empty. It is a metadata update, not a re-index - no PDF is re-read and no chunk is
re-embedded.
"""
import json
from typing import List, Optional

from src.core.config import CHROMA_DIR
from src.core.logging import get_logger, timed
from src.services import manifest, vectorstore

log = get_logger(__name__)

# Who owns documents that arrive with no owner in their path - i.e. PDFs copied into
# data/ by hand, or indexed by `python scripts/ingest.py`. Set once, to the first account
# created. Without it, hand-copied files after that first signup would be stamped with
# nobody and be invisible to everybody while still occupying the store.
_OWNER_OF_RECORD_PATH = CHROMA_DIR / "owner_of_record.json"


def owner_of_record() -> Optional[str]:
    try:
        return json.loads(_OWNER_OF_RECORD_PATH.read_text(encoding="utf-8")).get("user_id")
    except (OSError, ValueError):
        return None


def set_owner_of_record(user_id: str) -> None:
    _OWNER_OF_RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OWNER_OF_RECORD_PATH.write_text(json.dumps({"user_id": user_id}), encoding="utf-8")
    log.info("Documents added outside the web UI will belong to user %s", user_id)


def unowned_documents() -> List[str]:
    return manifest.unowned()


def adopt_unowned_documents(user_id: str) -> int:
    """
    Stamps `user_id` onto every ownerless document's chunks and manifest entry.

    Returns how many documents were adopted. Failure on one document is logged and skipped
    rather than raised: a signup must not fail because an old document could not be
    stamped, and the leftovers stay ownerless (invisible, but intact) for a later attempt.
    """
    # Also claim future hand-copied files, not just the ones already indexed.
    if owner_of_record() is None:
        set_owner_of_record(user_id)

    names = unowned_documents()
    if not names:
        return 0

    adopted = 0
    with timed(log, f"adopt {len(names)} pre-auth document(s)"):
        for name in names:
            try:
                stamped = vectorstore.set_owner(name, user_id)
                manifest.set_owner(name, user_id)
                log.info("Adopted '%s' (%d chunks) into user %s", name, stamped, user_id)
                adopted += 1
            except Exception:
                log.exception("Could not adopt '%s'; it stays unowned.", name)
    return adopted
