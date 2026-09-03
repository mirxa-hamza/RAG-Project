"""
Chat sessions: one conversation, its messages, and who owns it.

Stored in MongoDB beside the user accounts, because that is where per-user data already
lives and a session is meaningless without the account it belongs to.

    {
      _id, user_id, title,
      messages: [{role, content, at, sources?}],
      message_count, created_at, updated_at
    }

Two shape decisions worth stating, because they are the ones that would be expensive to
change later:

* **Messages are embedded, not a second collection.** A conversation is read and written as
  a whole, never joined across users, and one round trip beats two. The cost is the 16MB
  document ceiling, which MAX_MESSAGES turns into a bounded, oldest-first trim long before
  Mongo would complain (`$push` with `$slice`).
* **The sidebar never loads messages.** Every listing query projects them away, so opening
  the app with 200 saved conversations transfers titles and timestamps, not a book's worth
  of text. Messages arrive only when a specific session is opened.

ISOLATION. Every query here carries `user_id`, including the ones that already have an
`_id` - a session id is guessable enough that "I know the id" must never be sufficient. A
mismatch reads as "not found", not "forbidden": whether somebody else's session exists is
not information this API gives out.

Pagination is cursor-based on (updated_at, _id), not page numbers. Sessions reorder as they
are used, so "page 2" is a moving target that duplicates and skips rows while you scroll;
"everything older than this exact row" cannot.
"""
import base64
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import (
    MAX_SESSION_MESSAGES,
    MAX_SESSION_TITLE,
    SESSION_PAGE_SIZE,
)
from src.core.logging import get_logger
from src.services import database

log = get_logger(__name__)

# What a listing returns: everything except the messages.
LIST_FIELDS = {"title": 1, "created_at": 1, "updated_at": 1, "message_count": 1}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str):
    """A session id from the wire, or None if it could not possibly be one."""
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def title_from(text: str) -> str:
    """
    A conversation title from its first question.

    Truncation rather than an LLM call: a title is worth roughly nothing and an extra model
    round trip per conversation is worth rather more than nothing - on a free Groq tier it
    is a request you would rather spend on an answer. Markdown and collapsed whitespace are
    cleaned up so the sidebar shows a sentence, not syntax.
    """
    cleaned = (text or "").strip()
    # Strip markdown SYNTAX, not every character that markdown happens to use: a blanket
    # [`*_#>] strip turned "What is A* search?" into "What is A search?", which is a
    # different question.
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", cleaned)   # [text](link)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)            # **bold**
    cleaned = re.sub(r"__(.+?)__", r"\1", cleaned)                # __bold__
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"^[#>\s-]+", "", cleaned)                   # heading / quote markers
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "New chat"
    if len(cleaned) <= MAX_SESSION_TITLE:
        return cleaned

    # Cut at a word boundary when there is one near the limit; a title ending mid-word
    # ("What is the differ…") reads like a rendering bug.
    cut = cleaned[:MAX_SESSION_TITLE]
    space = cut.rfind(" ")
    if space > MAX_SESSION_TITLE * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:") + "…"


def _public(document: Dict[str, Any], with_messages: bool = False) -> Dict[str, Any]:
    """One session in the shape the API returns. `user_id` never leaves the server."""
    out = {
        "id": str(document["_id"]),
        "title": document.get("title") or "New chat",
        "created_at": _iso(document.get("created_at")),
        "updated_at": _iso(document.get("updated_at")),
        "message_count": document.get("message_count", len(document.get("messages") or [])),
    }
    if with_messages:
        out["messages"] = [
            {
                "role": m.get("role"),
                "content": m.get("content", ""),
                "at": _iso(m.get("at")),
                "sources": m.get("sources") or [],
            }
            for m in document.get("messages") or []
        ]
    return out


def _iso(value) -> Optional[str]:
    if isinstance(value, datetime):
        # Mongo stores naive UTC; say so explicitly rather than letting the browser guess.
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value


def encode_cursor(document: Dict[str, Any]) -> str:
    """A resume point: the exact row the last page ended on."""
    stamp = document.get("updated_at")
    stamp = stamp.timestamp() if isinstance(stamp, datetime) else 0.0
    raw = f"{stamp:.6f}|{document['_id']}"
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> Optional[Tuple[datetime, Any]]:
    """The (updated_at, _id) a cursor points at, or None if it is not one of ours."""
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        stamp, _, oid = base64.urlsafe_b64decode(padded.encode()).decode().partition("|")
        parsed = _oid(oid)
        if parsed is None:
            return None
        return datetime.fromtimestamp(float(stamp), tz=timezone.utc), parsed
    except Exception:
        # A malformed cursor is a client bug, not a server error: start from the top.
        return None


async def ensure_indexes() -> None:
    """
    The one index the sidebar needs.

    (user_id, updated_at desc) matches the listing query exactly - filter on the first
    field, sort on the second - so paging through history is an index scan rather than a
    collection scan plus an in-memory sort that Mongo refuses past 32MB.
    """
    await database._guard(
        database.sessions().create_index([("user_id", 1), ("updated_at", -1)])
    )


async def create(user_id: str, title: str = "New chat") -> Dict[str, Any]:
    now = _now()
    document = {
        "user_id": user_id,
        "title": title or "New chat",
        "messages": [],
        "message_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    result = await database._guard(database.sessions().insert_one(document))
    document["_id"] = result.inserted_id
    return _public(document, with_messages=True)


async def list_page(user_id: str, limit: int = SESSION_PAGE_SIZE,
                    cursor: Optional[str] = None) -> Dict[str, Any]:
    """
    One page of a user's sessions, newest first, without their messages.

    Returns {"sessions": [...], "next_cursor": str|None}. A null cursor means there is
    nothing older - which is what stops the browser asking again.
    """
    limit = max(1, min(50, int(limit or SESSION_PAGE_SIZE)))
    query: Dict[str, Any] = {"user_id": user_id}

    after = decode_cursor(cursor or "")
    if after:
        stamp, oid = after
        # Strictly older than the last row seen. The _id tie-break matters: several
        # sessions can share a timestamp to the millisecond, and without it they would be
        # returned again on the next page (duplicates) or skipped entirely.
        query["$or"] = [
            {"updated_at": {"$lt": stamp}},
            {"updated_at": stamp, "_id": {"$lt": oid}},
        ]

    # One extra row, purely to learn whether another page exists without a second query.
    found = await database._guard(
        database.sessions()
        .find(query, LIST_FIELDS)
        .sort([("updated_at", -1), ("_id", -1)])
        .limit(limit + 1)
        .to_list(limit + 1)
    )

    has_more = len(found) > limit
    page = found[:limit]
    return {
        "sessions": [_public(d) for d in page],
        "next_cursor": encode_cursor(page[-1]) if has_more and page else None,
    }


async def get(user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
    """One session with its messages, or None if it is not this user's."""
    oid = _oid(session_id)
    if oid is None:
        return None
    document = await database._guard(
        database.sessions().find_one({"_id": oid, "user_id": user_id})
    )
    return _public(document, with_messages=True) if document else None


async def append_message(user_id: str, session_id: str, role: str, content: str,
                         sources: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
    """
    Appends one message and bumps `updated_at` - which is what re-sorts the sidebar.

    The first user message also names the conversation, in the same update: a separate
    "set the title" call would be a second round trip and a window in which a refresh shows
    "New chat" for a conversation that already has content.
    """
    oid = _oid(session_id)
    if oid is None:
        return None

    now = _now()
    message = {"role": role, "content": content, "at": now}
    if sources:
        message["sources"] = sources

    update: Dict[str, Any] = {
        # $slice keeps the newest MAX_SESSION_MESSAGES and drops the oldest, so a very long
        # conversation stays a bounded document instead of walking towards Mongo's 16MB
        # ceiling, where the failure would be a write error mid-conversation.
        "$push": {"messages": {"$each": [message], "$slice": -MAX_SESSION_MESSAGES}},
        "$set": {"updated_at": now},
        "$inc": {"message_count": 1},
    }

    existing = await database._guard(
        database.sessions().find_one({"_id": oid, "user_id": user_id},
                                     {"title": 1, "message_count": 1})
    )
    if not existing:
        return None
    if role == "user" and (existing.get("message_count") or 0) == 0:
        update["$set"]["title"] = title_from(content)

    await database._guard(
        database.sessions().update_one({"_id": oid, "user_id": user_id}, update)
    )
    return {"id": session_id, "title": update["$set"].get("title", existing.get("title"))}


async def rename(user_id: str, session_id: str, title: str) -> bool:
    oid = _oid(session_id)
    if oid is None:
        return False
    result = await database._guard(database.sessions().update_one(
        {"_id": oid, "user_id": user_id},
        {"$set": {"title": title_from(title), "updated_at": _now()}},
    ))
    return bool(getattr(result, "modified_count", 0) or getattr(result, "matched_count", 0))


async def delete(user_id: str, session_id: str) -> bool:
    oid = _oid(session_id)
    if oid is None:
        return False
    result = await database._guard(
        database.sessions().delete_one({"_id": oid, "user_id": user_id})
    )
    return bool(getattr(result, "deleted_count", 0))


async def delete_all(user_id: str) -> int:
    """Every session of one user - called when the account itself is deleted."""
    result = await database._guard(database.sessions().delete_many({"user_id": user_id}))
    return int(getattr(result, "deleted_count", 0) or 0)
