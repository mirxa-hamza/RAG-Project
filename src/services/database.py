"""
MongoDB access: the one place that talks to Motor.

User accounts and chat history live in Mongo. Documents and their vectors stay in ChromaDB and on
disk, keyed by the Mongo user id - splitting a document's identity across two databases
would mean a delete could half-succeed.

The client is created lazily on first use rather than at import, for the same reason the
embedding model is: uvicorn imports this module before it binds the port, and Motor's
constructor starts background topology threads. Lazily also means the offline test suite
can substitute a fake collection without a server anywhere.
"""
from typing import Any, Optional

from src.core.config import (
    AUDIT_COLLECTION,
    MONGO_DB,
    MONGO_URI,
    SESSIONS_COLLECTION,
    USERS_COLLECTION,
)
from src.core.logging import get_logger

log = get_logger(__name__)

_client = None
_users = None
_audit = None
_sessions = None


class DatabaseUnavailable(RuntimeError):
    """
    Mongo is not reachable. The message is written to be shown to the user as-is.

    Raised instead of letting pymongo's ServerSelectionTimeoutError escape: that surfaced
    as a 500 with a 90-line traceback, which says "this app is broken" when the truth is
    "the database is not running".
    """


CANNOT_REACH = (
    f"Can't reach MongoDB at {MONGO_URI}. Start it and try again - on Windows: "
    "`net start MongoDB` in an admin terminal, or `mongod --dbpath C:\\data\\db`; "
    "with Docker: `docker run -d -p 27017:27017 --name mongo mongo`."
)


async def _guard(coroutine):
    """
    Runs one database call, turning "the server is not there" into DatabaseUnavailable.

    Only connection-level failures are converted. A DuplicateKeyError still propagates,
    because callers depend on catching it.
    """
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

    try:
        return await coroutine
    except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
        log.error("MongoDB is unreachable: %s", exc)
        raise DatabaseUnavailable(CANNOT_REACH) from exc


def get_client():
    """The Motor client singleton. Constructing it does not connect - the first query does."""
    global _client
    if _client is None:
        # Imported here so `import src.services.database` stays cheap and test doubles can
        # be installed before Motor is ever needed.
        from motor.motor_asyncio import AsyncIOMotorClient

        log.info("Connecting to MongoDB at %s (database '%s')", MONGO_URI, MONGO_DB)
        _client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return _client


def users() -> Any:
    """The users collection. Tests replace this via set_users_collection()."""
    global _users
    if _users is None:
        _users = get_client()[MONGO_DB][USERS_COLLECTION]
    return _users


def audit() -> Any:
    """The audit collection: who did what, when."""
    global _audit
    if _audit is None:
        _audit = get_client()[MONGO_DB][AUDIT_COLLECTION]
    return _audit


def sessions() -> Any:
    """The chat-history collection: one document per conversation."""
    global _sessions
    if _sessions is None:
        _sessions = get_client()[MONGO_DB][SESSIONS_COLLECTION]
    return _sessions


def set_users_collection(collection: Any, audit_collection: Any = None,
                         sessions_collection: Any = None) -> None:
    """Injection point for tests: swap in fake collections, no server needed."""
    global _users, _audit, _sessions
    _users = collection
    if audit_collection is not None:
        _audit = audit_collection
    if sessions_collection is not None:
        _sessions = sessions_collection


async def ensure_indexes() -> None:
    """
    Creates the unique index on `username`.

    This is what actually enforces uniqueness. A "does this username exist?" check in the
    signup handler is a race: two simultaneous requests both read "no" and both insert.
    The index makes the second insert fail, and the handler turns that into a 409.
    """
    await _guard(users().create_index("username", unique=True))


async def ping() -> None:
    """Raises DatabaseUnavailable with an actionable message if Mongo cannot be reached."""
    try:
        await get_client().admin.command("ping")
    except Exception as exc:
        raise DatabaseUnavailable(f"{CANNOT_REACH} ({type(exc).__name__})") from exc


async def find_user_by_username(username: str) -> Optional[dict]:
    return await _guard(users().find_one({"username": username}))


async def find_user_by_id(user_id: str) -> Optional[dict]:
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        oid = ObjectId(user_id)
    except (InvalidId, TypeError):
        # A token carrying a malformed id is a rejected login, not a server error.
        return None
    return await _guard(users().find_one({"_id": oid}))


async def create_user(username: str, password_hash: str, name: Optional[str] = None) -> dict:
    """
    Inserts a user. Raises DuplicateKeyError (from pymongo) if the username is taken -
    the caller turns that into a 409.
    """
    from datetime import datetime, timezone

    document = {
        "username": username,
        "password_hash": password_hash,
        "name": name,
        # Bumped on password change and on "sign out everywhere". Every token carries the
        # version it was minted with, and a mismatch is rejected - which is the only way to
        # invalidate a JWT before it expires.
        "token_version": 1,
        "created_at": datetime.now(timezone.utc),
    }
    result = await _guard(users().insert_one(document))
    document["_id"] = result.inserted_id
    return document


async def set_password(user_id: str, password_hash: str) -> None:
    """Changes the password AND invalidates every existing token for that account."""
    from bson import ObjectId

    await _guard(users().update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password_hash": password_hash},
         "$inc": {"token_version": 1}},
    ))


async def bump_token_version(user_id: str) -> None:
    """Signs the account out everywhere, without changing the password."""
    from bson import ObjectId

    await _guard(users().update_one({"_id": ObjectId(user_id)}, {"$inc": {"token_version": 1}}))


async def delete_user(user_id: str) -> None:
    from bson import ObjectId

    await _guard(users().delete_one({"_id": ObjectId(user_id)}))


async def record_audit(user_id: Optional[str], username: Optional[str], action: str,
                       detail: Optional[str] = None, ok: bool = True) -> None:
    """
    Appends one line to the audit trail.

    Never raises: an audit write that fails must not turn a successful login into a 500.
    A missing audit line is a gap in the record; a failed request is a broken app.
    """
    from datetime import datetime, timezone

    try:
        await audit().insert_one({
            "user_id": user_id,
            "username": username,
            "action": action,
            "detail": detail,
            "ok": ok,
            "at": datetime.now(timezone.utc),
        })
    except Exception:
        log.warning("Could not write the audit entry for %r", action, exc_info=True)


async def count_users() -> int:
    return await _guard(users().count_documents({}))


def close() -> None:
    global _client, _users, _sessions
    if _client is not None:
        _client.close()
    _client = None
    _users = None
    _sessions = None


# --- Synchronous access -----------------------------------------------------------------
# Motor is async, but the manifest, the ownership record, the rate limiter and the answer
# cache are all called from synchronous code (including the ingestion thread), where there
# is no event loop to await on. pymongo's blocking client is the honest answer there;
# sharing one keeps it to a single connection pool.
_sync_client = None
_sync_collections: dict = {}


def sync_collection(name: str) -> Any:
    """
    A blocking pymongo collection by name, for callers that are not async.

    Tests replace individual collections through set_sync_collection().
    """
    global _sync_client
    if name in _sync_collections:
        return _sync_collections[name]
    if _sync_client is None:
        from pymongo import MongoClient

        log.info("Opening a synchronous MongoDB connection to %s (database '%s')",
                 MONGO_URI, MONGO_DB)
        _sync_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    _sync_collections[name] = _sync_client[MONGO_DB][name]
    return _sync_collections[name]


def set_sync_collection(name: str, collection: Any) -> None:
    """Injection point for tests: no server needed."""
    _sync_collections[name] = collection
