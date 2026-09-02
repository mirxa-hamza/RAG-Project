"""
MongoDB access: the one place that talks to Motor.

Only user accounts live in Mongo. Documents and their vectors stay in ChromaDB and on
disk, keyed by the Mongo user id - splitting a document's identity across two databases
would mean a delete could half-succeed.

The client is created lazily on first use rather than at import, for the same reason the
embedding model is: uvicorn imports this module before it binds the port, and Motor's
constructor starts background topology threads. Lazily also means the offline test suite
can substitute a fake collection without a server anywhere.
"""
import re
from typing import Any, Optional

from src.core.config import AUDIT_COLLECTION, MONGO_DB, MONGO_URI, USERS_COLLECTION
from src.core.logging import get_logger

log = get_logger(__name__)


def _redact(uri: str) -> str:
    """
    "mongodb+srv://user:hunter2@cluster.mongodb.net" -> "mongodb+srv://user:***@cluster.mongodb.net"

    A hosted connection string carries the password inline, and this URI is both LOGGED on
    every connection and embedded in CANNOT_REACH below, which is shown to the user as-is
    on the sign-in screen. Unredacted, that means the database password is printed into the
    platform's log stream on every cold start, and rendered in a browser to anyone who
    loads the page during an outage. Neither is recoverable after the fact - the only fix
    once it has happened is rotating the credential.
    """
    return re.sub(r"(//[^:/?#@]+:)[^@]+@", r"\1***@", uri)


_client = None
_users = None
_audit = None
_sync_client = None
_sync_collections: dict = {}


class DatabaseUnavailable(RuntimeError):
    """
    Mongo is not reachable. The message is written to be shown to the user as-is.

    Raised instead of letting pymongo's ServerSelectionTimeoutError escape: that surfaced
    as a 500 with a 90-line traceback, which says "this app is broken" when the truth is
    "the database is not running".
    """


CANNOT_REACH = (
    f"Can't reach MongoDB at {_redact(MONGO_URI)}. Start it and try again - on Windows: "
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

        log.info("Connecting to MongoDB at %s (database '%s')", _redact(MONGO_URI), MONGO_DB)
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


def set_users_collection(collection: Any, audit_collection: Any = None) -> None:
    """Injection point for tests: swap in fake collections, no server needed."""
    global _users, _audit
    _users = collection
    if audit_collection is not None:
        _audit = audit_collection


def get_sync_client():
    """
    A synchronous (pymongo) client for the small pieces of STATE_STORE=mongo state - the
    rate limiter, the answer cache, the ingestion job, and the document manifest - that are
    called from plain synchronous code (the local background-ingestion thread has no event
    loop) as well as from async request handlers. See the note in core/config.py.

    Lazy for the same reason as get_client(): constructing it opens a background topology
    thread, and this module must stay cheap to import.
    """
    global _sync_client
    if _sync_client is None:
        from pymongo import MongoClient

        log.info("Connecting to MongoDB (sync client) at %s (database '%s')", _redact(MONGO_URI), MONGO_DB)
        _sync_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return _sync_client


def sync_collection(name: str) -> Any:
    """
    A collection reached through the synchronous client, by name. Tests replace one via
    set_sync_collection() - a plain in-memory fake, no server required, mirroring how
    set_users_collection() lets the async side run offline.
    """
    if name not in _sync_collections:
        _sync_collections[name] = get_sync_client()[MONGO_DB][name]
    return _sync_collections[name]


def set_sync_collection(name: str, collection: Any) -> None:
    """Injection point for tests: swap in a fake collection for one name, no server needed."""
    _sync_collections[name] = collection


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


async def create_user(username: str, password_hash: str) -> dict:
    """
    Inserts a user. Raises DuplicateKeyError (from pymongo) if the username is taken -
    the caller turns that into a 409.
    """
    from datetime import datetime, timezone

    document = {
        "username": username,
        "password_hash": password_hash,
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
    global _client, _users, _sync_client, _sync_collections
    if _client is not None:
        _client.close()
    _client = None
    _users = None
    if _sync_client is not None:
        _sync_client.close()
    _sync_client = None
    _sync_collections = {}
