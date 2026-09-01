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
from typing import Any, Optional

from src.core.config import MONGO_DB, MONGO_URI, USERS_COLLECTION
from src.core.logging import get_logger

log = get_logger(__name__)

_client = None
_users = None


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


def set_users_collection(collection: Any) -> None:
    """Injection point for tests: swap in a fake collection, no server needed."""
    global _users
    _users = collection


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
        "created_at": datetime.now(timezone.utc),
    }
    result = await _guard(users().insert_one(document))
    document["_id"] = result.inserted_id
    return document


async def count_users() -> int:
    return await _guard(users().count_documents({}))


def close() -> None:
    global _client, _users
    if _client is not None:
        _client.close()
    _client = None
    _users = None
