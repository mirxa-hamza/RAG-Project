"""
Signup, login, and "who am I".

These are the only unauthenticated routes in the application. Everything else depends on
`get_current_user`.

Signup also runs the one-off adoption of pre-auth documents: PDFs that were indexed before
accounts existed have no owner, and would otherwise be invisible to everyone forever. The
first account to be created adopts them (see services/ownership.py).
"""
from fastapi import APIRouter, HTTPException, status

from src.api.deps import get_current_user, user_id_of
from src.core.logging import get_logger
from src.models.schemas import Credentials, TokenResponse, UserPublic
from src.services import database, ownership, security

from fastapi import Depends

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["auth"])

# Deliberately vague and identical for both failure modes: "no such user" and "wrong
# password" as separate messages let anyone enumerate which accounts exist.
_BAD_CREDENTIALS = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Incorrect username or password.",
    headers={"WWW-Authenticate": "Bearer"},
)


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(credentials: Credentials):
    """
    Creates an account and signs it in immediately.

    Uniqueness is checked here AND enforced by the unique index on `username`. The index
    is what survives two simultaneous requests - both would pass the read below and both
    would insert - while the read is what still refuses duplicates if Mongo rejected
    ensure_indexes() at startup.
    """
    from pymongo.errors import DuplicateKeyError

    username = credentials.username.strip()

    # Belt (this check) and braces (the unique index). The index is the only thing that
    # survives two simultaneous requests, but it only exists if Mongo accepted
    # ensure_indexes() at startup - and a signup that silently creates a second account
    # with an existing name is far worse than one that is checked twice.
    if await database.find_user_by_username(username) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="That username is already taken.")

    first_account = await database.count_users() == 0

    try:
        user = await database.create_user(username, security.hash_password(credentials.password))
    except DuplicateKeyError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="That username is already taken.")

    log.info("Created account '%s'", username)

    if first_account:
        # Runs once, ever: hand the pre-auth corpus to the first person through the door.
        adopted = ownership.adopt_unowned_documents(user_id_of(user))
        if adopted:
            log.info("Adopted %d pre-existing document(s) into '%s'", adopted, username)

    return security.create_access_token(user_id_of(user), user["username"])


@router.post("/login", response_model=TokenResponse)
async def login(credentials: Credentials):
    user = await database.find_user_by_username(credentials.username.strip())
    if user is None:
        # Hash anyway, so a missing user and a wrong password take the same time. Skipping
        # it makes "user does not exist" measurably faster and therefore detectable.
        security.hash_password(credentials.password)
        raise _BAD_CREDENTIALS

    if not security.verify_password(credentials.password, user.get("password_hash", "")):
        raise _BAD_CREDENTIALS

    log.info("'%s' signed in", user["username"])
    return security.create_access_token(user_id_of(user), user["username"])


@router.get("/me", response_model=UserPublic)
async def me(user: dict = Depends(get_current_user)):
    """Lets the frontend confirm a stored token is still valid before showing the app."""
    return {"id": user_id_of(user), "username": user["username"]}
