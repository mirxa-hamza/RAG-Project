"""
Signup, login, and "who am I".

These are the only unauthenticated routes in the application. Everything else depends on
`get_current_user`.

Signup also runs the one-off adoption of pre-auth documents: PDFs that were indexed before
accounts existed have no owner, and would otherwise be invisible to everyone forever. The
first account to be created adopts them (see services/ownership.py).
"""
from fastapi import APIRouter, HTTPException, Request, status

from src.api.deps import get_current_user, user_id_of
from src.core import ratelimit
from src.core.logging import get_logger
from src.models.schemas import (
    ConfirmPassword,
    Credentials,
    PasswordChange,
    SignupCredentials,
    TokenResponse,
    UserPublic,
)
from src.services import database, ownership, security, sessions

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


def _enforce(limit, key: str) -> None:
    """429 with a Retry-After header when a limit is exceeded."""
    retry_after = ratelimit.check(limit, key)
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(credentials: SignupCredentials, request: Request):
    """
    Creates an account and signs it in immediately.

    Uniqueness is checked here AND enforced by the unique index on `username`. The index
    is what survives two simultaneous requests - both would pass the read below and both
    would insert - while the read is what still refuses duplicates if Mongo rejected
    ensure_indexes() at startup.
    """
    from pymongo.errors import DuplicateKeyError

    _enforce(ratelimit.SIGNUP, ratelimit.client_key(request))
    username = credentials.username.strip()
    name = credentials.name.strip()

    # Belt (this check) and braces (the unique index). The index is the only thing that
    # survives two simultaneous requests, but it only exists if Mongo accepted
    # ensure_indexes() at startup - and a signup that silently creates a second account
    # with an existing name is far worse than one that is checked twice.
    if await database.find_user_by_username(username) is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="That username is already taken.")

    first_account = await database.count_users() == 0

    try:
        user = await database.create_user(username, security.hash_password(credentials.password),
                                          name=name)
    except DuplicateKeyError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="That username is already taken.")

    log.info("Created account '%s'", username)
    await database.record_audit(user_id_of(user), username, "signup")

    if first_account:
        # Runs once, ever: hand the pre-auth corpus to the first person through the door.
        adopted = ownership.adopt_unowned_documents(user_id_of(user))
        if adopted:
            log.info("Adopted %d pre-existing document(s) into '%s'", adopted, username)

    return security.create_access_token(user_id_of(user), user["username"],
                                       user.get("token_version", 1), name=user.get("name"))


@router.post("/login", response_model=TokenResponse)
async def login(credentials: Credentials, request: Request):
    # Limited by BOTH the caller's address and the username being tried: by address alone,
    # a botnet spreads a guessing run across many IPs; by username alone, one attacker
    # locks out a real user by guessing at their name. Together, neither works.
    username = credentials.username.strip()
    _enforce(ratelimit.LOGIN, ratelimit.client_key(request))
    _enforce(ratelimit.LOGIN, f"user:{username.lower()}")

    user = await database.find_user_by_username(username)
    if user is None:
        # Hash anyway, so a missing user and a wrong password take the same time. Skipping
        # it makes "user does not exist" measurably faster and therefore detectable.
        security.hash_password(credentials.password)
        await database.record_audit(None, username, "login", "unknown user", ok=False)
        raise _BAD_CREDENTIALS

    if not security.verify_password(credentials.password, user.get("password_hash", "")):
        await database.record_audit(user_id_of(user), username, "login",
                                    "wrong password", ok=False)
        raise _BAD_CREDENTIALS

    log.info("'%s' signed in", user["username"])
    await database.record_audit(user_id_of(user), username, "login")
    return security.create_access_token(user_id_of(user), user["username"],
                                        user.get("token_version", 1), name=user.get("name"))


@router.get("/me", response_model=UserPublic)
async def me(user: dict = Depends(get_current_user)):
    """Lets the frontend confirm a stored token is still valid before showing the app."""
    return {"id": user_id_of(user), "username": user["username"], "name": user.get("name")}


@router.post("/me/password", response_model=TokenResponse)
async def change_password(body: PasswordChange, request: Request,
                          user: dict = Depends(get_current_user)):
    """
    Changes the password and signs out every other session.

    The current password is required even though the caller is already authenticated: a
    stolen token would otherwise be enough to take the account permanently.
    """
    _enforce(ratelimit.LOGIN, f"user:{user['username'].lower()}")

    if not security.verify_password(body.current_password, user.get("password_hash", "")):
        await database.record_audit(user_id_of(user), user["username"], "password_change",
                                    "wrong current password", ok=False)
        # 403, NOT 401. The caller's token is perfectly valid - it is the confirmation
        # that failed. A 401 here made the frontend's "any 401 ends the session" rule fire
        # and signed the user out for mistyping their old password.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="That is not your current password.")

    uid = user_id_of(user)
    await database.set_password(uid, security.hash_password(body.new_password))
    await database.record_audit(uid, user["username"], "password_change")

    # The version bump above invalidated the caller's own token too, so hand back a new one
    # rather than signing them out of the session they are actively using.
    refreshed = await database.find_user_by_id(uid)
    return security.create_access_token(uid, user["username"],
                                        refreshed.get("token_version", 1), name=user.get("name"))


@router.post("/me/signout-everywhere", status_code=status.HTTP_204_NO_CONTENT)
async def signout_everywhere(user: dict = Depends(get_current_user)):
    """Invalidates every token for this account, including the caller's own."""
    uid = user_id_of(user)
    await database.bump_token_version(uid)
    await database.record_audit(uid, user["username"], "signout_everywhere")
    return None


@router.delete("/me")
async def delete_account(body: ConfirmPassword, user: dict = Depends(get_current_user)):
    """
    Deletes the account and everything it owns: PDFs, chunks, manifest entries.

    Order matters. Documents go first, then the account: a failure half way leaves an
    account that can sign in and retry, whereas deleting the account first would strand
    every document under an id that no longer resolves - invisible and undeletable, which
    is exactly the state this endpoint exists to prevent.
    """
    if not security.verify_password(body.password, user.get("password_hash", "")):
        # 403 for the same reason as the password change above: the session is fine, the
        # confirmation is not.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="That is not your password.")

    uid = user_id_of(user)
    removed = ownership.delete_all_documents(uid)
    # The conversations go with the account. Leaving them behind would keep a person's
    # questions - and the passages quoted back to them - in the database after they asked
    # for everything to be deleted.
    conversations = await sessions.delete_all(uid)
    await database.delete_user(uid)
    await database.record_audit(uid, user["username"], "account_deleted",
                                f"{removed} document(s), {conversations} conversation(s)")
    log.info("Deleted account '%s', %d document(s) and %d conversation(s)",
             user["username"], removed, conversations)
    return {"deleted": True, "documents_removed": removed}
