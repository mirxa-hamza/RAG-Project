"""
Shared FastAPI dependencies.

`get_current_user` is the single gate: every route that touches documents depends on it,
and every isolation check downstream keys on the id it returns. There is deliberately no
"optional user" variant - a route that can run without a user is a route that can leak.
"""
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.logging import get_logger
from src.services import database, security

log = get_logger(__name__)

# auto_error=False so a missing header produces our own 401 with a useful message rather
# than FastAPI's bare 403.
_bearer = HTTPBearer(auto_error=False)

_UNAUTHORISED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Sign in to continue.",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """
    The signed-in user's Mongo document, or 401.

    Every failure - no header, wrong scheme, bad signature, expired, unknown user id -
    returns the same 401 with the same message. Distinguishing them would tell an attacker
    which half of their guess was right.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _UNAUTHORISED

    claims = security.decode_access_token(credentials.credentials)
    if not claims or not claims.get("sub"):
        raise _UNAUTHORISED

    user = await database.find_user_by_id(claims["sub"])
    if user is None:
        # A valid signature for an account that no longer exists: the token outlived the
        # user. Re-reading the user on every request is what makes deletion effective.
        raise _UNAUTHORISED

    # Revocation. A token minted before a password change or a "sign out everywhere"
    # carries an older version than the account now has, and stops working immediately.
    if int(claims.get("ver", 1)) != int(user.get("token_version", 1)):
        log.info("Rejected a token for '%s' issued before its version was bumped.",
                 user.get("username"))
        raise _UNAUTHORISED

    return user


def user_id_of(user: dict) -> str:
    """The canonical string form of a user's id - what gets written into chunk metadata."""
    return str(user["_id"])


async def current_user_id(user: dict = Depends(get_current_user)) -> str:
    return user_id_of(user)
