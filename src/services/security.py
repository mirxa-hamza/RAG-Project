"""
Password hashing and JWTs.

Hashing is `pwdlib` with Argon2id - deliberately not passlib, which is unmaintained and
breaks against recent bcrypt releases. `PasswordHash.recommended()` also gives migration
for free: `verify_and_update()` re-hashes an old hash when the recommended parameters
change, so accounts created today keep working when the defaults move.

Tokens are HS256 JWTs. The signing key is generated once and persisted to .env: a key
regenerated per restart would log everyone out on every reload, and a hard-coded default
would let anyone who has read this file mint a valid token for any account.
"""
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.core import config
from src.core.logging import get_logger

log = get_logger(__name__)

_hasher = None


def _password_hasher():
    global _hasher
    if _hasher is None:
        from pwdlib import PasswordHash

        _hasher = PasswordHash.recommended()
    return _hasher


def hash_password(password: str) -> str:
    return _password_hasher().hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """
    False for a wrong password AND for a malformed/unknown hash.

    A stored hash this build cannot parse must fail closed, not raise a 500 that tells an
    attacker their guess hit something unusual.
    """
    try:
        return _password_hasher().verify(password, password_hash)
    except Exception:
        log.warning("Could not verify a stored password hash; treating it as a mismatch.")
        return False


def _generate_secret_into_env() -> str:
    """
    Creates a signing key and appends it to .env so it survives restarts.

    If .env cannot be written (read-only checkout, packaged deployment) the key is still
    returned and used for this process - the app works, but every restart invalidates
    existing tokens, and the log says so.
    """
    secret = secrets.token_urlsafe(48)
    env_path = config.PROJECT_ROOT / ".env"
    try:
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        separator = "" if existing.endswith("\n") or not existing else "\n"
        with env_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{separator}\n# Generated automatically; changing it logs everyone out.\n"
                     f"JWT_SECRET={secret}\n")
        log.info("Generated a JWT signing key and saved it to %s", env_path)
    except OSError as exc:
        log.warning(
            "Generated a JWT signing key but could not write it to %s (%s). Sessions will "
            "not survive a restart until JWT_SECRET is set there by hand.", env_path, exc,
        )
    return secret


def get_secret() -> str:
    """The signing key, generating and persisting one on first use."""
    if not config.JWT_SECRET:
        config.JWT_SECRET = os.getenv("JWT_SECRET") or _generate_secret_into_env()
    return config.JWT_SECRET


def create_access_token(user_id: str, username: str) -> dict:
    """Returns {access_token, token_type, expires_in, username} - the login response body."""
    from jose import jwt

    expires_in = config.JWT_EXPIRE_HOURS * 3600
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,                     # the Mongo _id; every isolation check keys on it
        "username": username,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    token = jwt.encode(payload, get_secret(), algorithm=config.JWT_ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "username": username,
    }


def decode_access_token(token: str) -> Optional[dict]:
    """
    The token's claims, or None if it is expired, tampered with, or signed with another key.

    Never raises: every failure mode here is "this request is unauthenticated", which the
    caller turns into a 401.
    """
    from jose import JWTError, jwt

    try:
        # Algorithm is pinned: accepting whatever the token's header claims is the classic
        # JWT forgery (alg=none, or HS256 verified against an RS256 public key).
        return jwt.decode(token, get_secret(), algorithms=[config.JWT_ALGORITHM])
    except JWTError:
        return None
