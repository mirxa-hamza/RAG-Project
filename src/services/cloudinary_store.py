"""
Cloudinary integration: direct-from-browser PDF uploads, and reads-by-URL for ingestion.

Only reached when DOCUMENT_STORE=cloudinary (the RAG_MODE=cloud default). Talked to over
plain REST plus a computed signature, not the Cloudinary SDK - the signed-upload scheme is
about a dozen lines (see https://cloudinary.com/documentation/signatures), and pulling in a
whole SDK for it goes against this project's "no hidden abstractions" rule. httpx is
already a dependency (it makes the Pinecone/Cohere/Jina HTTP calls), so nothing new is
added to requirements.txt.

Why signed, browser-direct uploads at all: Vercel rejects any request body over 4.5MB, and
MAX_UPLOAD_MB defaults to 100. The browser has to talk to Cloudinary directly and only tell
this app the result afterwards (POST /upload/complete) - the server must never see the raw
PDF bytes arrive over the wire the way local mode's POST /upload does.

SECURITY. A signature only constrains what the BROWSER can ask Cloudinary to create (here:
"a new asset, in this account's folder"). It says nothing about what a client can later
CLAIM to this app via /upload/complete - a malicious caller could name any Cloudinary
public_id, including someone else's, and ask this app to fetch and index it under their own
account. `public_id_belongs_to()` is the check that closes that hole: it must be called
before /upload/complete does anything with a claimed public_id, because the folder is
namespaced per user, a legitimate upload can never produce a public_id outside it.
"""
import hashlib
import time
from typing import Dict

from src.core.config import (
    CLOUDINARY_API_KEY,
    CLOUDINARY_API_SECRET,
    CLOUDINARY_CLOUD_NAME,
    CLOUDINARY_FOLDER,
    PROVIDER_TIMEOUT_SECONDS,
)
from src.core.logging import get_logger

log = get_logger(__name__)


class CloudinaryError(Exception):
    """Something about talking to Cloudinary failed. Message is safe to show as-is."""


def _require_configured() -> None:
    if not (CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET):
        raise CloudinaryError(
            "Cloudinary is not configured - set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY "
            "and CLOUDINARY_API_SECRET."
        )


def _sign(params: Dict[str, str]) -> str:
    """
    Every parameter that will be sent EXCEPT file/api_key/signature/resource_type, sorted
    alphabetically by key, joined as "key=value&key=value", api_secret appended
    (unseparated), then SHA-1 hex-digested. Reproduced by hand rather than trusted to a
    dependency, since a wrong signature is silently rejected by Cloudinary with no useful
    error - get this wrong and every upload just fails.
    """
    to_sign = "&".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1((to_sign + CLOUDINARY_API_SECRET).encode("utf-8")).hexdigest()


def user_folder(user_id: str) -> str:
    """
    Where one account's PDFs live in the Cloudinary account: "<CLOUDINARY_FOLDER>/users/<id>".
    Mirrors DATA_DIR/users/<id>/ in local mode on purpose, so ownership stays a property of
    the path rather than something a request can assert.
    """
    return f"{CLOUDINARY_FOLDER}/users/{user_id}"


def public_id_belongs_to(public_id: str, user_id: str) -> bool:
    """
    True only if `public_id` sits inside this user's own folder. See the SECURITY note
    above - this is the check that stops /upload/complete from being told to fetch and
    index a public_id it did not itself just get a signature for.
    """
    folder = user_folder(user_id)
    return public_id == folder or public_id.startswith(f"{folder}/")


def sign_upload(user_id: str) -> Dict:
    """
    A signed payload the BROWSER posts straight to Cloudinary - this process never sees the
    PDF bytes. resource_type=raw so a PDF is stored as-is rather than run through
    Cloudinary's image pipeline (which would refuse or transcode it).

    Cloudinary itself rejects a signature once its `timestamp` is too old (a fixed window
    on Cloudinary's side, not one this app controls) - there is no separate expiry to
    enforce here.
    """
    _require_configured()
    timestamp = int(time.time())
    folder = user_folder(user_id)
    signature = _sign({"timestamp": str(timestamp), "folder": folder})

    return {
        "upload_url": f"https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/raw/upload",
        "api_key": CLOUDINARY_API_KEY,
        "timestamp": timestamp,
        "folder": folder,
        "signature": signature,
        "resource_type": "raw",
    }


def fetch_bytes(url: str) -> bytes:
    """Downloads a Cloudinary-hosted PDF back for ingestion to read."""
    import httpx

    try:
        response = httpx.get(url, timeout=PROVIDER_TIMEOUT_SECONDS, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise CloudinaryError(f"Could not fetch the uploaded file from Cloudinary: {exc}") from exc
    return response.content


def destroy(public_id: str) -> bool:
    """Deletes an asset from Cloudinary. Never raises - a failed cleanup call must not turn
    a successful document delete into a 500; it just leaves an orphaned Cloudinary asset,
    which is a cost issue to notice later, not a correctness one."""
    import httpx

    try:
        _require_configured()
        timestamp = int(time.time())
        signature = _sign({"timestamp": str(timestamp), "public_id": public_id})
        response = httpx.post(
            f"https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/raw/destroy",
            data={
                "public_id": public_id,
                "api_key": CLOUDINARY_API_KEY,
                "timestamp": timestamp,
                "signature": signature,
            },
            timeout=PROVIDER_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json().get("result") == "ok"
    except Exception:
        log.warning("Could not delete Cloudinary asset '%s'", public_id, exc_info=True)
        return False
