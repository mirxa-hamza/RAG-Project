"""
One-off diagnostic for "Could not fetch the uploaded file from Cloudinary: 401 Unauthorized".

Uploads a tiny probe PDF exactly the way the app does, then tries every way of reading it
back and reports which ones your account actually allows. Deletes the probe afterwards.

Run it, paste the output, and we implement whichever strategy works:

    python scripts/diagnose_cloudinary.py

Delete this file once the question is settled - it exists to answer one question, not to
become part of the app.
"""
import base64
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from src.core import config  # noqa: E402
from src.services import cloudinary_store  # noqa: E402

PROBE = b"%PDF-1.4\n% cloudinary delivery diagnostic\n"


def _sig(to_sign: str) -> str:
    """Cloudinary's signed-delivery-URL token: s--<first 8 of urlsafe b64 of sha1>--"""
    digest = hashlib.sha1(f"{to_sign}{config.CLOUDINARY_API_SECRET}".encode()).digest()
    return base64.urlsafe_b64encode(digest).decode()[:8]


def _try(label: str, url: str, **kwargs) -> None:
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True, **kwargs)
    except httpx.HTTPError as exc:
        print(f"  [ERR ] {label}\n         {type(exc).__name__}: {exc}")
        return
    ok = response.status_code == 200
    got_pdf = response.content[:5] == b"%PDF-"
    mark = "OK  " if ok and got_pdf else "FAIL"
    print(f"  [{mark}] {label}")
    print(f"         HTTP {response.status_code}"
          f"{', got PDF bytes' if got_pdf else ''}"
          f"{'' if ok else ' - ' + response.text[:120].replace(chr(10), ' ')}")


def main() -> int:
    for name in ("CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET"):
        if not getattr(config, name):
            print(f"{name} is not set - fill it in .env first.")
            return 1

    cloud = config.CLOUDINARY_CLOUD_NAME
    print(f"Cloudinary delivery diagnostic - cloud '{cloud}'\n")

    # 1. Upload a probe exactly the way the browser does, through the app's own signer.
    signed = cloudinary_store.sign_upload("diagnostic")
    response = httpx.post(
        signed["upload_url"],
        data={"api_key": signed["api_key"], "timestamp": signed["timestamp"],
              "folder": signed["folder"], "signature": signed["signature"]},
        files={"file": ("probe.pdf", PROBE, "application/pdf")},
        timeout=60,
    )
    if response.status_code != 200:
        print(f"Upload itself failed ({response.status_code}): {response.text[:300]}")
        return 1

    asset = response.json()
    public_id = asset["public_id"]
    version = asset.get("version")
    secure_url = asset.get("secure_url", "")
    print(f"Uploaded probe: {public_id}")
    print(f"  version:      {version}")
    print(f"  type:         {asset.get('type')}")
    print(f"  access_mode:  {asset.get('access_mode', '(not reported)')}")
    print(f"  secure_url:   {secure_url}\n")

    print("Reading it back:")

    # 2. What the app does today.
    _try("plain GET of secure_url (what the app does now)", secure_url)

    # 3. Signed delivery URLs. Which string Cloudinary expects to be signed differs by
    #    whether the version is part of it, so try both shapes, with and without /v<n>/.
    base = f"https://res.cloudinary.com/{cloud}/raw/upload"
    for label, to_sign, path in (
        ("signed URL, signing public_id, no version",
         public_id, f"{public_id}"),
        ("signed URL, signing public_id, with version",
         public_id, f"v{version}/{public_id}"),
        ("signed URL, signing v<n>/public_id, with version",
         f"v{version}/{public_id}", f"v{version}/{public_id}"),
    ):
        _try(label, f"{base}/s--{_sig(to_sign)}--/{path}")

    # 4. Admin API: proves the asset exists and shows how the account classifies it, even
    #    when delivery is refused. Basic auth with key:secret, no URL signing involved.
    admin = (f"https://api.cloudinary.com/v1_1/{cloud}/resources/raw/upload/{public_id}")
    try:
        meta = httpx.get(admin, auth=(config.CLOUDINARY_API_KEY, config.CLOUDINARY_API_SECRET),
                         timeout=30)
        print(f"  [{'OK  ' if meta.status_code == 200 else 'FAIL'}] Admin API metadata "
              f"(HTTP {meta.status_code})")
        if meta.status_code == 200:
            data = meta.json()
            for key in ("type", "access_mode", "resource_type", "secure_url"):
                if key in data:
                    print(f"         {key}: {data[key]}")
        else:
            print(f"         {meta.text[:200]}")
    except httpx.HTTPError as exc:
        print(f"  [ERR ] Admin API metadata: {exc}")

    cloudinary_store.destroy(public_id)
    print(f"\nProbe deleted. If every read failed, the account is restricting delivery of "
          f"raw/PDF assets\n(Settings -> Security -> Restricted media types) - see the notes "
          f"alongside this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
