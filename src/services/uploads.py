"""
Accepting PDFs from the browser and writing them into DATA_DIR.

This module is the boundary between "bytes someone sent over HTTP" and "a file the
ingestion pipeline will read", so it is deliberately the strictest code in the project.
Everything ingestion does downstream assumes the file is a real PDF sitting at a path
inside DATA_DIR; that assumption is established here and nowhere else.

Three things it refuses to trust:

* **The filename.** A browser can send any string, including "../../.env" or "C:\\evil.pdf".
  Only the base name is kept, and it is scrubbed to a conservative character set, so a
  crafted name can never write outside DATA_DIR.
* **The extension and content type.** Both are client-supplied labels, not evidence. The
  first bytes of the saved file must actually be a PDF header.
* **The length.** Content-Length is a claim, not a fact, so the size cap is enforced while
  streaming and the partial file is deleted the moment it is exceeded. Otherwise a single
  request could fill the disk.

Writes are atomic: the upload lands in a temp file in the same folder and is only renamed
into place once it is complete and validated, so the ingestion thread can never observe a
half-written PDF and index a truncated document.
"""
import os
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import BinaryIO, Tuple

from src.core.config import DATA_DIR, MAX_UPLOAD_BYTES
from src.core.logging import get_logger

log = get_logger(__name__)

PDF_MAGIC = b"%PDF-"
_READ_CHUNK = 1024 * 1024  # 1MB


class UploadError(Exception):
    """Rejected upload. The message is written to be shown to the user as-is."""


def safe_filename(raw: str) -> str:
    """
    Turns a client-supplied filename into one that is safe to join onto DATA_DIR.

    Keeps it recognisable (the name is shown in the UI and cited in answers) while
    guaranteeing it is a single path segment with no traversal and no device names.
    """
    # A Windows client sends "C:\\Users\\me\\book.pdf"; PurePath alone won't split that on
    # POSIX, so both separators are handled explicitly before taking the last segment.
    name = str(raw or "").replace("\\", "/").split("/")[-1].strip()
    # Normalise so visually-identical unicode can't produce two "different" documents.
    name = unicodedata.normalize("NFC", name)
    name = name.lstrip(".")                      # no hidden files, no "..", no "."
    name = re.sub(r'[\x00-\x1f<>:"|?*]', "", name)   # control chars + Windows-illegal set
    name = re.sub(r"\s+", " ", name).strip()

    stem, dot, ext = name.rpartition(".")
    if not dot or ext.lower() != "pdf":
        raise UploadError("Only PDF files can be uploaded.")
    stem = stem.strip(". ") or "document"

    # Windows refuses these names regardless of extension.
    if stem.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        stem = f"_{stem}"

    # Long names blow past the filesystem limit once a " (2)" suffix is added.
    return f"{stem[:120]}.pdf"


def unique_path(filename: str) -> Tuple[Path, str]:
    """
    Resolves a free path inside DATA_DIR. Returns (path, final_filename).

    An existing name is never overwritten - two people uploading different books both
    called "notes.pdf" would otherwise silently destroy one of them. The second becomes
    "notes (2).pdf", which is also what a browser download does.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = filename[:-4]
    candidate = DATA_DIR / filename
    counter = 2
    while candidate.exists():
        candidate = DATA_DIR / f"{stem} ({counter}).pdf"
        counter += 1

    final = candidate.name
    # Belt and braces: whatever the name did, the result has to be directly inside DATA_DIR.
    if candidate.resolve().parent != DATA_DIR.resolve():
        raise UploadError("Invalid filename.")
    return candidate, final


def save_pdf(stream: BinaryIO, raw_filename: str) -> dict:
    """
    Streams one uploaded PDF into DATA_DIR.

    Returns {"filename": <name on disk>, "bytes": <size>}; raises UploadError with a
    user-facing message on anything invalid.
    """
    filename = safe_filename(raw_filename)
    destination, final_name = unique_path(filename)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Same directory as the destination so the final rename is atomic (a cross-device
    # rename is not, and would leave a partially visible file).
    fd, temp_path = tempfile.mkstemp(prefix=".upload-", suffix=".part", dir=str(DATA_DIR))
    written = 0
    header = b""

    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                block = stream.read(_READ_CHUNK)
                if not block:
                    break
                if not header:
                    header = block[:len(PDF_MAGIC)]
                    if not header.startswith(PDF_MAGIC[:len(header)]):
                        raise UploadError(
                            f"'{final_name}' is not a PDF (its contents don't start with "
                            "a PDF header)."
                        )
                written += len(block)
                if written > MAX_UPLOAD_BYTES:
                    raise UploadError(
                        f"'{final_name}' is larger than the "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit."
                    )
                out.write(block)

        if written == 0:
            raise UploadError(f"'{final_name}' is empty.")
        if not header.startswith(PDF_MAGIC):
            raise UploadError(f"'{final_name}' is not a PDF.")

        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)   # never leave a partial upload behind

    log.info("Uploaded '%s' (%.1f MB) into the data folder.", final_name, written / 1e6)
    return {"filename": final_name, "bytes": written}


def resolve_document(filename: str) -> Path:
    """
    Maps a document name from a request onto its file inside DATA_DIR.

    Accepts the nested names the rest of the app uses ("textbooks/book.pdf") but refuses
    anything that escapes the folder, including via symlinks - resolve() is what makes
    that check real rather than textual.
    """
    candidate = (DATA_DIR / filename).resolve()
    root = DATA_DIR.resolve()
    if candidate == root or root not in candidate.parents:
        raise UploadError("Invalid document name.")
    if candidate.suffix.lower() != ".pdf":
        raise UploadError("Only PDF documents can be removed this way.")
    return candidate


def delete_document(filename: str) -> bool:
    """Deletes the PDF from DATA_DIR. Returns False if it was already gone."""
    path = resolve_document(filename)
    if not path.exists():
        return False
    path.unlink()
    # Tidy up a subfolder that the deletion just emptied; ignore anything still in use.
    parent = path.parent
    if parent != DATA_DIR.resolve():
        try:
            parent.rmdir()
        except OSError:
            pass
    log.info("Deleted '%s' from the data folder.", filename)
    return True


def free_space_hint() -> str:
    """Human-readable free space on the data volume, for error messages."""
    try:
        return f"{shutil.disk_usage(DATA_DIR).free / 1e9:.1f} GB"
    except OSError:
        return "unknown"
