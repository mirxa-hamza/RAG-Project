"""
The only way documents get into this system.

There is deliberately no upload endpoint - a user talking to the frontend can only ask
questions, never add documents. Whoever runs the backend drops PDFs into DATA_DIR
(see config.py) and either runs `python scripts/ingest.py` or calls POST /ingest.

Ingestion is fingerprinted: a file is (re-)ingested when it's new, or when its bytes have
changed since last time. Matching on filename alone meant an edited PDF was invisible
forever.

Long ingests run as a background job (see `start_job` / `job_status`) so the API never
blocks on embedding a 900-page textbook.
"""
import hashlib
import os
import threading
import time
from typing import Dict, List, Optional

from src.services import manifest
from src.core.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, DATA_DIR
from src.core.logging import get_logger, timed
from src.services.pdf import chunk_document, extract_pages
from src.services.vectorstore import add_chunks, delete_source

log = get_logger(__name__)

_HASH_CHUNK = 1024 * 1024  # 1MB reads while fingerprinting


# --------------------------------------------------------------------- helpers

def _pdf_filenames() -> List[str]:
    """
    Every PDF under DATA_DIR, including nested folders, as paths relative to DATA_DIR
    with forward slashes (so a document's identity is stable across Windows and POSIX).

    Recursion matters: dropping a PDF into `data/textbooks/` used to make it silently
    invisible, with no warning and no entry in /stats.
    """
    if not DATA_DIR.is_dir():
        log.warning("Data folder %s does not exist - nothing to ingest.", DATA_DIR)
        return []

    names = [
        path.relative_to(DATA_DIR).as_posix()
        for path in DATA_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    return sorted(names)


def _fingerprint(path) -> Dict:
    """sha256 + mtime + size. The hash is what actually decides whether to re-ingest."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(block)
    stat = os.stat(path)
    return {"sha256": digest.hexdigest(), "mtime": stat.st_mtime, "size": stat.st_size}


def _needs_ingest(filename: str, fingerprint: Dict) -> Optional[str]:
    """Returns None if already up to date, else the reason ('new' or 'changed')."""
    record = manifest.get(filename)
    if record is None:
        return "new"
    if record.get("sha256") != fingerprint["sha256"]:
        return "changed"
    return None


# --------------------------------------------------------------------- ingestion

def ingest_one(filename: str, *, reason: str = "new", fingerprint: Optional[Dict] = None) -> Dict:
    """Extracts, chunks, embeds, and stores a single PDF already sitting in DATA_DIR."""
    path = DATA_DIR / filename
    fingerprint = fingerprint or _fingerprint(path)

    with timed(log, f"extract '{filename}'"):
        pages = extract_pages(str(path))

    if not pages:
        return {
            "filename": filename,
            "status": "skipped",
            "reason": "no extractable text - likely a scanned/image PDF (no OCR yet)",
        }

    chunks = chunk_document(pages, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    log.info(
        "'%s': %d pages -> %d chunks. Embedding on CPU - a large document takes "
        "several minutes, this is not a hang.",
        filename, len(pages), len(chunks),
    )

    # Replace, don't duplicate: drop any existing vectors for this document first.
    #
    # This runs for reason == "new" as well, which looks redundant but is not. A job killed
    # part-way (Ctrl+C, or `uvicorn --reload` restarting on a file save) leaves whatever
    # chunks it had already stored in Chroma, while the manifest entry - written only once
    # the whole file finishes - never gets written. The next run therefore treats the file
    # as "new" and re-embeds it from scratch under a fresh run_id, so without this delete
    # the abandoned chunks would pile up as duplicates on every interrupted attempt.
    delete_source(filename)

    with timed(log, f"embed + store '{filename}'"):
        stored = add_chunks(filename, chunks)

    manifest.put(
        filename,
        sha256=fingerprint["sha256"],
        mtime=fingerprint["mtime"],
        size=fingerprint["size"],
        pages=len(pages),
        chunks=stored,
    )

    return {
        "filename": filename,
        "status": "ingested",
        "reason": reason,
        "pages": len(pages),
        "chunks_stored": stored,
    }


def prune_deleted(present: List[str]) -> List[Dict]:
    """
    Drops documents that are in the store but no longer on disk.

    Without this, deleting a PDF from data/ leaves its chunks searchable forever: the
    answer still quotes it and still cites it by name, and there is no way to tell from
    the app that the file is gone.
    """
    removed = []
    for filename in manifest.sources():
        if filename in present:
            continue
        log.info("'%s' is gone from the data folder - removing it from the store.", filename)
        try:
            delete_source(filename)
            manifest.remove(filename)
            removed.append({"filename": filename, "status": "removed"})
        except Exception as exc:
            log.exception("Failed to remove '%s'", filename)
            removed.append({"filename": filename, "status": "remove_failed", "error": str(exc)})
    return removed


def ingest_data_folder(force: bool = False, progress=None) -> List[Dict]:
    """
    Ingests every PDF in DATA_DIR that is new or has changed on disk, and removes stored
    documents whose file has been deleted. Safe to call repeatedly - unchanged files are
    reported, not re-embedded.

    force=True re-ingests everything (used after a reset).
    progress: optional callable(filename, index, total) for job reporting.

    A failure on one document never stops the others: a single malformed PDF used to abort
    the whole job, leaving every file after it silently un-indexed.
    """
    filenames = _pdf_filenames()
    results: List[Dict] = list(prune_deleted(filenames))
    seen_hashes: Dict[str, str] = {}

    for index, filename in enumerate(filenames):
        if progress:
            progress(filename, index, len(filenames))

        try:
            fingerprint = _fingerprint(DATA_DIR / filename)
        except OSError as exc:
            log.warning("Could not read '%s': %s", filename, exc)
            results.append({"filename": filename, "status": "failed", "error": str(exc)})
            continue

        # Same bytes under two names would be indexed twice and then compete with itself
        # for every retrieval slot, crowding out other documents.
        twin = seen_hashes.get(fingerprint["sha256"])
        if twin:
            log.warning("'%s' is byte-identical to '%s' - skipping the duplicate.", filename, twin)
            results.append({"filename": filename, "status": "skipped",
                            "reason": f"duplicate of '{twin}'"})
            continue
        seen_hashes[fingerprint["sha256"]] = filename

        reason = "forced" if force else _needs_ingest(filename, fingerprint)
        if reason is None:
            results.append({"filename": filename, "status": "already_stored"})
            continue

        log.info("Ingesting '%s' (%s)...", filename, reason)
        try:
            result = ingest_one(
                filename,
                reason="changed" if reason in ("changed", "forced") else "new",
                fingerprint=fingerprint,
            )
        except Exception as exc:
            # Corrupt file, encrypted PDF, unreadable bytes, OOM on one monster document -
            # report it and carry on with the rest of the corpus.
            log.exception("Failed to ingest '%s'", filename)
            result = {"filename": filename, "status": "failed", "error":
                      f"{type(exc).__name__}: {exc}"}

        log.info("%s", result)
        results.append(result)

    return results


# --------------------------------------------------------------------- background job

_lock = threading.Lock()
_state: Dict = {
    "state": "idle",          # idle | running | error
    "started_at": None,
    "finished_at": None,
    "current_file": None,
    "files_done": 0,
    "files_total": 0,
    "results": [],
    "error": None,
}


def job_status() -> Dict:
    with _lock:
        return dict(_state)


def is_running() -> bool:
    with _lock:
        return _state["state"] == "running"


def _run(force: bool) -> None:
    def progress(filename: str, index: int, total: int) -> None:
        with _lock:
            _state["current_file"] = filename
            _state["files_done"] = index
            _state["files_total"] = total

    try:
        results = ingest_data_folder(force=force, progress=progress)
        with _lock:
            _state.update(
                state="idle",
                current_file=None,
                files_done=_state["files_total"],
                results=results,
                error=None,
                finished_at=time.time(),
            )
    except Exception as exc:  # a failed ingest must not kill the server
        log.exception("Ingestion job failed")
        with _lock:
            _state.update(state="error", current_file=None, error=str(exc), finished_at=time.time())


def start_job(force: bool = False) -> Dict:
    """
    Kicks off ingestion in a background thread and returns immediately, so neither server
    startup nor an HTTP request ever blocks on embedding a large corpus.
    """
    with _lock:
        if _state["state"] == "running":
            return dict(_state)
        _state.update(
            state="running",
            started_at=time.time(),
            finished_at=None,
            current_file=None,
            files_done=0,
            files_total=0,
            results=[],
            error=None,
        )

    threading.Thread(target=_run, args=(force,), daemon=True, name="ingest").start()
    return job_status()
