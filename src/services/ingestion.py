"""
The only way documents get into this system.

Documents enter in one of two ways, and both end up in DATA_DIR (see config.py): dropped
into the folder by whoever runs the backend, or uploaded through POST /upload from the web
UI. Either way, ingestion itself only ever reads that folder - nothing is indexed straight
out of a request body.

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

# Uploads live in data/users/<user_id>/<filename>, so a document's owner is recoverable
# from its path alone. That matters after a crash: the manifest may be stale, but the
# filesystem still says whose file this is.
USERS_DIRNAME = "users"


def user_dir(user_id: str):
    """The folder a user's uploads live in."""
    return DATA_DIR / USERS_DIRNAME / str(user_id)


def owner_of(filename: str) -> Optional[str]:
    """
    Who a document belongs to: the user named by its path, or - for a file copied into
    data/ by hand or indexed by the CLI - the owner of record (the first account created).

    Ownerless documents are invisible to every filter, so a PDF dropped into data/ after
    the first signup would otherwise be indexed and then seen by nobody.
    """
    from src.services import ownership

    return owner_from_path(filename) or ownership.owner_of_record()


def owner_from_path(filename: str) -> Optional[str]:
    """
    'users/652.../book.pdf' -> '652...'; anything else -> None (an ownerless document).

    Ownership is derived from the path rather than trusted from the manifest, so a
    hand-edited or half-written manifest cannot hand one user another user's document.
    """
    parts = filename.split("/")
    if len(parts) >= 3 and parts[0] == USERS_DIRNAME and parts[1]:
        return parts[1]
    return None


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

def ingest_one(filename: str, *, reason: str = "new", fingerprint: Optional[Dict] = None,
               on_stage=None) -> Dict:
    """
    Extracts, chunks, embeds, and stores a single PDF already sitting in DATA_DIR.

    on_stage: optional callable(stage, done, total) where stage is "extracting" or
    "embedding". The UI turns this into a per-document progress bar.
    """
    path = DATA_DIR / filename
    fingerprint = fingerprint or _fingerprint(path)
    owner = owner_of(filename)

    if on_stage:
        on_stage("extracting", 0, 0)

    with timed(log, f"extract '{filename}'"):
        pages = extract_pages(str(path))

    if not pages:
        return {
            "filename": filename,
            "user_id": owner,
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
    # Unscoped by owner ON PURPOSE. A document's name is its path relative to data/, and
    # uploads live under users/<id>/, so one document's name can never match another
    # user's file. Scoping the delete by owner would leave behind chunks written before an
    # owner existed - an interrupted pre-auth ingest, for instance - which no later delete
    # would ever match.
    delete_source(filename)

    if on_stage:
        on_stage("embedding", 0, len(chunks))

    with timed(log, f"embed + store '{filename}'"):
        stored = add_chunks(
            filename, chunks,
            on_progress=(lambda done, total: on_stage("embedding", done, total)) if on_stage else None,
            user_id=owner,
        )

    manifest.put(
        filename,
        sha256=fingerprint["sha256"],
        mtime=fingerprint["mtime"],
        size=fingerprint["size"],
        pages=len(pages),
        chunks=stored,
        user_id=owner,
    )

    return {
        "filename": filename,
        "user_id": owner,
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
        owner = owner_of(filename)
        log.info("'%s' is gone from the data folder - removing it from the store.", filename)
        try:
            delete_source(filename)   # unscoped: see the note in ingest_one()
            manifest.remove(filename)
            removed.append({"filename": filename, "user_id": owner, "status": "removed"})
        except Exception as exc:
            log.exception("Failed to remove '%s'", filename)
            removed.append({"filename": filename, "user_id": owner,
                            "status": "remove_failed", "error": str(exc)})
    return removed


def ingest_data_folder(force: bool = False, progress=None, stage=None) -> List[Dict]:
    """
    Ingests every PDF in DATA_DIR that is new or has changed on disk, and removes stored
    documents whose file has been deleted. Safe to call repeatedly - unchanged files are
    reported, not re-embedded.

    force=True re-ingests everything (used after a reset).
    progress: optional callable(filename, index, total) for job reporting.
    stage: optional callable(stage_name, done, total) for within-document progress.

    A failure on one document never stops the others: a single malformed PDF used to abort
    the whole job, leaving every file after it silently un-indexed.
    """
    filenames = _pdf_filenames()
    results: List[Dict] = list(prune_deleted(filenames))
    seen_hashes: Dict[tuple, str] = {}

    for index, filename in enumerate(filenames):
        if progress:
            progress(filename, index, len(filenames))

        try:
            fingerprint = _fingerprint(DATA_DIR / filename)
        except OSError as exc:
            log.warning("Could not read '%s': %s", filename, exc)
            results.append({"filename": filename, "user_id": owner_of(filename),
                            "status": "failed", "error": str(exc)})
            continue

        # Same bytes under two names would be indexed twice and then compete with itself
        # for every retrieval slot, crowding out other documents.
        #
        # Scoped PER OWNER: globally, two users uploading the same textbook would leave the
        # second one with a "skipped" document they can never see or search, because the
        # only stored copy belongs to somebody else.
        owner = owner_of(filename)
        dedupe_key = (owner, fingerprint["sha256"])
        twin = seen_hashes.get(dedupe_key)
        if twin:
            log.warning("'%s' is byte-identical to '%s' - skipping the duplicate.", filename, twin)
            results.append({"filename": filename, "user_id": owner, "status": "skipped",
                            "reason": f"duplicate of '{twin}'"})
            continue
        seen_hashes[dedupe_key] = filename

        reason = "forced" if force else _needs_ingest(filename, fingerprint)
        if reason is None:
            results.append({"filename": filename, "user_id": owner_of(filename),
                            "status": "already_stored"})
            continue

        log.info("Ingesting '%s' (%s)...", filename, reason)
        try:
            result = ingest_one(
                filename,
                reason="changed" if reason in ("changed", "forced") else "new",
                fingerprint=fingerprint,
                on_stage=stage,
            )
        except Exception as exc:
            # Corrupt file, encrypted PDF, unreadable bytes, OOM on one monster document -
            # report it and carry on with the rest of the corpus.
            log.exception("Failed to ingest '%s'", filename)
            result = {"filename": filename, "user_id": owner_of(filename),
                      "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

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
    # Within the current document: "extracting" or "embedding", plus chunk counts, so the
    # UI can show real progress for one big book instead of a bar stuck at 0/1.
    "stage": None,
    "chunks_done": 0,
    "chunks_total": 0,
    "results": [],
    "error": None,
}


def job_status(user_id: Optional[str] = None) -> Dict:
    """
    The job's state. With a user_id, it is redacted to what that user may know.

    The job is global - one thread indexes everybody's uploads - but its progress is not
    public: `current_file` and `results` would otherwise tell you the filenames of every
    other user's documents. Someone else's file in progress is reported as busy with no
    name, so the UI can still say "indexing" without naming what.
    """
    with _lock:
        snapshot = dict(_state)

    if user_id is None:
        return snapshot

    results = [r for r in snapshot.get("results", []) if r.get("user_id") == user_id]
    mine = owner_from_path(snapshot.get("current_file") or "") == user_id

    snapshot["results"] = results
    if not mine:
        snapshot["current_file"] = None
        snapshot["stage"] = None
        snapshot["chunks_done"] = 0
        snapshot["chunks_total"] = 0
        # Progress counts describe the whole queue, most of which may not be theirs.
        snapshot["files_done"] = 0
        snapshot["files_total"] = 0
        snapshot["other_user_busy"] = snapshot["state"] == "running"
    return snapshot


def is_running() -> bool:
    with _lock:
        return _state["state"] == "running"


def _run(force: bool) -> None:
    def progress(filename: str, index: int, total: int) -> None:
        with _lock:
            _state["current_file"] = filename
            _state["files_done"] = index
            _state["files_total"] = total
            # A new file starts from scratch; leaving the previous file's counts in place
            # made the bar jump backwards.
            _state["stage"] = None
            _state["chunks_done"] = 0
            _state["chunks_total"] = 0

    def stage(name: str, done: int, total: int) -> None:
        with _lock:
            _state["stage"] = name
            _state["chunks_done"] = done
            _state["chunks_total"] = total

    try:
        results = ingest_data_folder(force=force, progress=progress, stage=stage)

        # Anything uploaded while this run was in flight is picked up now, in the same
        # thread, so the job the client is polling covers it too.
        while _consume_rescan_request():
            log.info("Files arrived during ingestion - scanning the data folder again.")
            results.extend(ingest_data_folder(force=False, progress=progress, stage=stage))

        with _lock:
            _state.update(
                state="idle",
                current_file=None,
                stage=None,
                chunks_done=0,
                chunks_total=0,
                files_done=_state["files_total"],
                results=results,
                error=None,
                finished_at=time.time(),
            )
    except Exception as exc:  # a failed ingest must not kill the server
        log.exception("Ingestion job failed")
        with _lock:
            _state.update(state="error", current_file=None, error=str(exc), finished_at=time.time())


_rescan_requested = False


def start_job(force: bool = False) -> Dict:
    """
    Kicks off ingestion in a background thread and returns immediately, so neither server
    startup nor an HTTP request ever blocks on embedding a large corpus.

    Called while a job is already running, it flags a re-scan instead of starting a second
    thread. That matters for uploads: the running job listed the folder when it started, so
    a PDF that lands mid-run would otherwise sit unindexed until someone pressed sync.
    """
    global _rescan_requested
    with _lock:
        if _state["state"] == "running":
            _rescan_requested = True
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


def _consume_rescan_request() -> bool:
    global _rescan_requested
    with _lock:
        wanted, _rescan_requested = _rescan_requested, False
    return wanted
