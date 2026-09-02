"""
The only way documents get into this system.

Two document stores, chosen by DOCUMENT_STORE (see core/config.py):

* **local** (DATA_DIR on this process's disk). Documents enter by being dropped into the
  folder by whoever runs the backend, or uploaded through POST /upload from the web UI.
  Ingestion reads that folder directly - nothing is indexed straight out of a request body.
* **cloudinary** (RAG_MODE=cloud). PDFs are uploaded straight from the browser to
  Cloudinary (see cloudinary_store.py) and registered in `cloud_documents.py`'s Mongo
  registry, keyed by the SAME "users/<id>/name.pdf" path local mode uses - so ownership,
  per-owner dedupe, and pruning all keep working unchanged; only how the PDF's bytes are
  actually read differs (fetched by URL into a temp file here, vs. opened from disk).

Ingestion is fingerprinted: a file is (re-)ingested when it's new, or when its bytes have
changed since last time. Matching on filename alone meant an edited PDF was invisible
forever.

Two job models, also chosen by DOCUMENT_STORE/IS_CLOUD:

* **local**: a background thread (`start_job` / `job_status` below) so the API never blocks
  on embedding a 900-page textbook. Correct only because the app is a single long-lived
  process - see the "single worker" note in core/ratelimit.py.
* **cloud**: no background thread - a Vercel function is killed the instant it responds, so
  there is no "meanwhile, in the background". Instead, `start_job()` computes the queue of
  work and returns immediately, and `continue_job()` does ONE file's worth of work per call.
  The frontend polls `/ingest/continue` in a loop until the job reports `done`. Progress is
  a Mongo document (`ingestion_jobs`, one per user) rather than the in-process `_state`
  dict, so it survives between the many small requests that make up one cloud ingest.
"""
import hashlib
import os
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

from src.services import answer_cache, manifest
from src.core.config import (CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, DATA_DIR, DOCUMENT_STORE,
                             INGEST_CHUNKS_PER_REQUEST)
from src.core.logging import get_logger, timed
from src.ml.embeddings import split_to_token_limit
from src.services.pdf import chunk_document, extract_pages
from src.services.vectorstore import add_chunks, delete_source

log = get_logger(__name__)

_HASH_CHUNK = 1024 * 1024  # 1MB reads while fingerprinting
IS_CLOUDINARY = DOCUMENT_STORE == "cloudinary"

# Cap on the per-job result/event lists. Held in memory (local mode) or in one Mongo
# document (cloud mode), and returned on every status poll.
MAX_JOB_RESULTS = 500
MAX_JOB_EVENTS = 60

# Uploads live in data/users/<user_id>/<filename> (local) or are registered under the same
# path (cloud), so a document's owner is recoverable from its path alone in both modes.
USERS_DIRNAME = "users"


def user_dir(user_id: str):
    """The folder a user's uploads live in (local mode only)."""
    return DATA_DIR / USERS_DIRNAME / str(user_id)


def owner_of(filename: str) -> Optional[str]:
    """
    Who a document belongs to: the user named by its path, or - for a file copied into
    data/ by hand or indexed by the CLI (local mode only; cloud mode has no such path) - the
    owner of record (the first account created).

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

def _pdf_filenames(user_id: Optional[str] = None) -> List[str]:
    """
    Every known PDF, as paths in the "users/<id>/name.pdf" shape (local: relative to
    DATA_DIR; cloud: the registry key), sorted.

    With a user_id, only that user's documents are listed - an upload triggers a scan, and
    walking/reading every other account's documents to find one new file is wasted work
    that grows with the number of users.
    """
    if IS_CLOUDINARY:
        from src.services import cloud_documents

        records = cloud_documents.list_for_user(user_id) if user_id else cloud_documents.list_all()
        return sorted(d["filename"] for d in records)

    if not DATA_DIR.is_dir():
        log.warning("Data folder %s does not exist - nothing to ingest.", DATA_DIR)
        return []

    if not user_id:
        roots = [DATA_DIR]
    else:
        # The user's own uploads...
        roots = [user_dir(user_id)]
        # ...plus, for the owner of record, the hand-copied files that live outside
        # users/. They belong to that account, so a scoped pass that skipped them would
        # "prune" every one of them as deleted.
        from src.services import ownership

        if ownership.owner_of_record() == user_id:
            roots.append(DATA_DIR)

    names = set()
    users_root = (DATA_DIR / USERS_DIRNAME).resolve()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            # When scanning DATA_DIR for the owner of record, skip other people's folders.
            if user_id and root == DATA_DIR and users_root in path.resolve().parents:
                if owner_from_path(path.relative_to(DATA_DIR).as_posix()) != user_id:
                    continue
            names.add(path.relative_to(DATA_DIR).as_posix())
    return sorted(names)


def _fingerprint(path) -> Dict:
    """sha256 + mtime + size of a LOCAL file. The hash is what actually decides re-ingest."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(block)
    stat = os.stat(path)
    return {"sha256": digest.hexdigest(), "mtime": stat.st_mtime, "size": stat.st_size}


def _fingerprint_cloud(filename: str) -> Dict:
    """
    The equivalent for a Cloudinary-registered document: the hash was already computed once,
    when uploads.validate_uploaded_bytes() checked the file at /upload/complete time, and is
    stored in the registry - there is no local file here to re-read and re-hash.
    """
    from src.services import cloud_documents

    record = cloud_documents.get(filename)
    if record is None:
        raise FileNotFoundError(f"'{filename}' is not in the Cloudinary registry.")
    return {"sha256": record["sha256"], "mtime": 0, "size": record.get("bytes", 0)}


def _fingerprint_any(filename: str) -> Dict:
    return _fingerprint_cloud(filename) if IS_CLOUDINARY else _fingerprint(DATA_DIR / filename)


def _extract_pages_any(filename: str) -> List[Dict]:
    """Extraction, from wherever the bytes live."""
    if not IS_CLOUDINARY:
        return extract_pages(str(DATA_DIR / filename))

    from src.services import cloud_documents, cloudinary_store

    record = cloud_documents.get(filename)
    if record is None:
        raise FileNotFoundError(f"'{filename}' is not in the Cloudinary registry.")
    data = cloudinary_store.fetch_bytes(record["url"])

    # PyMuPDF wants a path (or a stream, but extract_pages() is written against a path, and
    # keeping that one shared code path is worth a throwaway temp file). Vercel's /tmp is
    # writable and ephemeral, which is exactly what a file that only needs to live for the
    # length of one request wants.
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="ingest-")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return extract_pages(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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
               on_stage=None, start_chunk: int = 0,
               max_chunks: Optional[int] = None) -> Dict:
    """
    Extracts, chunks, embeds, and stores a single PDF, wherever its bytes live.

    on_stage: optional callable(stage, done, total) where stage is "extracting" or
    "embedding". The UI turns this into a per-document progress bar.

    start_chunk / max_chunks make this RESUMABLE, which is what lets a document of any size
    be ingested inside a fixed request budget. Local mode leaves both alone and does the
    whole file in one call, as it always has. Cloud mode passes a window, because a
    serverless function is killed at maxDuration: doing a whole 636-page book in one
    invocation timed out, threw away the partial work, and retried forever.

    A partial call returns status "partial" with `next_chunk`. It deliberately does NOT
    write the manifest entry or bump the answer cache - those mark a document as finished,
    and it isn't. Re-extraction on each call is intentional and cheap next to embedding:
    chunking is deterministic, so slice N sees exactly the chunk list slice N-1 saw, which
    is what makes resuming by index safe without persisting the chunks anywhere.
    """
    fingerprint = fingerprint or _fingerprint_any(filename)
    owner = owner_of(filename)

    if on_stage:
        on_stage("extracting", 0, 0)
    _event("Reading pages", filename, owner)

    with timed(log, f"extract '{filename}'"):
        pages = _extract_pages_any(filename)

    if not pages:
        _event("No extractable text - skipped (is it a scan?)", filename, owner, kind="warn")
        return {
            "filename": filename,
            "user_id": owner,
            "status": "skipped",
            "reason": "no extractable text - likely a scanned/image PDF (no OCR yet)",
        }

    chunks = chunk_document(pages, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    # The chunker counts words; the model counts tokens. Anything still over the window is
    # split here rather than being silently truncated at embedding time.
    chunks = split_to_token_limit(chunks)
    total_chunks = len(chunks)
    window = chunks[start_chunk:] if max_chunks is None else chunks[start_chunk:start_chunk + max_chunks]
    next_chunk = start_chunk + len(window)
    is_final = next_chunk >= total_chunks

    if start_chunk == 0:
        _event(f"Split {len(pages)} pages into {total_chunks} passages", filename, owner)
    log.info(
        "'%s': %d pages -> %d chunks. Embedding - a large document takes "
        "several minutes, this is not a hang.",
        filename, len(pages), len(chunks),
    )

    # Replace, don't duplicate: drop any existing vectors for this document first.
    #
    # This runs for reason == "new" as well, which looks redundant but is not. A job killed
    # part-way (Ctrl+C, `uvicorn --reload` restarting, or - in cloud mode - a function that
    # got killed between slices) leaves whatever chunks it had already stored, while the
    # manifest entry - written only once the whole file finishes - never gets written. The
    # next run therefore treats the file as "new" and re-embeds it from scratch, so without
    # this delete the abandoned chunks would pile up as duplicates on every interrupted
    # attempt. Unscoped by owner ON PURPOSE - see the note this had before the cloud split.
    #
    # ONLY on the first slice: a resumed call must not wipe the slices before it.
    if start_chunk == 0:
        delete_source(filename)

    if on_stage:
        on_stage("embedding", start_chunk, total_chunks)
    if start_chunk == 0:
        _event(f"Embedding and storing {total_chunks} passages", filename, owner)

    with timed(log, f"embed + store '{filename}' [{start_chunk}:{next_chunk}]"):
        add_chunks(
            filename, window,
            on_progress=(lambda done, total: on_stage("embedding", start_chunk + done, total_chunks))
            if on_stage else None,
            user_id=owner,
            index_offset=start_chunk,
        )

    if not is_final:
        log.info("'%s': stored %d/%d passages, more to do next call.",
                 filename, next_chunk, total_chunks)
        return {
            "filename": filename,
            "user_id": owner,
            "status": "partial",
            "reason": reason,
            "next_chunk": next_chunk,
            "chunks_total": total_chunks,
        }

    # Answers built from the previous version of this document are now wrong.
    answer_cache.bump(owner)
    _event(f"Ready - {total_chunks} passages searchable", filename, owner, kind="done")

    manifest.put(
        filename,
        sha256=fingerprint["sha256"],
        mtime=fingerprint["mtime"],
        size=fingerprint["size"],
        pages=len(pages),
        chunks=total_chunks,
        user_id=owner,
    )

    return {
        "filename": filename,
        "user_id": owner,
        "status": "ingested",
        "reason": reason,
        "pages": len(pages),
        "chunks_stored": total_chunks,
    }


def prune_deleted(present: List[str], user_id: Optional[str] = None) -> List[Dict]:
    """
    Drops documents that are in the store but no longer known (gone from disk locally, or
    removed from the Cloudinary registry in cloud mode).

    Without this, deleting a PDF leaves its chunks searchable forever: the answer still
    quotes it and still cites it by name, with no way to tell from the app that it's gone.
    """
    removed = []
    for filename in manifest.sources(user_id):
        if filename in present:
            continue
        owner = owner_of(filename)
        log.info("'%s' is no longer present - removing it from the store.", filename)
        try:
            delete_source(filename)   # unscoped: see the note in ingest_one()
            manifest.remove(filename)
            answer_cache.bump(owner)
            _event("Removed - the file is no longer present", filename, owner)
            removed.append({"filename": filename, "user_id": owner, "status": "removed"})
        except Exception as exc:
            log.exception("Failed to remove '%s'", filename)
            removed.append({"filename": filename, "user_id": owner,
                            "status": "remove_failed", "error": str(exc)})
    return removed


def _plan_ingest(filenames: List[str], force: bool) -> List[Dict]:
    """
    Shared by local's ingest_data_folder() and cloud's queue builder: for each filename, in
    order, decide skip/duplicate/already-stored/needs-ingest. Returns ONE ordered list, not
    two separate (decided, pending) lists - a pending entry carries {"pending": True,
    "filename", "reason", "fingerprint"}, everything else is a terminal result dict.

    This has to stay a single ordered list, not "everything already decided, then everything
    freshly ingested": callers append results as they go, and a caller that appends all
    decided items first and all pending items second reorders the results relative to the
    original filename-scan order whenever a pending file happens to sort before a decided one
    (e.g. "book (2).pdf" before "book.pdf" - space sorts below period in ASCII). Both the
    offline test suite and anyone reading a job's "results" list depend on scan order.
    """
    plan: List[Dict] = []
    seen_hashes: Dict[tuple, str] = {}

    for filename in filenames:
        try:
            fingerprint = _fingerprint_any(filename)
        except (OSError, FileNotFoundError) as exc:
            log.warning("Could not read '%s': %s", filename, exc)
            plan.append({"filename": filename, "user_id": owner_of(filename),
                        "status": "failed", "error": str(exc)})
            continue

        # Same bytes under two names would be indexed twice and compete with itself for
        # every retrieval slot. Scoped PER OWNER - see the note this had before the cloud
        # split: globally, two users uploading the same textbook would leave the second one
        # with a "skipped" document they can never see.
        owner = owner_of(filename)
        dedupe_key = (owner, fingerprint["sha256"])
        twin = seen_hashes.get(dedupe_key)
        if twin:
            log.warning("'%s' is byte-identical to '%s' - skipping the duplicate.", filename, twin)
            _event(f"Skipped - identical to '{twin}'", filename, owner, kind="warn")
            plan.append({"filename": filename, "user_id": owner, "status": "skipped",
                        "reason": f"duplicate of '{twin}'"})
            continue
        seen_hashes[dedupe_key] = filename

        reason = "forced" if force else _needs_ingest(filename, fingerprint)
        if reason is None:
            plan.append({"filename": filename, "user_id": owner, "status": "already_stored"})
            continue

        plan.append({
            "pending": True,
            "filename": filename,
            "reason": "changed" if reason in ("changed", "forced") else "new",
            "fingerprint": fingerprint,
        })

    return plan


def ingest_data_folder(force: bool = False, progress=None, stage=None,
                       user_id: Optional[str] = None) -> List[Dict]:
    """
    Ingests every known PDF that is new or has changed, and removes stored documents that
    are no longer present. Safe to call repeatedly - unchanged files are reported, not
    re-embedded. force=True re-ingests everything (used after a reset).

    This is the LOCAL-MODE entry point (used by the background job, the CLI, and cloud's
    own scripts/ingest.py run outside of a request). It runs the whole pass in one call,
    which is fine off a request thread but is exactly what cloud mode's per-request job
    (start_job/continue_job below) avoids doing on the request path.

    A failure on one document never stops the others: a single malformed PDF used to abort
    the whole job, leaving every file after it silently un-indexed.
    """
    filenames = _pdf_filenames(user_id)
    results: List[Dict] = list(prune_deleted(filenames, user_id=user_id))
    plan = _plan_ingest(filenames, force)

    pending_total = sum(1 for item in plan if item.get("pending"))
    pending_index = 0
    for item in plan:
        if not item.get("pending"):
            results.append(item)
            continue

        filename = item["filename"]
        if progress:
            progress(filename, pending_index, pending_total)
        pending_index += 1

        log.info("Ingesting '%s' (%s)...", filename, item["reason"])
        try:
            result = ingest_one(
                filename, reason=item["reason"], fingerprint=item["fingerprint"], on_stage=stage,
            )
        except Exception as exc:
            # Corrupt file, encrypted PDF, unreadable bytes, OOM on one monster document -
            # report it and carry on with the rest of the corpus.
            log.exception("Failed to ingest '%s'", filename)
            _event(f"Failed: {type(exc).__name__}", filename, owner_of(filename), kind="error")
            result = {"filename": filename, "user_id": owner_of(filename),
                      "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        log.info("%s", result)
        results.append(result)

    return results


# --------------------------------------------------------------------- local-mode background job

_lock = threading.Lock()
_state: Dict = {
    "state": "idle",          # idle | running | error
    "started_at": None,
    "finished_at": None,
    # Whose scan this is: a user id for an upload-triggered pass, None for a full one
    # (startup, the CLI, "sync"). This is what tells job_status() that a run belongs to the
    # caller even at moments when no file is being processed.
    "scope": None,
    "current_file": None,
    "files_done": 0,
    "files_total": 0,
    # Within the current document: "extracting" or "embedding", plus chunk counts, so the
    # UI can show real progress for one big book instead of a bar stuck at 0/1.
    "stage": None,
    "chunks_done": 0,
    "chunks_total": 0,
    "results": [],
    # A short, human-readable trail of what the pipeline actually did, newest last. The
    # progress bar answers "how far"; this answers "what is it doing", which is the question
    # people actually ask while watching a 900-page book index.
    "events": [],
    "error": None,
}


def _event(message: str, filename: Optional[str] = None, user_id: Optional[str] = None,
           kind: str = "info") -> None:
    """Appends one line to the activity trail, tagged with whose document it concerns."""
    if IS_CLOUDINARY:
        _cloud_event(message, filename, user_id, kind)
        return
    with _lock:
        _state["events"].append({
            "at": time.time(),
            "kind": kind,              # info | done | warn | error
            "message": message,
            "file": filename,
            "user_id": user_id,
        })
        if len(_state["events"]) > MAX_JOB_EVENTS:
            del _state["events"][:-MAX_JOB_EVENTS]


def _job_status_local(user_id: Optional[str] = None) -> Dict:
    with _lock:
        snapshot = dict(_state)

    if user_id is None:
        return snapshot

    results = [r for r in snapshot.get("results", []) if r.get("user_id") == user_id]
    events = [e for e in snapshot.get("events", []) if e.get("user_id") in (user_id, None)]

    # The run belongs to the caller if it was started FOR them, or if the file being
    # processed right now is theirs.
    #
    # Checking only `current_file` was a bug: it is None at the start of a run, between
    # files, and for the whole finished state - so the counts were blanked at exactly the
    # moments the UI needed them, and the progress bar sat at 0% and then jumped to nothing.
    mine = (snapshot.get("scope") == user_id
            or owner_from_path(snapshot.get("current_file") or "") == user_id)

    snapshot["results"] = results
    snapshot["events"] = events
    if not mine:
        snapshot["current_file"] = None
        snapshot["stage"] = None
        snapshot["chunks_done"] = 0
        snapshot["chunks_total"] = 0
        snapshot["files_done"] = 0
        snapshot["files_total"] = 0
        snapshot["other_user_busy"] = snapshot["state"] == "running"
    return snapshot


def _run(force: bool, user_id: Optional[str] = None) -> None:
    def progress(filename: str, index: int, total: int) -> None:
        with _lock:
            _state["current_file"] = filename
            _state["files_done"] = index
            _state["files_total"] = total
            _state["stage"] = None
            _state["chunks_done"] = 0
            _state["chunks_total"] = 0

    def stage(name: str, done: int, total: int) -> None:
        with _lock:
            _state["stage"] = name
            _state["chunks_done"] = done
            _state["chunks_total"] = total

    try:
        results = ingest_data_folder(force=force, progress=progress, stage=stage,
                                     user_id=user_id)

        while True:
            wanted, scopes = _consume_rescan_request()
            if not wanted:
                break
            log.info("Files arrived during ingestion - scanning again (%s).",
                     ", ".join(scope or "all documents" for scope in scopes))
            for scope in scopes:
                results.extend(ingest_data_folder(force=False, progress=progress,
                                                  stage=stage, user_id=scope))
            if len(results) > MAX_JOB_RESULTS:
                dropped = len(results) - MAX_JOB_RESULTS
                results = results[-MAX_JOB_RESULTS:]
                log.info("Job result list trimmed; dropped %d older entries.", dropped)

        with _lock:
            _state.update(
                state="idle", current_file=None, stage=None, chunks_done=0, chunks_total=0,
                files_done=_state["files_total"], results=results, error=None,
                finished_at=time.time(),
            )
    except Exception as exc:  # a failed ingest must not kill the server
        log.exception("Ingestion job failed")
        with _lock:
            _state.update(state="error", current_file=None, error=str(exc), finished_at=time.time())


_rescan_requested = False
_queued_scopes: List[Optional[str]] = []


def _start_job_local(force: bool = False, user_id: Optional[str] = None) -> Dict:
    global _rescan_requested
    with _lock:
        if _state["state"] == "running":
            _rescan_requested = True
            _queued_scopes.append(user_id)
            return dict(_state)
        _state.update(
            state="running", scope=user_id, started_at=time.time(), finished_at=None,
            current_file=None, files_done=0, files_total=0, stage=None, chunks_done=0,
            chunks_total=0, results=[], events=[], error=None,
        )

    threading.Thread(target=_run, args=(force, user_id), daemon=True, name="ingest").start()
    return _job_status_local()


def _consume_rescan_request() -> Tuple[bool, List[Optional[str]]]:
    """Returns (was one requested, the scopes to scan). Clears both."""
    global _rescan_requested, _queued_scopes
    with _lock:
        wanted, _rescan_requested = _rescan_requested, False
        scopes, _queued_scopes = _queued_scopes, []
    if None in scopes:
        return wanted, [None]
    return wanted, list(dict.fromkeys(scopes)) or [None]


# --------------------------------------------------------------------- cloud-mode per-request job
#
# No background thread: a Vercel function is killed the instant it responds. `start_job()`
# computes the work queue and writes it to Mongo; `continue_job()` (called by the frontend
# in a loop, via POST /ingest/continue) does exactly one file per call. Always scoped to one
# user - cloud mode has no local data/ folder to hand-copy files into, so there is no
# "owner of record" full-corpus scan to support the way local mode's startup pass needs.

def _jobs_collection():
    from src.services import database

    return database.sync_collection("ingestion_jobs")


def _cloud_event(message: str, filename: Optional[str], user_id: Optional[str], kind: str) -> None:
    if not user_id:
        return
    _jobs_collection().update_one(
        {"_id": user_id},
        {"$push": {"events": {
            "$each": [{"at": time.time(), "kind": kind, "message": message,
                       "file": filename, "user_id": user_id}],
            "$slice": -MAX_JOB_EVENTS,
        }}},
        upsert=True,
    )


def _start_job_cloud(force: bool = False, user_id: Optional[str] = None) -> Dict:
    if not user_id:
        raise ValueError("Cloud-mode ingestion jobs must be scoped to a user.")

    jobs = _jobs_collection()
    existing = jobs.find_one({"_id": user_id})
    if existing and existing.get("state") == "running":
        # Mirrors local mode's "queue a rescan" behaviour: a second start while one is
        # already in flight just asks for another pass once this one drains, rather than
        # racing two queues against each other.
        jobs.update_one({"_id": user_id}, {"$set": {"rescan_requested": True, "rescan_force": force}})
        return _job_status_cloud(user_id)

    filenames = _pdf_filenames(user_id)
    prune_results = prune_deleted(filenames, user_id=user_id)
    plan = _plan_ingest(filenames, force)
    decided = [item for item in plan if not item.get("pending")]
    pending = [item for item in plan if item.get("pending")]

    jobs.update_one(
        {"_id": user_id},
        {"$set": {
            "state": "running" if pending else "idle",
            "started_at": time.time(),
            "finished_at": None if pending else time.time(),
            "current_file": None,
            "files_done": 0,
            "files_total": len(pending),
            "stage": None,
            "chunks_done": 0,
            "chunks_total": 0,
            "pending": pending,
            "results": (prune_results + decided)[-MAX_JOB_RESULTS:],
            "error": None,
            "rescan_requested": False,
            "rescan_force": False,
        }, "$setOnInsert": {"events": []}},
        upsert=True,
    )
    return _job_status_cloud(user_id)


def continue_job(user_id: str) -> Dict:
    """
    Does ONE file's worth of ingestion for this user's queued job and returns the updated
    status, including `done`. The frontend calls this in a loop (see script.js) until
    `done` is true. A no-op, returning the current (already-done) status, if there is
    nothing queued - so it is always safe to call.
    """
    jobs = _jobs_collection()
    job = jobs.find_one({"_id": user_id})
    if job is None or job.get("state") != "running" or not job.get("pending"):
        if job and job.get("rescan_requested"):
            # The previous pass finished exactly as a new one was requested; start it now
            # rather than leaving the request stranded until someone happens to poll again.
            jobs.update_one({"_id": user_id}, {"$set": {"rescan_requested": False}})
            _start_job_cloud(force=bool(job.get("rescan_force")), user_id=user_id)
            return continue_job(user_id)
        return _job_status_cloud(user_id)

    pending = list(job["pending"])

    # A file already part-way through stays at the head of the queue with the offset it
    # reached, so this call picks up exactly where the last one stopped. Without that,
    # each call restarted the same document and a book too big for one invocation could
    # never finish - it just burned a function call and re-embedded the same chunks.
    item = pending[0]
    filename = item["filename"]
    start_chunk = int(item.get("next_chunk") or 0)

    jobs.update_one({"_id": user_id}, {"$set": {
        "current_file": filename, "stage": None,
        "chunks_done": start_chunk, "chunks_total": int(item.get("chunks_total") or 0),
    }})

    def stage(name: str, done: int, total: int) -> None:
        jobs.update_one({"_id": user_id},
                        {"$set": {"stage": name, "chunks_done": done, "chunks_total": total}})

    try:
        result = ingest_one(filename, reason=item["reason"], fingerprint=item.get("fingerprint"),
                            on_stage=stage, start_chunk=start_chunk,
                            max_chunks=INGEST_CHUNKS_PER_REQUEST)
    except Exception as exc:
        log.exception("Failed to ingest '%s'", filename)
        _cloud_event(f"Failed: {type(exc).__name__}", filename, user_id, "error")
        result = {"filename": filename, "user_id": user_id, "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}"}

    # Still mid-document: keep it at the head of the queue, remember how far we got, and
    # report progress. No result is recorded and files_done does not move - the file is
    # not done, and counting it would make the progress bar lie.
    if result.get("status") == "partial":
        item = dict(item, next_chunk=result["next_chunk"], chunks_total=result["chunks_total"])
        pending[0] = item
        jobs.update_one({"_id": user_id}, {"$set": {
            "pending": pending,
            "current_file": filename,
            "stage": "embedding",
            "chunks_done": result["next_chunk"],
            "chunks_total": result["chunks_total"],
        }})
        return _job_status_cloud(user_id)

    pending.pop(0)
    files_done = job.get("files_done", 0) + 1
    update = {
        "pending": pending,
        "files_done": files_done,
        "current_file": None,
        "stage": None,
        "chunks_done": 0,
        "chunks_total": 0,
    }
    if not pending:
        update.update(state="idle", finished_at=time.time())
    jobs.update_one({"_id": user_id}, {
        "$set": update,
        "$push": {"results": {"$each": [result], "$slice": -MAX_JOB_RESULTS}},
    })

    status = _job_status_cloud(user_id)
    if status.get("state") != "running" and job.get("rescan_requested"):
        jobs.update_one({"_id": user_id}, {"$set": {"rescan_requested": False}})
        _start_job_cloud(force=bool(job.get("rescan_force")), user_id=user_id)
        return _job_status_cloud(user_id)
    return status


def _job_status_cloud(user_id: Optional[str]) -> Dict:
    if not user_id:
        return dict(_state)  # shape compatibility for any unscoped caller
    job = _jobs_collection().find_one({"_id": user_id})
    if job is None:
        return {
            "state": "idle", "started_at": None, "finished_at": None, "scope": user_id,
            "current_file": None, "files_done": 0, "files_total": 0, "stage": None,
            "chunks_done": 0, "chunks_total": 0, "results": [], "events": [], "error": None,
            "done": True,
        }
    job = dict(job)
    job["scope"] = job.pop("_id")
    job.pop("pending", None)
    job.pop("rescan_requested", None)
    job.pop("rescan_force", None)
    job["done"] = job.get("state") != "running"
    return job


# --------------------------------------------------------------------- public API

def job_status(user_id: Optional[str] = None) -> Dict:
    """
    The job's state, redacted to what `user_id` may know (local mode - see the note there;
    cloud-mode jobs are already per-user, so nothing to redact).
    """
    if IS_CLOUDINARY:
        return _job_status_cloud(user_id)
    return _job_status_local(user_id)


def is_running(user_id: Optional[str] = None) -> bool:
    if IS_CLOUDINARY:
        if not user_id:
            return False
        job = _jobs_collection().find_one({"_id": user_id}, {"state": 1})
        return bool(job and job.get("state") == "running")
    with _lock:
        return _state["state"] == "running"


def start_job(force: bool = False, user_id: Optional[str] = None) -> Dict:
    """
    Starts (or queues) ingestion for `user_id` (or, in local mode with no user_id, a full
    scan of the whole data folder). Local mode: a background thread. Cloud mode: writes the
    work queue and returns immediately - the caller must then drive it with continue_job().
    """
    if IS_CLOUDINARY:
        return _start_job_cloud(force=force, user_id=user_id)
    return _start_job_local(force=force, user_id=user_id)
