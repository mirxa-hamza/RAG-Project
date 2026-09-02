"""
Corpus management endpoints.

Documents reach the index through the data folder on the backend's filesystem, and there
are exactly two ways to put one there: copy it in yourself, or upload it here. POST /upload
writes the file into that folder (validated and size-capped - see src/services/uploads.py)
and then kicks off the same background ingestion job as everything else, so uploaded and
hand-copied PDFs travel identical code paths. Nothing is ever indexed straight from a
request body.

Note the consequence, which is deliberate and worth keeping in mind before exposing this
server beyond localhost: anyone who can reach these routes can add documents to the corpus
and delete them from disk. There is no authentication here.
"""
import hashlib
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from src.api.auth import _enforce
from src.api.deps import get_current_user, user_id_of
from src.core import ratelimit
from src.core.config import DOCUMENT_STORE, MAX_UPLOAD_BYTES, MAX_USER_STORAGE_BYTES
from src.core.logging import get_logger
from src.ml import embeddings
from src.services import answer_cache, database, ingestion, manifest, uploads, vectorstore

log = get_logger(__name__)

router = APIRouter(tags=["documents"])

IS_CLOUDINARY = DOCUMENT_STORE == "cloudinary"


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def start_ingest(user: dict = Depends(get_current_user)):
    """
    Starts a background scan of the data folder, ingesting anything new or changed and
    pruning anything deleted. Poll GET /ingest/status for progress.

    Local mode: the scan itself is global - it is the whole data folder being reconciled,
    not one user's documents - but each file is stored under the owner its path names, and
    the status this returns is filtered to the caller. Cloud mode: there is no shared data
    folder to reconcile, so this is scoped to the caller's own documents, and - having no
    background thread - only computes the work queue; poll POST /ingest/continue to drive
    it (see services/ingestion.py's module docstring).
    """
    uid = user_id_of(user)
    ingestion.start_job(user_id=uid if IS_CLOUDINARY else None)
    return ingestion.job_status(uid)


@router.get("/ingest/status")
def ingest_status(user: dict = Depends(get_current_user)):
    """Progress, redacted to the caller's own files (see ingestion.job_status)."""
    return ingestion.job_status(user_id_of(user))


@router.get("/api/documents")
def list_documents(user: dict = Depends(get_current_user)):
    """The caller's documents. Never anyone else's, and never the ownerless ones."""
    return {"documents": manifest.summary(user_id_of(user))["documents"]}


@router.get("/stats")
def stats(user: dict = Depends(get_current_user)):
    """Reads the manifest, not every chunk's metadata. Scoped to the caller."""
    uid = user_id_of(user)
    mine = manifest.summary(uid)
    return {
        # The caller's passage count, not the store's - the total would leak how much
        # other people have uploaded.
        "total_chunks": sum(d.get("chunks", 0) for d in mine["documents"]),
        "ingesting": ingestion.is_running(),
        # The model loads in the background after the port opens; the UI holds its loading
        # screen until this is true, so the first question is never the thing that waits.
        "embedding_model_ready": embeddings.is_ready(),
        # The UI checks file sizes before uploading, so it needs the server's real limit
        # rather than a hard-coded copy that can drift out of sync with .env.
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "storage_used_bytes": uploads.used_bytes(uid),
        "storage_quota_bytes": MAX_USER_STORAGE_BYTES,
        "username": user["username"],
        # Tells the frontend which upload flow to use: the local multipart POST /upload, or
        # the sign -> direct-to-Cloudinary -> /upload/complete dance cloud mode requires.
        "document_store": DOCUMENT_STORE,
        **mine,
    }


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(request: Request, files: List[UploadFile] = File(...),
                 user: dict = Depends(get_current_user)):
    """
    Accepts one or more PDFs, writes them into the data folder, and starts indexing.

    Returns immediately with the names that were accepted - embedding a large book takes
    minutes, so the client polls GET /ingest/status for progress rather than holding a
    request open. Files are validated one by one: a bad file in a batch is reported in
    `rejected` while the good ones still go through.
    """
    if IS_CLOUDINARY:
        # Vercel rejects any request body over 4.5MB, so this endpoint - a raw multipart
        # body - cannot be the cloud-mode upload path at all, not even for small files: the
        # two modes must not silently differ by file size. Use POST /upload/sign then
        # POST /upload/complete instead (see below).
        raise HTTPException(
            status_code=400,
            detail="This deployment stores documents in Cloudinary. Use POST /upload/sign "
                   "and POST /upload/complete instead of a direct multipart upload.",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files were sent.")

    uid = user_id_of(user)
    # Per account, not per address: uploads cost minutes of CPU and disk, and the account
    # is the thing being held responsible.
    _enforce(ratelimit.UPLOAD, uid)
    accepted, rejected = [], []
    for item in files:
        try:
            # item.file is a SpooledTemporaryFile - a real file object, so the upload is
            # streamed to disk in 1MB blocks instead of being held in memory. The owner is
            # the authenticated caller, never anything the request body claims.
            accepted.append(uploads.save_pdf(item.file, item.filename or "", user_id=uid))
        except uploads.UploadError as exc:
            rejected.append({"filename": item.filename, "error": str(exc)})
        except OSError as exc:
            log.exception("Could not write '%s'", item.filename)
            rejected.append({
                "filename": item.filename,
                "error": f"Could not save the file ({exc.strerror or exc}). "
                         f"Free space: {uploads.free_space_hint()}.",
            })
        finally:
            await item.close()

    if not accepted:
        # Nothing landed on disk, so there is nothing to index and no job to poll.
        raise HTTPException(
            status_code=400,
            detail=rejected[0]["error"] if rejected else "No files were accepted.",
        )

    for ok in accepted:
        await database.record_audit(uid, user["username"], "upload", ok["filename"])
    for bad in rejected:
        await database.record_audit(uid, user["username"], "upload_rejected",
                                    f"{bad['filename']}: {bad['error']}", ok=False)

    # Scoped to the uploader: there is no reason to re-walk everyone else's folders, and
    # with many accounts that walk is the slow part of a small upload.
    ingestion.start_job(user_id=uid)
    return {"accepted": accepted, "rejected": rejected,
            "job": ingestion.job_status(uid),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}


class UploadCompleteRequest(BaseModel):
    """What the browser tells us after it has uploaded straight to Cloudinary."""
    public_id: str = Field(..., min_length=1, max_length=500)
    url: str = Field(..., min_length=1, max_length=2000)
    filename: str = Field(..., min_length=1, max_length=255)
    bytes: Optional[int] = Field(None, ge=0)


@router.post("/upload/sign")
async def upload_sign(user: dict = Depends(get_current_user)):
    """
    Cloud mode only: a short-lived signed payload the BROWSER posts straight to Cloudinary.
    This process never sees the PDF bytes - see cloudinary_store.py's module docstring for
    why that's forced on us by Vercel's 4.5MB request-body limit.
    """
    if not IS_CLOUDINARY:
        raise HTTPException(status_code=404, detail="Not found.")

    from src.services import cloudinary_store

    uid = user_id_of(user)
    _enforce(ratelimit.UPLOAD_SIGN, uid)
    try:
        return cloudinary_store.sign_upload(uid)
    except cloudinary_store.CloudinaryError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/upload/complete", status_code=status.HTTP_202_ACCEPTED)
async def upload_complete(body: UploadCompleteRequest, user: dict = Depends(get_current_user)):
    """
    Cloud mode only: the browser calls this after its direct Cloudinary upload succeeds, so
    this app can validate and register what actually landed there and start indexing it.

    SECURITY: `public_id` is client-supplied, so before anything else it is checked against
    the caller's own Cloudinary folder (see cloudinary_store.public_id_belongs_to). Without
    that check, an authenticated caller could name any public_id - including one belonging
    to another account - and have this app fetch and index it under their own library.
    """
    if not IS_CLOUDINARY:
        raise HTTPException(status_code=404, detail="Not found.")

    from src.services import cloud_documents, cloudinary_store

    uid = user_id_of(user)
    _enforce(ratelimit.UPLOAD, uid)

    if not cloudinary_store.public_id_belongs_to(body.public_id, uid):
        raise HTTPException(status_code=403, detail="That upload does not belong to you.")

    try:
        data = cloudinary_store.fetch_bytes(body.url)
    except cloudinary_store.CloudinaryError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    try:
        # Same three checks local mode's save_pdf() makes while streaming - magic bytes,
        # size cap, per-account quota - now applied to bytes fetched back from Cloudinary.
        # This is deliberately not skipped: see validate_uploaded_bytes()'s docstring.
        safe_name = uploads.validate_uploaded_bytes(body.filename, data, user_id=uid)
    except uploads.UploadError as exc:
        cloudinary_store.destroy(body.public_id)  # don't leave the rejected file billed
        raise HTTPException(status_code=400, detail=str(exc))

    final_name = cloud_documents.unique_filename(safe_name, uid)
    filename = f"{ingestion.USERS_DIRNAME}/{uid}/{final_name}"
    sha256 = hashlib.sha256(data).hexdigest()

    cloud_documents.register(
        filename, user_id=uid, public_id=body.public_id, url=body.url,
        size_bytes=len(data), sha256=sha256,
    )
    await database.record_audit(uid, user["username"], "upload", filename)

    job = ingestion.start_job(user_id=uid)
    return {"filename": filename, "bytes": len(data), "job": job}


@router.post("/ingest/continue")
async def ingest_continue(user: dict = Depends(get_current_user)):
    """
    Cloud mode only: does one file's worth of the caller's queued ingestion job and returns
    the updated status, including `done`. The frontend calls this in a loop after
    /upload/complete or /ingest until `done` is true - there is no background thread to do
    it for them (see the module docstring in services/ingestion.py for why).

    Harmless (and unnecessary) to call in local mode: it just reports the job status, since
    local mode's background thread has already done the work.
    """
    uid = user_id_of(user)
    if not IS_CLOUDINARY:
        return ingestion.job_status(uid)
    return ingestion.continue_job(uid)


@router.delete("/documents/{filename:path}")
async def delete_document(filename: str, user: dict = Depends(get_current_user)):
    """
    Removes a document completely: its vectors, its manifest entry, and the PDF itself.

    The file has to go too. Deleting only the vectors would leave the PDF in the data
    folder, and the next ingestion pass - including the one that runs at every startup -
    would faithfully index it again.
    """
    uid = user_id_of(user)
    if ingestion.is_running(uid):
        raise HTTPException(
            status_code=409,
            detail="Indexing is running. Wait for it to finish before removing a document.",
        )

    record = manifest.get(filename)

    # Someone else's document is reported as missing, not as forbidden: a 403 would
    # confirm that a document by that name exists and who might own it.
    if ingestion.owner_from_path(filename) != uid or (record and record.get("user_id") != uid):
        raise HTTPException(status_code=404, detail=f"'{filename}' is not in your library.")

    try:
        existed_on_disk = uploads.delete_document(filename, user_id=uid)
    except uploads.UploadError:
        raise HTTPException(status_code=404, detail=f"'{filename}' is not in your library.")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete the file: {exc}")

    if not existed_on_disk and record is None:
        raise HTTPException(status_code=404, detail=f"'{filename}' is not in your library.")

    # Ownership was verified above; the delete itself is unscoped so that any chunk of
    # this document is removed, including ones stored before it had an owner.
    vectorstore.delete_source(filename)
    manifest.remove(filename)
    answer_cache.bump(uid)
    await database.record_audit(uid, user["username"], "delete_document", filename)
    log.info("Removed '%s' from the library.", filename)
    return {"filename": filename, "removed": True, "file_deleted": existed_on_disk}


@router.post("/reset", status_code=status.HTTP_202_ACCEPTED)
async def reset(user: dict = Depends(get_current_user)):
    """
    Rebuilds the caller's documents from their files on disk.

    Deliberately NOT the global wipe it used to be: with several accounts, one person
    pressing "Rebuild index" must not throw away everyone else's vectors. Each of the
    caller's documents is dropped from the store and re-embedded from the PDF that is
    still in their folder.
    """
    uid = user_id_of(user)
    if ingestion.is_running(uid):
        raise HTTPException(status_code=409, detail="An ingestion job is already running.")

    mine = manifest.sources(uid)
    for name in mine:
        vectorstore.delete_source(name)
        manifest.remove(name)
    answer_cache.bump(uid)
    log.info("Rebuilding %d document(s) for user %s", len(mine), uid)
    await database.record_audit(uid, user["username"], "rebuild_index",
                                f"{len(mine)} document(s)")

    ingestion.start_job(user_id=uid)
    return ingestion.job_status(uid)
