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
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.deps import get_current_user, user_id_of
from src.core.config import MAX_UPLOAD_BYTES
from src.core.logging import get_logger
from src.ml import embeddings
from src.services import ingestion, manifest, uploads, vectorstore

log = get_logger(__name__)

router = APIRouter(tags=["documents"])


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def start_ingest(user: dict = Depends(get_current_user)):
    """
    Starts a background scan of the data folder, ingesting anything new or changed and
    pruning anything deleted. Poll GET /ingest/status for progress.

    The scan itself is global - it is the filesystem being reconciled, not one user's
    documents - but each file is stored under the owner its path names, and the status this
    returns is filtered to the caller.
    """
    ingestion.start_job()
    return ingestion.job_status(user_id_of(user))


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
        "username": user["username"],
        **mine,
    }


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(files: List[UploadFile] = File(...), user: dict = Depends(get_current_user)):
    """
    Accepts one or more PDFs, writes them into the data folder, and starts indexing.

    Returns immediately with the names that were accepted - embedding a large book takes
    minutes, so the client polls GET /ingest/status for progress rather than holding a
    request open. Files are validated one by one: a bad file in a batch is reported in
    `rejected` while the good ones still go through.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were sent.")

    uid = user_id_of(user)
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

    ingestion.start_job()
    return {"accepted": accepted, "rejected": rejected,
            "job": ingestion.job_status(uid),
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}


@router.delete("/documents/{filename:path}")
def delete_document(filename: str, user: dict = Depends(get_current_user)):
    """
    Removes a document completely: its vectors, its manifest entry, and the PDF itself.

    The file has to go too. Deleting only the vectors would leave the PDF in the data
    folder, and the next ingestion pass - including the one that runs at every startup -
    would faithfully index it again.
    """
    if ingestion.is_running():
        raise HTTPException(
            status_code=409,
            detail="Indexing is running. Wait for it to finish before removing a document.",
        )

    uid = user_id_of(user)
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
    log.info("Removed '%s' from the library.", filename)
    return {"filename": filename, "removed": True, "file_deleted": existed_on_disk}


@router.post("/reset", status_code=status.HTTP_202_ACCEPTED)
def reset(user: dict = Depends(get_current_user)):
    """
    Rebuilds the caller's documents from their files on disk.

    Deliberately NOT the global wipe it used to be: with several accounts, one person
    pressing "Rebuild index" must not throw away everyone else's vectors. Each of the
    caller's documents is dropped from the store and re-embedded from the PDF that is
    still in their folder.
    """
    if ingestion.is_running():
        raise HTTPException(status_code=409, detail="An ingestion job is already running.")

    uid = user_id_of(user)
    mine = manifest.sources(uid)
    for name in mine:
        vectorstore.delete_source(name)
        manifest.remove(name)
    log.info("Rebuilding %d document(s) for user %s", len(mine), uid)

    ingestion.start_job()
    return ingestion.job_status(uid)
