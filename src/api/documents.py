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

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.core.config import MAX_UPLOAD_BYTES
from src.core.logging import get_logger
from src.ml import embeddings
from src.services import ingestion, manifest, uploads, vectorstore

log = get_logger(__name__)

router = APIRouter(tags=["documents"])


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def start_ingest():
    """
    Starts a background scan of the data folder, ingesting anything new or changed and
    pruning anything deleted. Poll GET /ingest/status for progress.
    """
    return ingestion.start_job()


@router.get("/ingest/status")
def ingest_status():
    return ingestion.job_status()


@router.get("/stats")
def stats():
    """Reads the manifest, not every chunk's metadata."""
    return {
        "total_chunks": vectorstore.count(),
        "ingesting": ingestion.is_running(),
        # The model loads in the background after the port opens; the UI holds its loading
        # screen until this is true, so the first question is never the thing that waits.
        "embedding_model_ready": embeddings.is_ready(),
        # The UI checks file sizes before uploading, so it needs the server's real limit
        # rather than a hard-coded copy that can drift out of sync with .env.
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        **manifest.summary(),
    }


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(files: List[UploadFile] = File(...)):
    """
    Accepts one or more PDFs, writes them into the data folder, and starts indexing.

    Returns immediately with the names that were accepted - embedding a large book takes
    minutes, so the client polls GET /ingest/status for progress rather than holding a
    request open. Files are validated one by one: a bad file in a batch is reported in
    `rejected` while the good ones still go through.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were sent.")

    accepted, rejected = [], []
    for item in files:
        try:
            # item.file is a SpooledTemporaryFile - a real file object, so the upload is
            # streamed to disk in 1MB blocks instead of being held in memory.
            accepted.append(uploads.save_pdf(item.file, item.filename or ""))
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

    job = ingestion.start_job()
    return {"accepted": accepted, "rejected": rejected, "job": job,
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}


@router.delete("/documents/{filename:path}")
def delete_document(filename: str):
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

    try:
        existed_on_disk = uploads.delete_document(filename)
    except uploads.UploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete the file: {exc}")

    known = manifest.get(filename) is not None
    if not existed_on_disk and not known:
        raise HTTPException(status_code=404, detail=f"'{filename}' is not in the library.")

    vectorstore.delete_source(filename)
    manifest.remove(filename)
    log.info("Removed '%s' from the library.", filename)
    return {"filename": filename, "removed": True, "file_deleted": existed_on_disk}


@router.post("/reset", status_code=status.HTTP_202_ACCEPTED)
def reset():
    """
    Wipes the vector store, then re-ingests the data folder from scratch in the
    background - a clean rebuild, not a way to make documents disappear (they only leave
    the store if you also remove them from the data folder).
    """
    if ingestion.is_running():
        raise HTTPException(status_code=409, detail="An ingestion job is already running.")

    vectorstore.reset_collection()
    manifest.clear()
    return ingestion.start_job(force=True)
