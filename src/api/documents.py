"""
Corpus management endpoints.

Note what is absent: there is no upload route, and there should not be one. Documents
enter the system only by being placed in the data folder on the backend's own filesystem
(see src/services/ingestion.py). These endpoints only ever tell the backend to re-read
what is already on its disk.
"""
from fastapi import APIRouter, HTTPException, status

from src.core.logging import get_logger
from src.ml import embeddings
from src.services import ingestion, manifest, vectorstore

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
        **manifest.summary(),
    }


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
