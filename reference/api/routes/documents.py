"""Document ingestion and management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.deps import get_document_store, get_ingestion_service
from app.core.config import get_settings
from app.core.errors import NotFoundError, ValidationError
from app.schemas.documents import DocumentListResponse, DocumentOut, DocumentUploadResponse
from app.services.document_store import DocumentStore
from app.services.ingestion.service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    settings = get_settings()
    data = await file.read()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) == 0:
        raise ValidationError("Uploaded file is empty")
    if len(data) > max_bytes:
        raise ValidationError(f"File exceeds max upload size of {settings.max_upload_mb} MB")

    doc = service.ingest(
        filename=file.filename or "upload",
        content_type=file.content_type or "application/octet-stream",
        data=data,
    )
    return DocumentUploadResponse(document_id=doc.id, status=doc.status)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    store: DocumentStore = Depends(get_document_store),
) -> DocumentListResponse:
    docs = store.list()
    return DocumentListResponse(
        items=[DocumentOut.from_model(d) for d in docs], total=len(docs)
    )


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str,
    store: DocumentStore = Depends(get_document_store),
) -> DocumentOut:
    doc = store.get(document_id)
    if doc is None:
        raise NotFoundError(f"Document {document_id} not found")
    return DocumentOut.from_model(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> None:
    service.delete_document(document_id)
