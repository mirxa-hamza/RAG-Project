"""
FastAPI backend for the RAG system.

Documents come ONLY from the backend's data folder (see config.DATA_DIR) - there is no
upload endpoint, so a user on the frontend can ask questions but can never add or change
what's in the vector store.

Ingestion runs as a background job, so the server answers requests immediately even while
a 900-page book is still being embedded.

Endpoints:
  POST /ingest         - start a background scan of the data folder (202)
  GET  /ingest/status  - progress of the current/last ingestion job
  POST /chat           - ask a question, get an answer grounded in the ingested PDF(s)
  GET  /stats          - what's currently stored
  POST /reset          - wipe the vector store and re-ingest from scratch (202)
  GET  /health         - liveness check

Run with:  uvicorn app.main:app --reload --port 8000     (from the backend/ folder)
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app import ingest, manifest, vectorstore
from app.config import MAX_TOP_K, TOP_K
from app.llm import generate_answer
from app.logging_setup import get_logger, timed
from app.pdf_utils import format_pages

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick ingestion off in the background - the API is up immediately, and new or
    # changed PDFs get picked up while it serves requests.
    ingest.start_job()
    yield


app = FastAPI(title="RAG API", version="2.0", lifespan=lifespan)

# Allows the frontend (served from a different origin/port, e.g. file:// or :5500)
# to call this API from the browser. Tighten allow_origins before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Bounded: an unbounded top_k lets a client build an enormous prompt.
    top_k: int = Field(TOP_K, ge=1, le=MAX_TOP_K)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Question can't be empty.")
        return stripped


class Source(BaseModel):
    source: Optional[str]
    pages: str
    similarity: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def start_ingest():
    """
    Starts a background scan of the data folder, ingesting anything new or changed.
    No file upload - this only ever reads what's already on disk on the backend.
    Poll GET /ingest/status for progress.
    """
    return ingest.start_job()


@app.get("/ingest/status")
def ingest_status():
    return ingest.job_status()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question  # already stripped and validated non-empty

    with timed(log, "chat request"):
        chunks = vectorstore.query_chunks(question, top_k=req.top_k)
        answer = generate_answer(question, chunks)

    sources = [
        Source(
            source=c["source"],
            pages=format_pages(c["page_start"], c["page_end"]),
            similarity=c["similarity"],
            snippet=(c["text"][:220] + "...") if len(c["text"]) > 220 else c["text"],
        )
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/stats")
def stats():
    """Reads the manifest, not every chunk's metadata."""
    return {
        "total_chunks": vectorstore.count(),
        "ingesting": ingest.is_running(),
        **manifest.summary(),
    }


@app.post("/reset", status_code=status.HTTP_202_ACCEPTED)
def reset():
    """
    Wipes the vector store, then re-ingests the data folder from scratch in the
    background - a clean rebuild, not a way to make documents disappear (they only leave
    the store if you also remove them from the data folder).
    """
    if ingest.is_running():
        raise HTTPException(status_code=409, detail="An ingestion job is already running.")

    vectorstore.reset_collection()
    manifest.clear()
    return ingest.start_job(force=True)
