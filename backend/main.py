"""
FastAPI backend for the RAG system.

Endpoints:
  POST /upload   - upload a PDF, it gets chunked + embedded + stored
  POST /chat     - ask a question, get an answer grounded in the uploaded PDF(s)
  GET  /stats    - see what's currently stored
  POST /reset    - wipe the vector store (handy while testing)
  GET  /health   - basic liveness check

Run with:  uvicorn main:app --reload --port 8000
"""
import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pdf_utils import extract_pages, chunk_document
from vectorstore import add_chunks, query_chunks, collection_stats, reset_collection
from llm import generate_answer
from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, TOP_K

app = FastAPI(title="RAG API", version="1.0")

# Allows the frontend (served from a different origin/port, e.g. file:// or :5500)
# to call this API from the browser. Tighten allow_origins before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class ChatResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save to a temp file because PyMuPDF opens from a file path/stream, not raw bytes directly
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pages = extract_pages(tmp_path)
        if not pages:
            raise HTTPException(
                status_code=422,
                detail="Couldn't extract any text from this PDF. It may be a scanned "
                       "image PDF, which needs OCR (not covered by this basic pipeline).",
            )

        chunks = chunk_document(pages, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        stored = add_chunks(file.filename, chunks)

        return {
            "filename": file.filename,
            "pages": len(pages),
            "chunks_stored": stored,
        }
    finally:
        os.remove(tmp_path)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question can't be empty.")

    chunks = query_chunks(req.question, top_k=req.top_k)
    answer = generate_answer(req.question, chunks)

    sources = [
        {
            "source": c["source"],
            "page": c["page"],
            "similarity": round(1 - c["distance"], 3),
            "snippet": (c["text"][:220] + "...") if len(c["text"]) > 220 else c["text"],
        }
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/stats")
def stats():
    return collection_stats()


@app.post("/reset")
def reset():
    reset_collection()
    return {"status": "vector store cleared"}
