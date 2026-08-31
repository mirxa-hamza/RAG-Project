"""
FastAPI backend for the RAG system.

Documents come ONLY from the backend's data folder (see config.DATA_DIR) - there is no
upload endpoint, so a user on the frontend can ask questions but can never add or change
what's in the vector store. Drop a PDF into that folder and either restart the server
(ingestion runs on startup) or call POST /ingest to pick it up without a restart.

Endpoints:
  POST /ingest   - (re)scan the data folder and ingest any PDF not already stored
  POST /chat     - ask a question, get an answer grounded in the ingested PDF(s)
  GET  /stats    - see what's currently stored
  POST /reset    - wipe the vector store, then immediately re-ingest the data folder
  GET  /health   - basic liveness check

Run with:  uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import TOP_K
from ingest import ingest_data_folder
from llm import generate_answer
from vectorstore import collection_stats, query_chunks, reset_collection

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


@app.on_event("startup")
def on_startup():
    """Ingests whatever is already sitting in the data folder when the server boots."""
    results = ingest_data_folder()
    for r in results:
        print(f"[startup] {r}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    """Re-scans the data folder and ingests any PDF that isn't already stored. No file
    upload - this only ever reads what's already on disk on the backend."""
    return {"results": ingest_data_folder()}


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
    """Wipes the vector store, then immediately re-ingests everything in the data
    folder - a clean rebuild, not a way to make documents disappear (they only leave
    the store if you also remove them from the data folder)."""
    reset_collection()
    results = ingest_data_folder()
    return {"status": "vector store cleared and re-ingested", "results": results}
