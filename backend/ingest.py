"""
The only way documents get into this system.

There is deliberately no upload endpoint - a user talking to the frontend can only
ask questions, never add documents. Whoever runs the backend drops PDFs into DATA_DIR
(see config.py) and either restarts the server (ingestion runs on startup) or calls
POST /ingest to pick up new files without a restart.
"""
import os
from typing import Dict, List

from config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS, DATA_DIR
from pdf_utils import chunk_document, extract_pages
from vectorstore import add_chunks, list_sources


def _pdf_filenames() -> List[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))


def ingest_one(filename: str) -> Dict:
    """Extracts, chunks, embeds, and stores a single PDF that's already sitting in DATA_DIR."""
    path = os.path.join(DATA_DIR, filename)
    pages = extract_pages(path)
    if not pages:
        return {
            "filename": filename,
            "status": "skipped",
            "reason": "no extractable text - likely a scanned/image PDF (no OCR yet)",
        }

    chunks = chunk_document(pages, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    print(
        f"[ingest] '{filename}': {len(pages)} pages -> {len(chunks)} chunks. "
        f"Embedding on CPU - large documents can take several minutes, not a hang."
    )
    stored = add_chunks(filename, chunks)
    return {"filename": filename, "status": "ingested", "pages": len(pages), "chunks_stored": stored}


def ingest_data_folder() -> List[Dict]:
    """
    Ingests every PDF in DATA_DIR that isn't already stored (by filename). Safe to call
    repeatedly - already-stored files are reported, not re-embedded - so it can run on
    every startup and be re-triggered any time via POST /ingest.
    """
    already_stored = set(list_sources())
    results = []
    for filename in _pdf_filenames():
        if filename in already_stored:
            results.append({"filename": filename, "status": "already_stored"})
            continue
        print(f"[ingest] Ingesting '{filename}'...")
        result = ingest_one(filename)
        print(f"[ingest] {result}")
        results.append(result)
    return results
