"""
Step 2 of the pipeline: turn text into vectors.

Loading a sentence-transformers model takes a few seconds, so we load it once
(module-level singleton) instead of once per request.
"""
from typing import List
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

print(f"[embeddings] Loading model '{EMBEDDING_MODEL}' (first run downloads it, then it's cached)...")
_model = SentenceTransformer(EMBEDDING_MODEL)
print("[embeddings] Model loaded.")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of chunks (used during ingestion). Shows a progress bar for large
    batches - a big PDF can mean thousands of chunks, and this runs on CPU, so without
    visible progress a long ingest looks identical to a hang."""
    embeddings = _model.encode(
        texts, show_progress_bar=len(texts) > 50, convert_to_numpy=True
    )
    return embeddings.tolist()


def embed_query(text: str) -> List[float]:
    """Embed a single user question (used at retrieval time)."""
    embedding = _model.encode([text], show_progress_bar=False, convert_to_numpy=True)
    return embedding[0].tolist()
