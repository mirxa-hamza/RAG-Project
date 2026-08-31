"""
Step 3 of the pipeline: store chunk vectors so we can later find the ones
closest to a question (retrieval). ChromaDB with a persistent client just
writes to a folder on disk - no separate server process needed.
"""
import uuid
from typing import List, Dict
import chromadb
from config import CHROMA_DIR, CHROMA_COLLECTION
from embeddings import embed_texts, embed_query

_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=chromadb.config.Settings(anonymized_telemetry=False),  # silences a noisy, harmless warning
)
_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},  # cosine similarity is the standard choice for sentence-transformers output
)


def add_chunks(source_name: str, chunks: List[Dict]) -> int:
    """
    Embeds and stores chunks from one document.
    chunks: [{"text": "...", "page": 3}, ...]
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = [f"{source_name}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "page": c["page"]} for c in chunks]

    _collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def query_chunks(question: str, top_k: int = 4) -> List[Dict]:
    """Returns the top_k chunks most similar to the question."""
    if _collection.count() == 0:
        return []

    query_embedding = embed_query(question)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, _collection.count()),
    )

    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({
            "text": text,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "distance": distance,
        })
    return hits


def list_sources() -> List[str]:
    """Unique document names currently in the store."""
    if _collection.count() == 0:
        return []
    all_meta = _collection.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in all_meta})


def collection_stats() -> Dict:
    return {"total_chunks": _collection.count(), "sources": list_sources()}


def reset_collection():
    """Wipes everything - useful while testing so old PDFs don't linger."""
    global _collection
    _client.delete_collection(CHROMA_COLLECTION)
    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
