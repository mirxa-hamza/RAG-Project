"""
Step 3 of the pipeline: store chunk vectors so we can later find the ones closest to a
question (retrieval). ChromaDB with a persistent client just writes to a folder on disk -
no separate server process needed.
"""
import uuid
from typing import Dict, List, Optional, Set, Tuple

import chromadb

from src.core.config import CHROMA_ADD_BATCH, CHROMA_COLLECTION, CHROMA_DIR
from src.ml.embeddings import embed_passages, embed_query, warn_if_truncated
from src.core.logging import get_logger, timed

log = get_logger(__name__)

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)


def _get_collection():
    # cosine similarity is the standard choice for sentence-transformers output
    return _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


_collection = _get_collection()


def add_chunks(source_name: str, chunks: List[Dict], on_progress=None) -> int:
    """
    Embeds and stores chunks from one document, in batches.

    Batching matters twice over: `encode()` on thousands of texts at once spikes memory,
    and Chroma enforces a maximum batch size per add() call that a large book would hit.

    chunks: [{"text": "...", "page_start": 3, "page_end": 4}, ...]
    on_progress: optional callable(stored, total), called after each batch. This is what
    drives the per-document progress bar in the UI - without it a 1,900-chunk book is a
    five-minute silence.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    warn_if_truncated(texts)

    run_id = uuid.uuid4().hex[:8]
    stored = 0

    for offset in range(0, len(chunks), CHROMA_ADD_BATCH):
        batch = chunks[offset:offset + CHROMA_ADD_BATCH]
        batch_texts = [c["text"] for c in batch]

        with timed(log, f"embed batch {offset}-{offset + len(batch)} of '{source_name}'"):
            embeddings = embed_passages(batch_texts)

        _collection.add(
            ids=[f"{run_id}_{offset + i}" for i in range(len(batch))],
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=[
                {
                    "source": source_name,
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "chunk_index": offset + i,
                }
                for i, c in enumerate(batch)
            ],
        )
        stored += len(batch)
        log.info("Stored %d/%d chunks of '%s'", stored, len(chunks), source_name)
        if on_progress:
            try:
                on_progress(stored, len(chunks))
            except Exception:  # a broken reporter must never abort an ingest
                log.exception("Progress callback failed")

    _invalidate_keyword_index()
    return stored


def delete_source(source_name: str) -> None:
    """Removes every chunk belonging to one document (used when a PDF changed on disk)."""
    _collection.delete(where={"source": source_name})
    _invalidate_keyword_index()
    log.info("Deleted existing chunks for '%s'", source_name)


def _invalidate_keyword_index() -> None:
    """The BM25 index caches the corpus in memory; anything that changes it must reset it.
    Imported lazily because bm25 reads back from this module."""
    from src.services import bm25
    bm25.invalidate()


def _row(text: str, meta: Dict, distance: Optional[float] = None) -> Dict:
    row = {
        "text": text,
        "source": meta.get("source"),
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "chunk_index": meta.get("chunk_index"),
    }
    if distance is not None:
        row["distance"] = distance
        # Chroma's cosine distance runs 0-2, so a raw `1 - distance` can go negative for a
        # genuinely dissimilar chunk. Clamp it before showing it to anyone.
        row["similarity"] = round(max(0.0, 1.0 - distance), 3)
    return row


def query_chunks(question: str, top_k: int = 4, source: Optional[str] = None) -> List[Dict]:
    """Vector search. Returns the top_k chunks most similar to the question."""
    total = _collection.count()
    if total == 0:
        return []

    with timed(log, "embed query"):
        query_embedding = embed_query(question)

    where = {"source": source} if source else None
    with timed(log, f"vector search (n={top_k}{', source-scoped' if source else ''})"):
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            where=where,
        )

    return [
        _row(text, meta, distance)
        for text, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def get_neighbors_bulk(wanted: Dict[str, Set[int]]) -> Dict[Tuple[str, int], Dict]:
    """
    Fetches many neighbour chunks in ONE query per source, keyed by (source, chunk_index).

    `wanted` maps a source filename to the chunk indices needed from it. Doing this per
    hit meant a query with top_k=8 fired eight separate Chroma round-trips; batching makes
    it one per document regardless of how many hits it contributed.
    """
    found: Dict[Tuple[str, int], Dict] = {}

    for source, indices in wanted.items():
        indices = sorted(i for i in indices if i is not None and i >= 0)
        if not indices:
            continue
        got = _collection.get(
            where={"$and": [{"source": source}, {"chunk_index": {"$in": indices}}]},
            include=["documents", "metadatas"],
        )
        for text, meta in zip(got["documents"], got["metadatas"]):
            row = _row(text, meta)
            found[(row["source"], row["chunk_index"])] = row

    return found


def get_neighbors(source: str, chunk_index: int, radius: int = 1) -> List[Dict]:
    """Single-hit convenience wrapper around get_neighbors_bulk()."""
    if chunk_index is None or radius < 1:
        return []

    wanted = {i for i in range(chunk_index - radius, chunk_index + radius + 1)
              if i >= 0 and i != chunk_index}
    found = get_neighbors_bulk({source: wanted})
    return list(found.values())


def all_chunks() -> List[Dict]:
    """
    Every stored chunk, for building the in-memory BM25 index.

    This is an O(corpus) read and is called once per process (see bm25.py), not per query.
    """
    if _collection.count() == 0:
        return []
    got = _collection.get(include=["documents", "metadatas"])
    return [_row(text, meta) for text, meta in zip(got["documents"], got["metadatas"])]


def count() -> int:
    return _collection.count()


def reset_collection() -> None:
    """Wipes every stored vector. The manifest is cleared separately by the caller."""
    global _collection
    _client.delete_collection(CHROMA_COLLECTION)
    _collection = _get_collection()
    _invalidate_keyword_index()
    log.info("Vector store cleared.")
