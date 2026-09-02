"""
Step 3 of the pipeline: store chunk vectors so we can later find the ones closest to a
question (retrieval). ChromaDB with a persistent client just writes to a folder on disk -
no separate server process needed.
"""
import threading
import uuid
from typing import Dict, List, Optional, Set, Tuple

import chromadb

from src.core.config import (
    CHROMA_ADD_BATCH,
    CHROMA_API_KEY,
    CHROMA_BACKEND,
    CHROMA_COLLECTION,
    CHROMA_DATABASE,
    CHROMA_DIR,
    CHROMA_TENANT,
)
from src.ml.embeddings import embed_passages, embed_query, warn_if_truncated
from src.core.logging import get_logger, timed

log = get_logger(__name__)

_client = None
_collection = None
_store_lock = threading.Lock()


def _make_client():
    """
    The Chroma client for the configured backend.

    "disk" is a PersistentClient over a folder: no server, no network, and single-process
    only - which is why this app runs with --workers 1. "cloud" is Chroma Cloud over HTTP,
    which is what makes serverless deployment possible at all: a Vercel function has no
    persistent disk, so a folder-backed store would be empty on every cold start.
    """
    if CHROMA_BACKEND == "cloud":
        missing = [name for name, value in (
            ("CHROMA_API_KEY", CHROMA_API_KEY),
            ("CHROMA_TENANT", CHROMA_TENANT),
            ("CHROMA_DATABASE", CHROMA_DATABASE),
        ) if not value]
        if missing:
            raise RuntimeError(
                "CHROMA_BACKEND=cloud but " + ", ".join(missing) + " not set. "
                "Get them from the Chroma Cloud dashboard and put them in .env."
            )
        log.info("Vector store: Chroma Cloud (tenant %s, database %s).",
                 CHROMA_TENANT, CHROMA_DATABASE)
        return chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY,
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Vector store: local folder %s", CHROMA_DIR)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def _get_collection():
    # cosine similarity is the standard choice for sentence-transformers output
    return _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _col():
    """
    The collection, opened on first use.

    Lazily, not at import: with a cloud backend, opening it is a network call, and doing
    that at import time means a missing key or a slow network stops the process before it
    can serve even the login page. Failing on the first query instead keeps the failure
    where someone can see it.
    """
    global _client, _collection
    if _collection is not None:
        return _collection
    with _store_lock:
        if _collection is None:
            _client = _make_client()
            _collection = _get_collection()
    return _collection


def add_chunks(source_name: str, chunks: List[Dict], on_progress=None,
               user_id: Optional[str] = None) -> int:
    """
    Embeds and stores chunks from one document, in batches.

    Batching matters twice over: `encode()` on thousands of texts at once spikes memory,
    and Chroma enforces a maximum batch size per add() call that a large book would hit.

    user_id: the owner stamped onto every chunk's metadata. This is the single source of
    truth for isolation - a chunk stored without it is visible to nobody except the
    unfiltered internal callers (the CLI, the eval harness).

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

        _col().add(
            ids=[f"{run_id}_{offset + i}" for i in range(len(batch))],
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=[
                {
                    "source": source_name,
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "chunk_index": offset + i,
                    # Chroma rejects a None metadata value, so an ownerless chunk simply
                    # has no user_id key - which is exactly what owner_filter never matches.
                    **({"user_id": user_id} if user_id else {}),
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

    _invalidate_keyword_index(user_id)
    return stored


def delete_source(source_name: str, user_id: Optional[str] = None) -> None:
    """Removes every chunk belonging to one document (used when a PDF changed on disk)."""
    _col().delete(where=_and({"source": source_name}, owner_filter(user_id)))
    _invalidate_keyword_index()
    log.info("Deleted existing chunks for '%s'", source_name)


def _invalidate_keyword_index(user_id: Optional[str] = None) -> None:
    """
    BM25 caches chunk text in memory; anything that changes the store must reset it.

    Scoped when the caller knows whose documents changed, so one upload does not force a
    rebuild for every other user. Imported lazily because bm25 reads back from this module.
    """
    from src.services import bm25
    bm25.invalidate(user_id)


def _row(text: str, meta: Dict, distance: Optional[float] = None) -> Dict:
    row = {
        "text": text,
        "source": meta.get("source"),
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "chunk_index": meta.get("chunk_index"),
        # Carried through so BM25 (which ranks in memory, without Chroma's where-clause)
        # and the post-retrieval ownership assertion can both see it.
        "user_id": meta.get("user_id"),
    }
    if distance is not None:
        row["distance"] = distance
        # Chroma's cosine distance runs 0-2, so a raw `1 - distance` can go negative for a
        # genuinely dissimilar chunk. Clamp it before showing it to anyone.
        row["similarity"] = round(max(0.0, 1.0 - distance), 3)
    return row


def owner_filter(user_id: Optional[str]) -> Optional[Dict]:
    """
    The Chroma `where` clause that limits a query to one user's documents.

    ISOLATION POINT 1 of 3 (the others are bm25.search and get_neighbors_bulk). None means
    "no filter" and is only correct for internal callers - never pass None on a request
    path, or one user's chunks answer another user's question.
    """
    return {"user_id": user_id} if user_id else None


def _and(*clauses) -> Optional[Dict]:
    """Combines Chroma where-clauses; Chroma needs an explicit $and for more than one."""
    present = [c for c in clauses if c]
    if not present:
        return None
    return present[0] if len(present) == 1 else {"$and": present}


def source_filter(source) -> Optional[Dict]:
    """
    The where-clause limiting a search to chosen documents.

    Accepts one name or a list of them, because the UI now lets a person tick several
    documents. An empty list means "no restriction", not "match nothing" - see
    ChatRequest.wanted_sources() for why that distinction matters.
    """
    if not source:
        return None
    if isinstance(source, str):
        return {"source": source}
    names = [s for s in source if s]
    if not names:
        return None
    return {"source": names[0]} if len(names) == 1 else {"source": {"$in": names}}


def query_chunks(question: str, top_k: int = 4, source=None,
                 user_id: Optional[str] = None) -> List[Dict]:
    """Vector search. Returns the top_k chunks most similar to the question."""
    total = _col().count()
    if total == 0:
        return []

    with timed(log, "embed query"):
        query_embedding = embed_query(question)

    where = _and(source_filter(source), owner_filter(user_id))
    with timed(log, f"vector search (n={top_k}{', source-scoped' if source else ''})"):
        results = _col().query(
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


def get_neighbors_bulk(wanted: Dict[str, Set[int]],
                       user_id: Optional[str] = None) -> Dict[Tuple[str, int], Dict]:
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
        # ISOLATION POINT 3 of 3. Neighbours are fetched by chunk_index, not by search, so
        # they bypass every filter applied during ranking. Without the owner clause here, a
        # hit on your own document would happily pull in the adjacent chunk of someone
        # else's document that happens to share a filename.
        got = _col().get(
            where=_and(
                {"source": source},
                {"chunk_index": {"$in": indices}},
                owner_filter(user_id),
            ),
            include=["documents", "metadatas"],
        )
        for text, meta in zip(got["documents"], got["metadatas"]):
            row = _row(text, meta)
            found[(row["source"], row["chunk_index"])] = row

    return found


def get_neighbors(source: str, chunk_index: int, radius: int = 1,
                  user_id: Optional[str] = None) -> List[Dict]:
    """Single-hit convenience wrapper around get_neighbors_bulk()."""
    if chunk_index is None or radius < 1:
        return []

    wanted = {i for i in range(chunk_index - radius, chunk_index + radius + 1)
              if i >= 0 and i != chunk_index}
    found = get_neighbors_bulk({source: wanted}, user_id=user_id)
    return list(found.values())


def set_owner(source_name: str, user_id: str) -> int:
    """
    Stamps `user_id` onto every stored chunk of one document, in place.

    Used once, when the first account adopts the documents that were indexed before
    accounts existed. Chroma has no "update where" - metadata is rewritten by id - so the
    ids and existing metadata are read first and updated in batches.
    """
    got = _col().get(where={"source": source_name}, include=["metadatas"])
    ids = got.get("ids") or []
    if not ids:
        return 0

    metadatas = [dict(meta or {}, user_id=user_id) for meta in got.get("metadatas") or []]
    for offset in range(0, len(ids), CHROMA_ADD_BATCH):
        _col().update(
            ids=ids[offset:offset + CHROMA_ADD_BATCH],
            metadatas=metadatas[offset:offset + CHROMA_ADD_BATCH],
        )

    _invalidate_keyword_index()   # the cached rows still carry the old metadata
    log.info("Stamped %d chunk(s) of '%s' with user_id=%s", len(ids), source_name, user_id)
    return len(ids)


def all_chunks(user_id: Optional[str] = None) -> List[Dict]:
    """
    Every stored chunk, or one user's, for building a BM25 index.

    This is an O(corpus) read, done once per cached index (see bm25.py), not per query.
    Scoping it by user is what keeps one person's upload from making everyone else pay to
    rebuild.
    """
    if _col().count() == 0:
        return []
    got = _col().get(where=owner_filter(user_id), include=["documents", "metadatas"])
    return [_row(text, meta) for text, meta in zip(got["documents"], got["metadatas"])]


def count() -> int:
    return _col().count()


def reset_collection() -> None:
    """Wipes every stored vector. The manifest is cleared separately by the caller."""
    global _collection
    _col()                       # make sure the client exists before deleting through it
    _client.delete_collection(CHROMA_COLLECTION)
    _collection = _get_collection()
    _invalidate_keyword_index()
    log.info("Vector store cleared.")
