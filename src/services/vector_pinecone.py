"""
The production vector store: Pinecone.

Same interface as src/services/vector_chroma.py (see src/services/vectorstore.py for the
contract). The differences that actually shaped this file:

* **No documents field.** Chroma stores the chunk text for you; Pinecone stores only
  vectors and metadata, so the text rides along in metadata under "text". Pinecone caps
  metadata at 40KB per record, which a ~300-word chunk is nowhere near.
* **Deterministic ids.** Chroma got random ids and relied on delete-then-add. Here an id is
  `<sha1 of the source name>-<chunk_index>`, so re-ingesting a document overwrites its own
  chunks instead of duplicating them, and every chunk can be addressed without a search.
  That is what makes neighbour expansion a plain fetch rather than a second query.
* **Deleting by prefix, not by filter.** Pinecone's delete-by-metadata is documented but
  rate-limited (5 requests/second/namespace) and has not always been available on
  serverless indexes. Listing ids by prefix and deleting those works everywhere, and the
  ids above make the prefix exact.
* **Scores, not distances.** A cosine index returns similarity directly (higher is better),
  where Chroma returns a distance. Both are normalised into the same row shape so nothing
  downstream has to know which store answered.

ISOLATION. Every read here filters by `user_id`: the query passes it to Pinecone as a
metadata filter, and the fetch-based paths (neighbours, ownership stamping) check it in
Python after the fetch, because fetch-by-id takes no filter. A fetch without that check
would hand one account the chunk next to another account's hit.
"""
import hashlib
import threading
from typing import Dict, List, Optional, Set, Tuple

from src.core.config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_EMBED_DIM,
    PINECONE_INDEX,
    PINECONE_REGION,
    PINECONE_UPSERT_BATCH,
)
from src.core.logging import get_logger, timed
from src.ml.embeddings import embed_passages, embed_query, warn_if_truncated

log = get_logger(__name__)

# Pinecone paginates id listings; this is how many to ask for at a time.
_LIST_PAGE = 100
# A guard on all_chunks(), which is the one O(corpus) read in the interface. Without a cap,
# a large corpus turns one keyword-index rebuild into thousands of fetches.
MAX_SCAN_CHUNKS = 20000

_index = None
_lock = threading.Lock()


def _source_prefix(source_name: str) -> str:
    """
    The id prefix for one document.

    Hashed rather than used raw: a source name is a path ("users/<id>/book.pdf") and can
    hold characters and lengths Pinecone does not accept in an id.
    """
    return hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:16] + "-"


def _chunk_id(source_name: str, chunk_index: int) -> str:
    return f"{_source_prefix(source_name)}{chunk_index}"


def _get_index():
    """
    The index handle, opened on first use, creating the index if it isn't there yet.

    Creating it here rather than in a setup script means a fresh deployment with a fresh
    API key works on its own - there is no shell on a serverless host to run a setup step
    from.
    """
    global _index
    if _index is not None:
        return _index

    with _lock:
        if _index is None:
            if not PINECONE_API_KEY:
                raise RuntimeError(
                    "VECTOR_STORE=pinecone but PINECONE_API_KEY is not set. Create a free "
                    "key at https://app.pinecone.io and put it in .env."
                )
            from pinecone import Pinecone, ServerlessSpec

            client = Pinecone(api_key=PINECONE_API_KEY)
            if not client.has_index(PINECONE_INDEX):
                log.info("Creating Pinecone index '%s' (%d dimensions, cosine)...",
                         PINECONE_INDEX, PINECONE_EMBED_DIM)
                client.create_index(
                    name=PINECONE_INDEX,
                    # An index's dimension is fixed at creation. If PINECONE_EMBED_MODEL is
                    # changed later, PINECONE_EMBED_DIM must change with it AND the index
                    # must be recreated - vectors of the wrong width are rejected outright.
                    dimension=PINECONE_EMBED_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                )
            _index = client.Index(PINECONE_INDEX)
            log.info("Pinecone index '%s' ready.", PINECONE_INDEX)
    return _index


def _row(meta: Dict, score: Optional[float] = None) -> Dict:
    """One stored chunk in the shape the rest of the pipeline expects."""
    meta = meta or {}
    row = {
        "text": meta.get("text"),
        "source": meta.get("source"),
        "page_start": _as_int(meta.get("page_start")),
        "page_end": _as_int(meta.get("page_end")),
        "chunk_index": _as_int(meta.get("chunk_index")),
        # Carried through so keyword ranking and the post-retrieval ownership assertion can
        # both see it.
        "user_id": meta.get("user_id"),
    }
    if score is not None:
        # A cosine index scores -1..1; clamp so a genuinely dissimilar chunk cannot show a
        # negative "similarity" to a user.
        row["similarity"] = round(max(0.0, float(score)), 3)
        row["distance"] = round(1.0 - float(score), 3)
    return row


def _as_int(value):
    """Pinecone returns every numeric metadata value as a float."""
    return None if value is None else int(value)


def _invalidate_keyword_index(user_id: Optional[str] = None) -> None:
    """Anything that changes the store must reset the cached keyword index."""
    from src.services import bm25
    bm25.invalidate(user_id)


def owner_filter(user_id: Optional[str]) -> Optional[Dict]:
    """
    The metadata filter that limits a query to one user's documents.

    None means "no filter" and is only correct for internal callers - never pass None on a
    request path, or one user's chunks answer another user's question.
    """
    return {"user_id": {"$eq": user_id}} if user_id else None


def _and(*clauses) -> Optional[Dict]:
    present = [c for c in clauses if c]
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return {"$and": present}


def add_chunks(source_name: str, chunks: List[Dict], on_progress=None,
               user_id: Optional[str] = None) -> int:
    """Embeds and upserts the chunks of one document. Returns how many were stored."""
    if not chunks:
        return 0

    warn_if_truncated([c["text"] for c in chunks])
    index = _get_index()
    stored = 0

    for offset in range(0, len(chunks), PINECONE_UPSERT_BATCH):
        batch = chunks[offset:offset + PINECONE_UPSERT_BATCH]
        texts = [c["text"] for c in batch]

        with timed(log, f"embed batch {offset}-{offset + len(batch)} of '{source_name}'"):
            embeddings = embed_passages(texts)

        index.upsert(vectors=[
            {
                "id": _chunk_id(source_name, offset + i),
                "values": embeddings[i],
                "metadata": {
                    "text": chunk["text"],
                    "source": source_name,
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "chunk_index": offset + i,
                    # An ownerless chunk simply has no user_id key, which is exactly what
                    # owner_filter never matches.
                    **({"user_id": user_id} if user_id else {}),
                },
            }
            for i, chunk in enumerate(batch)
        ])

        stored += len(batch)
        log.info("Stored %d/%d chunks of '%s'", stored, len(chunks), source_name)
        if on_progress:
            try:
                on_progress(stored, len(chunks))
            except Exception:  # a broken reporter must never abort an ingest
                log.exception("Progress callback failed")

    _invalidate_keyword_index(user_id)
    return stored


def _page_ids(page) -> List[str]:
    """
    The ids out of one page of index.list().

    The shape has changed across SDK versions - current releases yield a page object whose
    `.vectors` holds records with an `.id`, older ones yielded a bare list of id strings -
    and getting it wrong produces an empty list rather than an error, which would silently
    turn "delete this document" into a no-op. So all three shapes are handled.
    """
    if isinstance(page, list):
        return [str(item) for item in page]

    records = getattr(page, "vectors", None)
    if records is None and isinstance(page, dict):
        records = page.get("vectors")
    if records:
        return [r.get("id") if isinstance(r, dict) else getattr(r, "id", None)
                for r in records if (r.get("id") if isinstance(r, dict)
                                     else getattr(r, "id", None))]

    ids = getattr(page, "ids", None)
    if ids is None and isinstance(page, dict):
        ids = page.get("ids")
    return [str(i) for i in ids or []]


def _ids_for_source(source_name: str) -> List[str]:
    """Every stored id belonging to one document, via prefix listing."""
    index = _get_index()
    prefix = _source_prefix(source_name)
    ids: List[str] = []
    for page in index.list(prefix=prefix, limit=_LIST_PAGE):
        ids.extend(_page_ids(page))
    return ids


def delete_source(source_name: str, user_id: Optional[str] = None) -> None:
    """
    Removes every chunk of one document.

    By id rather than by metadata filter: filter-deletes are rate-limited and were not
    always supported on serverless indexes, while ids derived from the source name make the
    prefix listing exact.
    """
    ids = _ids_for_source(source_name)
    if not ids:
        return

    index = _get_index()
    # Pinecone accepts at most 1000 ids per delete.
    for offset in range(0, len(ids), 1000):
        index.delete(ids=ids[offset:offset + 1000])

    _invalidate_keyword_index()
    log.info("Deleted %d chunk(s) for '%s'", len(ids), source_name)


def source_filter(source) -> Optional[Dict]:
    """
    The metadata filter limiting a search to chosen documents - one name or several.

    An empty list means "no restriction", not "match nothing".
    """
    if not source:
        return None
    if isinstance(source, str):
        return {"source": {"$eq": source}}
    names = [s for s in source if s]
    if not names:
        return None
    return {"source": {"$eq": names[0]}} if len(names) == 1 else {"source": {"$in": names}}


def query_chunks(question: str, top_k: int = 4, source=None,
                 user_id: Optional[str] = None) -> List[Dict]:
    """Vector search. ISOLATION POINT 1 of 3."""
    index = _get_index()

    with timed(log, "embed query"):
        vector = embed_query(question)

    where = _and(source_filter(source), owner_filter(user_id))
    with timed(log, f"vector search (n={top_k}{', source-scoped' if source else ''})"):
        result = index.query(
            vector=vector,
            top_k=max(1, top_k),
            filter=where,
            include_metadata=True,
        )

    matches = result.get("matches") if isinstance(result, dict) else result.matches
    return [_row(_match_metadata(m), _match_score(m)) for m in matches or []]


def _match_metadata(match):
    return match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", {})


def _match_score(match):
    return match.get("score") if isinstance(match, dict) else getattr(match, "score", None)


def _fetch_rows(ids: List[str]) -> List[Dict]:
    """Fetch by id, in batches, returning rows in no particular order."""
    if not ids:
        return []
    index = _get_index()
    rows: List[Dict] = []
    for offset in range(0, len(ids), 100):
        response = index.fetch(ids=ids[offset:offset + 100])
        vectors = (response.get("vectors") if isinstance(response, dict)
                   else getattr(response, "vectors", {})) or {}
        for record in vectors.values():
            meta = (record.get("metadata") if isinstance(record, dict)
                    else getattr(record, "metadata", {})) or {}
            rows.append(_row(meta))
    return rows


def get_neighbors_bulk(wanted: Dict[str, Set[int]],
                       user_id: Optional[str] = None) -> Dict[Tuple[str, int], Dict]:
    """
    Fetches neighbouring chunks by id - no search involved, because the ids are derived
    from (source, chunk_index).

    ISOLATION POINT 3 of 3, and the one that has to be enforced in Python: fetch-by-id
    takes no metadata filter, so the owner check happens here on the way out. Skipping it
    would let a hit on your own document pull in the adjacent chunk of a same-named file
    belonging to somebody else.
    """
    ids: List[str] = []
    for source, indices in wanted.items():
        ids.extend(_chunk_id(source, i)
                   for i in sorted(x for x in indices if x is not None and x >= 0))

    found: Dict[Tuple[str, int], Dict] = {}
    for row in _fetch_rows(ids):
        if user_id and row.get("user_id") != user_id:
            continue
        if row.get("source") is None or row.get("chunk_index") is None:
            continue
        found[(row["source"], row["chunk_index"])] = row
    return found


def get_neighbors(source: str, chunk_index: int, radius: int = 1,
                  user_id: Optional[str] = None) -> List[Dict]:
    """Single-hit convenience wrapper around get_neighbors_bulk()."""
    if chunk_index is None or radius < 1:
        return []
    wanted = {i for i in range(chunk_index - radius, chunk_index + radius + 1)
              if i >= 0 and i != chunk_index}
    return list(get_neighbors_bulk({source: wanted}, user_id=user_id).values())


def set_owner(source_name: str, user_id: str) -> int:
    """
    Stamps `user_id` onto every stored chunk of one document.

    One update call per chunk - Pinecone has no bulk metadata update. That is acceptable
    only because this runs once, when the first account adopts pre-account documents.
    """
    ids = _ids_for_source(source_name)
    if not ids:
        return 0

    index = _get_index()
    for chunk_id in ids:
        index.update(id=chunk_id, set_metadata={"user_id": user_id})

    _invalidate_keyword_index()   # the cached rows still carry the old metadata
    log.info("Stamped %d chunk(s) of '%s' with user_id=%s", len(ids), source_name, user_id)
    return len(ids)


def all_chunks(user_id: Optional[str] = None) -> List[Dict]:
    """
    Every stored chunk, or one user's, for building a keyword index.

    This is the expensive one. Pinecone has no "get everything" call, so it lists ids and
    fetches them in batches - fine for a personal corpus, and capped at MAX_SCAN_CHUNKS so
    a large one degrades the keyword half of hybrid search instead of hanging a request.

    A sparse Pinecone index would do this job server-side and is the proper fix; see
    HYBRID_ENABLED if you would rather turn the keyword stage off in production than pay
    for this scan.
    """
    index = _get_index()
    ids: List[str] = []
    for page in index.list(limit=_LIST_PAGE):
        ids.extend(_page_ids(page))
        if len(ids) >= MAX_SCAN_CHUNKS:
            log.warning(
                "Keyword index scan hit its %d-chunk cap; keyword ranking will only see "
                "part of the corpus. Disable HYBRID_ENABLED or move to a sparse index.",
                MAX_SCAN_CHUNKS,
            )
            ids = ids[:MAX_SCAN_CHUNKS]
            break

    rows = _fetch_rows(ids)
    if user_id:
        rows = [r for r in rows if r.get("user_id") == user_id]
    return rows


def count() -> int:
    stats = _get_index().describe_index_stats()
    total = (stats.get("total_vector_count") if isinstance(stats, dict)
             else getattr(stats, "total_vector_count", 0))
    return int(total or 0)


def reset_collection() -> None:
    """Wipes every stored vector. The manifest is cleared separately by the caller."""
    _get_index().delete(delete_all=True)
    _invalidate_keyword_index()
    log.info("Vector store cleared.")
