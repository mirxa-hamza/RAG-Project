"""
Step 3 of the pipeline: store chunk vectors so we can later find the ones closest to a
question.

This module is the interface the rest of the app talks to. The actual store is chosen by
VECTOR_STORE (which defaults from RAG_MODE):

* **chroma** (`src/services/vector_chroma.py`) - development. A folder on disk, no server,
  no network, and no bill. Single-process only, which is why the app runs `--workers 1`.
* **pinecone** (`src/services/vector_pinecone.py`) - production. A hosted index, which is
  what makes serverless deployment possible: a Vercel function has no persistent disk, so
  a folder-backed store is empty again on every cold start.

The two backends are NOT interchangeable at runtime for the same data - vectors written by
one are not readable by the other, and the embedding models differ. Switching modes means
re-indexing, which is what `python scripts/ingest.py --force` is for.

Every function below is a thin pass-through. It exists so that the isolation contract is
stated in exactly one place, and so a new backend has one list of things it must implement:

    add_chunks, delete_source, query_chunks, get_neighbors_bulk, get_neighbors,
    set_owner, all_chunks, count, reset_collection

ISOLATION. Three of those reach stored text and MUST filter by owner: query_chunks,
get_neighbors_bulk, and all_chunks (which feeds BM25). A backend that ignores `user_id` in
any of them leaks one account's documents into another's answers, silently and without an
error anywhere. src/services/retrieval.py re-asserts ownership on the way out as a second
line of defence, but that is a net, not the fix.
"""
import threading
from typing import Dict, List, Optional, Set, Tuple

from src.core.config import VECTOR_STORE
from src.core.logging import get_logger

log = get_logger(__name__)

_impl = None
_impl_lock = threading.Lock()


def backend():
    """
    The active backend module, imported on first use.

    Lazily, because importing the Pinecone backend pulls in its SDK and opening the index
    is a network call - doing either at import time means a missing key or a slow network
    stops the process before it can serve even the login page.
    """
    global _impl
    if _impl is not None:
        return _impl
    with _impl_lock:
        if _impl is None:
            if VECTOR_STORE == "pinecone":
                from src.services import vector_pinecone as module
            elif VECTOR_STORE == "chroma":
                from src.services import vector_chroma as module
            else:
                raise ValueError(
                    f"VECTOR_STORE must be 'chroma' or 'pinecone', got {VECTOR_STORE!r}"
                )
            _impl = module
            log.info("Vector store backend: %s", VECTOR_STORE)
    return _impl


def backend_name() -> str:
    return VECTOR_STORE


def reset_backend() -> None:
    """Drop the cached backend. Only for tests that flip the configuration."""
    global _impl
    with _impl_lock:
        _impl = None


def add_chunks(source_name: str, chunks: List[Dict], on_progress=None,
               user_id: Optional[str] = None, index_offset: int = 0) -> int:
    """
    Embeds and stores the chunks of one document, in batches, and returns how many landed.

    `user_id` is the owner stamped onto every chunk's metadata - the single source of truth
    for isolation. A chunk stored without it is visible only to the unfiltered internal
    callers (the CLI, the eval harness).

    `on_progress(stored, total)` is called after each batch; it drives the per-document
    progress bar, without which a 1,900-chunk book is a five-minute silence.

    `index_offset` is where this batch of chunks sits within the whole document. It exists
    because cloud mode ingests a large PDF as several separate requests (see
    ingestion.ingest_one's `start_chunk`): each call passes only its own slice, and without
    the offset every slice would number its chunks from 0 again - colliding with the
    previous slice's `chunk_index` and corrupting neighbour expansion, which addresses
    chunks by that index.
    """
    return backend().add_chunks(source_name, chunks, on_progress=on_progress,
                                user_id=user_id, index_offset=index_offset)


def delete_source(source_name: str, user_id: Optional[str] = None) -> None:
    """Removes every chunk of one document (a changed PDF, or a deleted one)."""
    return backend().delete_source(source_name, user_id=user_id)


def query_chunks(question: str, top_k: int = 4, source=None,
                 user_id: Optional[str] = None) -> List[Dict]:
    """
    Vector search. ISOLATION POINT 1 of 3 - `user_id` must be passed on request paths.

    `source` is one document name or a list of them; None searches everything.
    """
    return backend().query_chunks(question, top_k=top_k, source=source, user_id=user_id)


def get_neighbors_bulk(wanted: Dict[str, Set[int]],
                       user_id: Optional[str] = None) -> Dict[Tuple[str, int], Dict]:
    """
    Fetches many neighbouring chunks in one lookup per source, keyed by (source, index).

    ISOLATION POINT 3 of 3. Neighbours are fetched by chunk_index rather than by search, so
    they bypass every filter applied during ranking; without the owner clause, a hit on
    your own document pulls in the adjacent chunk of someone else's file of the same name.
    """
    return backend().get_neighbors_bulk(wanted, user_id=user_id)


def get_neighbors(source: str, chunk_index: int, radius: int = 1,
                  user_id: Optional[str] = None) -> List[Dict]:
    """Single-hit convenience wrapper around get_neighbors_bulk()."""
    return backend().get_neighbors(source, chunk_index, radius=radius, user_id=user_id)


def set_owner(source_name: str, user_id: str) -> int:
    """
    Stamps `user_id` onto every stored chunk of one document, in place.

    Used once, when the first account adopts the documents indexed before accounts existed.
    """
    return backend().set_owner(source_name, user_id)


def all_chunks(user_id: Optional[str] = None) -> List[Dict]:
    """
    Every stored chunk, or one user's, for building a keyword index.

    An O(corpus) read, done once per cached index rather than per query. Scoping it by user
    is what stops one person's upload from making everyone else pay for a rebuild.
    """
    return backend().all_chunks(user_id=user_id)


def count() -> int:
    return backend().count()


def reset_collection() -> None:
    """Wipes every stored vector. The manifest is cleared separately by the caller."""
    return backend().reset_collection()
