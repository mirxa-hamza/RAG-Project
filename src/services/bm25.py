"""
Keyword (BM25) half of hybrid retrieval.

Why it exists: dense embeddings are weakest exactly where a textbook is strongest - exact
technical terms, notation, named algorithms. Ask "what is A* search" and a pure-vector
store happily returns general search-algorithm prose that never mentions A*. BM25 scores
on literal term overlap, so it nails those; it is correspondingly bad at paraphrase, which
is what the vector side is for. Fusing the two ranked lists (see retrieval.py) gets both.

The index lives in memory and is built lazily on first use, then reused. Building it means
reading every chunk out of Chroma once per process - deliberate: an O(corpus) read at
first query is fine, an O(corpus) read per query would not be. `invalidate()` is called
whenever ingestion changes the store.
"""
import re
import threading
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.core.logging import get_logger, timed

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9*+\-_.]*")

_lock = threading.Lock()
_index: Optional[BM25Okapi] = None
_rows: List[Dict] = []          # parallel to the index: the chunk each row scores


def tokenize(text: str) -> List[str]:
    """
    Lowercase word tokens, keeping the characters that carry meaning in technical text:
    `a*`, `k-means`, `f1`, `p(x|y)` -> `p`, `x`, `y`. A plain \\w+ split would turn "A*"
    into "a", which is a stopword-ish token that matches everything.
    """
    return _TOKEN.findall(text.lower())


def invalidate() -> None:
    """Drops the cached index. Call after anything that changes stored chunks."""
    global _index, _rows
    with _lock:
        _index, _rows = None, []
    log.debug("BM25 index invalidated.")


def _build() -> None:
    global _index, _rows
    # Imported here rather than at module scope to avoid a circular import
    # (vectorstore -> retrieval -> bm25 -> vectorstore).
    from src.services.vectorstore import all_chunks

    with timed(log, "build BM25 index"):
        rows = all_chunks()
        if not rows:
            _index, _rows = None, []
            return
        _index = BM25Okapi([tokenize(r["text"]) for r in rows])
        _rows = rows
    log.info("BM25 index built over %d chunks.", len(_rows))


def search(question: str, limit: int, source: Optional[str] = None,
           user_id: Optional[str] = None) -> List[Tuple[Dict, float]]:
    """
    Returns [(chunk, score), ...] best first. Empty list if the store is empty.

    ISOLATION POINT 2 of 3. The index itself is built over EVERY chunk in the store -
    rebuilding it per user would mean an O(corpus) read per request - so ownership is
    enforced when filtering the ranking, before the limit is applied. Filtering after the
    slice would silently return fewer results than asked for whenever another user's
    chunks rank highly, so the order here matters.
    """
    with _lock:
        if _index is None:
            _build()
        index, rows = _index, _rows

    if index is None or not rows:
        return []

    tokens = tokenize(question)
    if not tokens:
        return []

    scores = index.get_scores(tokens)
    ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)

    if user_id:
        ranked = [pair for pair in ranked if pair[0].get("user_id") == user_id]
    if source:
        ranked = [pair for pair in ranked if pair[0].get("source") == source]

    # A zero score means no query term appears in the chunk - not a weak match, no match.
    return [pair for pair in ranked[:limit] if pair[1] > 0]
