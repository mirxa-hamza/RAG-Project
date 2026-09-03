"""
Keyword (BM25) half of hybrid retrieval.

Why it exists: dense embeddings are weakest exactly where a textbook is strongest - exact
technical terms, notation, named algorithms. Ask "what is A* search" and a pure-vector
store happily returns general search-algorithm prose that never mentions A*. BM25 scores
on literal term overlap, so it nails those; it is correspondingly bad at paraphrase, which
is what the vector side is for. Fusing the two ranked lists (see retrieval.py) gets both.

Indices are PER USER and built lazily, then cached in a small LRU. This is both a
correctness and a cost decision:

* Correctness: BM25 ranks in memory, so it cannot use Chroma's owner filter. A shared index
  has to be filtered after ranking, which is easy to get wrong (filter after the limit and
  you silently return fewer results than asked for).
* Cost: a single shared index is invalidated by ANY user's upload, and the next question
  from anybody pays for a full re-read of every chunk plus a rebuild. Measured:
  0.09s at 3k chunks, 0.52s at 20k, 2.95s at 60k, on top of the Chroma read. Per-user
  indices mean one person's upload only costs that person.

`invalidate(user_id)` drops one user's index; `invalidate()` with no argument drops all of
them, which is what a global change (a reset, a model swap) needs.
"""
import re
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from src.core.logging import get_logger, timed

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9*+\-_.]*")

# How many users' indices to keep in memory at once. Each is roughly the size of that
# user's text, so this is the memory/rebuild trade-off knob.
MAX_CACHED_INDICES = 8

_lock = threading.Lock()
# user_id (or "" for the unfiltered index used by the CLI and the eval harness) ->
# (index, rows). Ordered by use, oldest first, so eviction is plain LRU.
_cache: "OrderedDict[str, Tuple[Optional[BM25Okapi], List[Dict]]]" = OrderedDict()


def tokenize(text: str) -> List[str]:
    """
    Lowercase word tokens, keeping the characters that carry meaning in technical text:
    `a*`, `k-means`, `f1`, `p(x|y)` -> `p`, `x`, `y`. A plain \\w+ split would turn "A*"
    into "a", which is a stopword-ish token that matches everything.
    """
    return _TOKEN.findall(text.lower())


def invalidate(user_id: Optional[str] = None) -> None:
    """
    Drops cached indices. Call after anything that changes stored chunks.

    With a user_id, only that user's index goes - one person's upload must not make
    everyone else's next question pay for a rebuild. Without one, everything goes, which is
    what a store-wide change needs.

    The unfiltered index ("") is always dropped: it contains every user's chunks, so any
    change anywhere invalidates it.
    """
    with _lock:
        if user_id is None:
            _cache.clear()
            log.debug("All BM25 indices invalidated.")
        else:
            _cache.pop(user_id, None)
            _cache.pop("", None)
            log.debug("BM25 index for %s invalidated.", user_id)


_warned_remote = False


def enabled() -> bool:
    """
    Whether the keyword half of hybrid retrieval should run at all.

    BM25 lives in this process, so building its index means reading the WHOLE corpus. That
    is a disk read against a local Chroma folder and a bulk download against a hosted store
    - where it takes minutes on a free tier and can have the connection dropped mid-way.
    Losing the keyword stage costs some ranking quality; failing the question costs the
    answer, so "auto" trades the first away rather than the second.
    """
    global _warned_remote
    from src.core.config import CHROMA_BACKEND, KEYWORD_SEARCH, VECTOR_STORE

    if KEYWORD_SEARCH == "off":
        return False
    if KEYWORD_SEARCH == "on":
        return True

    remote = VECTOR_STORE != "chroma" or CHROMA_BACKEND != "disk"
    if remote and not _warned_remote:
        _warned_remote = True
        log.info(
            "Keyword (BM25) search is off: the corpus lives in a hosted store, and "
            "building the index would download all of it on every cold start. Answers "
            "use vector search plus the re-ranker. Set KEYWORD_SEARCH=on to force it."
        )
    return not remote


def _build(user_id: str) -> Tuple[Optional[BM25Okapi], List[Dict]]:
    """Builds one index. `user_id` of "" means the whole store (CLI and eval only)."""
    # Imported here rather than at module scope to avoid a circular import
    # (vectorstore -> retrieval -> bm25 -> vectorstore).
    from src.services.vectorstore import all_chunks

    try:
        with timed(log, f"build BM25 index ({user_id or 'all documents'})"):
            rows = all_chunks(user_id=user_id or None)
            if not rows:
                return None, []
            index = BM25Okapi([tokenize(r["text"]) for r in rows])
    except Exception as exc:
        # A store that is slow, rate-limited or drops the connection must not take the
        # question down with it. This exact failure - Chroma Cloud closing the connection
        # part-way through the bulk read - surfaced to the user as "Request failed (500)".
        log.warning("Could not build the keyword index (%s) - answering with vector "
                    "search and the re-ranker only.", exc)
        return None, []
    log.info("BM25 index built over %d chunks for %s.", len(rows), user_id or "all documents")
    return index, rows


def _get(user_id: str) -> Tuple[Optional[BM25Okapi], List[Dict]]:
    """The cached index for one user, building it if needed. LRU by last use."""
    with _lock:
        cached = _cache.get(user_id)
        if cached is not None:
            _cache.move_to_end(user_id)
            return cached

    # Built OUTSIDE the lock: it reads the whole corpus and can take seconds, and holding
    # the lock would block every other user's search for that whole time.
    built = _build(user_id)

    with _lock:
        _cache[user_id] = built
        _cache.move_to_end(user_id)
        while len(_cache) > MAX_CACHED_INDICES:
            evicted, _ = _cache.popitem(last=False)
            log.debug("Evicted the BM25 index for %s.", evicted or "all documents")
    return built


def search(question: str, limit: int, source=None,
           user_id: Optional[str] = None) -> List[Tuple[Dict, float]]:
    """
    Returns [(chunk, score), ...] best first. Empty list if the store is empty.

    ISOLATION POINT 2 of 3. Each user gets their own index, so another user's chunks are
    not in the ranking at all. The post-ranking owner filter below is kept anyway as a
    second line of defence - it costs one comparison per row and it is what would catch a
    caching mistake here.
    """
    if not enabled():
        return []

    index, rows = _get(user_id or "")

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
        # One document or several - the UI lets a person tick a subset.
        wanted = {source} if isinstance(source, str) else {s for s in source if s}
        if wanted:
            ranked = [pair for pair in ranked if pair[0].get("source") in wanted]

    # A zero score means no query term appears in the chunk - not a weak match, no match.
    return [pair for pair in ranked[:limit] if pair[1] > 0]
