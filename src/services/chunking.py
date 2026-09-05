"""
Chunking strategies, and the one function the rest of the pipeline calls.

`chunk_pages(pages)` dispatches on CHUNK_STRATEGY:

* **fixed** (default) - `pdf.chunk_document()`. Packs whole paragraphs up to
  CHUNK_SIZE_WORDS with CHUNK_OVERLAP_WORDS carried into the next chunk. Boundaries fall
  where the word count runs out.
* **semantic** - the chunker in this module. Boundaries fall where the *meaning* changes:
  every sentence is embedded, the distance between neighbouring sentences is measured, and
  a cut is made wherever that distance spikes.

WHY BOTHER. The fixed packer's failure mode is a boundary that lands in the middle of an
argument: half the explanation in chunk N, half in chunk N+1, and neither one retrieves
well on its own. Overlap and NEIGHBOR_EXPANSION are both patches over that. The semantic
chunker aims at the boundary itself.

WHAT IT COSTS. One embedding per sentence at ingest, against one per ~300-word chunk for
the fixed packer - roughly 15x the embedding calls. On a local model that is CPU minutes.
On a metered API it is money, which is why _warn_about_cost() exists.

HONEST EXPECTATIONS. Published comparisons (Chroma's chunking study among them) mostly
find semantic chunking inside the noise of a well-tuned fixed chunker on structured
documents, and clearly ahead only where formatting carries no signal - transcripts, chat
logs, OCR with the paragraph breaks gone. This corpus is textbooks with intact paragraphs,
so "no significant difference" is a likely and legitimate result. That is what
`scripts/ab_chunking.py` is for: measure it here rather than believing either of us.

NOT INTERCHANGEABLE. The two strategies produce different chunk text, therefore different
vectors. Vectors from one are not comparable with vectors from the other, so switching
CHUNK_STRATEGY requires a full re-ingest into a collection of its own. Mixing them in one
store does not error - it just quietly ruins every number you measure afterwards.
"""
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNK_STRATEGY,
    EMBEDDINGS_PROVIDER,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_BUFFER_SIZE,
    SEMANTIC_MAX_CHUNK_WORDS,
    SEMANTIC_MIN_CHUNK_WORDS,
)
from src.core.logging import get_logger, timed
from src.services.pdf import chunk_document, sentences_with_pages

log = get_logger(__name__)


# ---------------------------------------------------------------- maths, without numpy
# Deliberately dependency-free. numpy is present in local mode (sentence-transformers pulls
# it in) but is not guaranteed in the cloud bundle, and a chunker that only works on one
# deployment target is worse than a slightly slower loop. A few thousand sentences is
# nothing next to the embedding calls that produced their vectors.

def _percentile(values: Sequence[float], percentile: float) -> float:
    """
    Linear-interpolation percentile, matching numpy.percentile's default method so the
    threshold is the one every write-up of this technique is describing.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (max(0.0, min(100.0, percentile)) / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """
    1 - cosine similarity, clamped to [0, 2].

    Normalisation is done here rather than assumed: the local provider returns unit vectors
    (normalize_embeddings=True) but the HTTP providers make no such promise, and an
    unnormalised dot product would make the distances depend on vector magnitude - i.e. on
    sentence length - which is exactly the signal this must not pick up.
    """
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        # A zero vector has no direction, so no meaningful distance. Return 0 (= "same
        # topic") rather than 1: an empty embedding is a provider hiccup, not evidence of a
        # topic change, and treating it as one would plant a spurious boundary.
        return 0.0
    similarity = dot / ((norm_a ** 0.5) * (norm_b ** 0.5))
    return max(0.0, min(2.0, 1.0 - similarity))


# ---------------------------------------------------------------- semantic chunker

def _windows(sentences: List[Tuple[str, int]], buffer_size: int) -> List[str]:
    """
    Each sentence widened by `buffer_size` neighbours on each side, for embedding.

    A lone sentence embeds badly - "It does not." has no topic in it at all - and the
    resulting noise reads as a topic change at every such sentence. Embedding a small
    window instead smooths that out; the boundary is still placed *between* sentences i and
    i+1, only the vector used to judge it is wider.
    """
    if buffer_size <= 0:
        return [text for text, _ in sentences]
    out: List[str] = []
    for i in range(len(sentences)):
        low = max(0, i - buffer_size)
        high = min(len(sentences), i + buffer_size + 1)
        out.append(" ".join(text for text, _ in sentences[low:high]))
    return out


def _split_oversized(group: List[int], distances: List[float],
                     sentences: List[Tuple[str, int]], max_words: int) -> List[List[int]]:
    """
    Breaks a group of sentence indices that exceeds max_words into pieces that don't.

    The cut is made at the largest INTERNAL distance - the weakest seam still inside the
    group - so an over-long group is divided at its next-most-plausible topic boundary
    rather than at an arbitrary word count. Iterative, not recursive: a page of single-word
    lines could otherwise recurse hundreds deep.
    """
    pending = [group]
    done: List[List[int]] = []
    while pending:
        current = pending.pop()
        words = sum(len(sentences[i][0].split()) for i in current)
        if words <= max_words or len(current) < 2:
            done.append(current)
            continue
        # distances[i] is the gap between sentence i and i+1, so the seams inside this
        # group are distances[current[0] .. current[-1] - 1].
        seams = [(distances[i], i) for i in range(current[0], current[-1])]
        _, at = max(seams)
        cut = at - current[0] + 1
        pending.append(current[cut:])
        pending.append(current[:cut])
    done.sort(key=lambda g: g[0])
    return done


def _merge_undersized(groups: List[List[int]], distances: List[float],
                      sentences: List[Tuple[str, int]], min_words: int,
                      max_words: int) -> List[List[int]]:
    """
    Folds a group under min_words into whichever NEIGHBOUR it is more similar to.

    Direction matters and is cheap to get right: a stray heading belongs with the section
    it introduces, not with the one that just ended, and the distance either side already
    says which. A merge that would push the result past max_words is skipped - being
    slightly too small beats being too large to embed.
    """
    if len(groups) < 2:
        return groups

    words_of = lambda g: sum(len(sentences[i][0].split()) for i in g)  # noqa: E731
    out = [list(g) for g in groups]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for position, group in enumerate(out):
            if words_of(group) >= min_words:
                continue
            before = out[position - 1] if position > 0 else None
            after = out[position + 1] if position + 1 < len(out) else None
            # Distance across each seam; None where there is no neighbour on that side.
            gap_before = distances[group[0] - 1] if before else None
            gap_after = distances[group[-1]] if after else None

            options = []
            if before is not None and words_of(before) + words_of(group) <= max_words:
                options.append((gap_before, position - 1, position))
            if after is not None and words_of(after) + words_of(group) <= max_words:
                options.append((gap_after, position, position + 1))
            if not options:
                continue
            # Smallest gap = most similar neighbour.
            _, left, right = min(options, key=lambda o: o[0])
            out[left] = out[left] + out[right]
            del out[right]
            changed = True
            break
    return out


def _warn_about_cost(sentence_count: int) -> None:
    if EMBEDDINGS_PROVIDER != "local":
        log.warning(
            "Semantic chunking embeds every sentence: %d extra embedding calls to '%s' for "
            "this document alone, on top of the chunk embeddings. On a metered or "
            "free-tier API this is the expensive part of the run - consider running the "
            "chunking experiment with EMBEDDINGS_PROVIDER=local.",
            sentence_count, EMBEDDINGS_PROVIDER,
        )


def semantic_chunks(
    pages: List[Dict],
    percentile: Optional[float] = None,
    buffer_size: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    embed=None,
) -> List[Dict]:
    """
    Splits `pages` where the meaning changes rather than where the word count runs out.

    1. Flatten to sentences, keeping each one's page.
    2. Embed each sentence together with `buffer_size` neighbours either side.
    3. Distance between consecutive windows = 1 - cosine similarity.
    4. Cut wherever that distance exceeds the `percentile`-th percentile of this document's
       own distances. A percentile rather than a fixed threshold because the absolute
       numbers move with the document and the embedding model; a constant that behaves on
       one book cuts every other sentence in the next.
    5. Enforce the size bounds - split what is too big at its weakest internal seam, merge
       what is too small into its more similar neighbour.

    `embed` is injectable purely so the offline tests can drive this with a deterministic
    stub instead of loading a 130MB model.

    Returns the same shape as chunk_document(): [{"text", "page_start", "page_end"}, ...].
    NOTE there is no overlap between semantic chunks, by design - an overlap would smear
    the boundary this whole strategy exists to place. NEIGHBOR_EXPANSION still widens the
    context at retrieval time, so the LLM does not see a chunk in isolation.
    """
    percentile = SEMANTIC_BREAKPOINT_PERCENTILE if percentile is None else percentile
    buffer_size = SEMANTIC_BUFFER_SIZE if buffer_size is None else buffer_size
    min_words = SEMANTIC_MIN_CHUNK_WORDS if min_words is None else min_words
    max_words = SEMANTIC_MAX_CHUNK_WORDS if max_words is None else max_words

    if embed is None:
        from src.ml.embeddings import embed_passages
        embed = embed_passages

    sentences = sentences_with_pages(pages, max_words)
    if not sentences:
        return []
    if len(sentences) == 1:
        text, page = sentences[0]
        return [{"text": text, "page_start": page, "page_end": page}]

    _warn_about_cost(len(sentences))
    with timed(log, f"semantic chunking: embed {len(sentences)} sentences"):
        vectors = embed(_windows(sentences, buffer_size))

    if len(vectors) != len(sentences):
        # A provider that silently drops inputs would misalign every boundary after the
        # gap, which is worse than not doing this at all. Fall back rather than guess.
        log.error(
            "Embedding returned %d vectors for %d sentences; falling back to fixed "
            "chunking for this document.", len(vectors), len(sentences),
        )
        return chunk_document(pages, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

    distances = [_cosine_distance(vectors[i], vectors[i + 1])
                 for i in range(len(vectors) - 1)]
    threshold = _percentile(distances, percentile)

    groups: List[List[int]] = []
    current: List[int] = [0]
    for i, distance in enumerate(distances):
        # Strictly greater: with a flat distance profile (a uniform document, or a stub
        # embedder in a test) threshold equals every distance, and `>=` would cut at every
        # single sentence.
        if distance > threshold:
            groups.append(current)
            current = []
        current.append(i + 1)
    groups.append(current)

    sized: List[List[int]] = []
    for group in groups:
        sized.extend(_split_oversized(group, distances, sentences, max_words))
    sized = _merge_undersized(sized, distances, sentences, min_words, max_words)

    chunks: List[Dict] = []
    for group in sized:
        pages_in = [sentences[i][1] for i in group]
        chunks.append({
            "text": " ".join(sentences[i][0] for i in group),
            "page_start": min(pages_in),
            "page_end": max(pages_in),
        })

    log.info(
        "Semantic chunking: %d sentences -> %d chunks (breakpoint p%.0f = %.4f, "
        "median gap %.4f).",
        len(sentences), len(chunks), percentile, threshold, _percentile(distances, 50),
    )
    return chunks


# ---------------------------------------------------------------- dispatch + cache

# Ingestion re-extracts and re-chunks on every resumed slice, relying on chunking being
# deterministic so slice N sees the same chunk list slice N-1 saw. That is free for the
# fixed packer and emphatically not free here - it would re-embed every sentence of the
# document on every slice. Both strategies are still deterministic; this cache just stops
# the semantic one paying for that determinism repeatedly.
#
# Keyed by the page text itself, so a re-uploaded or edited document can never collide with
# a stale entry. Small and bounded: these lists are the whole document in memory.
_CACHE_LIMIT = 3
_cache: "dict[str, List[Dict]]" = {}
_cache_order: List[str] = []


def _cache_key(pages: List[Dict]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{CHUNK_STRATEGY}|{CHUNK_SIZE_WORDS}|{CHUNK_OVERLAP_WORDS}|"
                  f"{SEMANTIC_BREAKPOINT_PERCENTILE}|{SEMANTIC_BUFFER_SIZE}|"
                  f"{SEMANTIC_MIN_CHUNK_WORDS}|{SEMANTIC_MAX_CHUNK_WORDS}".encode())
    for page in pages:
        digest.update(str(page.get("page", "")).encode())
        digest.update(page.get("text", "").encode("utf-8", "replace"))
    return digest.hexdigest()


def clear_cache() -> None:
    """Drop the memoised chunk lists. For tests and for scripts that flip the strategy."""
    _cache.clear()
    _cache_order.clear()


def strategy_name() -> str:
    """Which chunker is in use. Reported on /info and asserted on by the tests."""
    return CHUNK_STRATEGY


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    """
    Chunk a document with whichever strategy is configured.

    This is the only chunking entry point the pipeline should call; `chunk_document()` and
    `semantic_chunks()` stay importable so the A/B script can run both in one process.
    """
    if CHUNK_STRATEGY == "fixed":
        # No cache: the fixed packer is pure string work and memoising a whole document to
        # save it would cost more memory than it saves time.
        return chunk_document(pages, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

    key = _cache_key(pages)
    if key in _cache:
        log.debug("Semantic chunks served from cache (resumed slice).")
        return _cache[key]

    chunks = semantic_chunks(pages)
    _cache[key] = chunks
    _cache_order.append(key)
    while len(_cache_order) > _CACHE_LIMIT:
        _cache.pop(_cache_order.pop(0), None)
    return chunks
