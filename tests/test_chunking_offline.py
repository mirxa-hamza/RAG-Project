"""
Offline checks for the chunking strategies. No model download, no network, no API key.

    python tests/test_chunking_offline.py

The embedding model is stubbed with a deterministic fake whose vectors encode a KNOWN
topic structure, so every assertion about "did it cut in the right place" is checkable
rather than a vibe. A real model is exercised by scripts/ab_chunking.py instead - that is a
measurement, this is a correctness test, and mixing the two gives you neither.

Every check here was confirmed to FAIL when the behaviour it guards was deliberately
broken; a test that cannot fail is decoration.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Set before importing config: these are read at import time.
os.environ.setdefault("JWT_SECRET", "offline-test-secret-not-used-anywhere-real")
os.environ["CHUNK_STRATEGY"] = "fixed"

from src.services import chunking  # noqa: E402
from src.services.pdf import chunk_document, sentences_with_pages  # noqa: E402

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}{('  -> ' + str(detail)) if detail else ''}")


# ---------------------------------------------------------------- a stub embedder
# Each sentence in the fixtures below starts with a topic marker ("T1", "T2", ...). The
# stub returns a one-hot vector for that marker, so sentences of the same topic are
# distance 0 apart and sentences of different topics are distance 1 apart. That makes the
# correct boundaries knowable in advance, which is the entire point.
TOPICS = ["T1", "T2", "T3", "T4"]


def stub_embed(texts):
    vectors = []
    for text in texts:
        vector = [0.0] * len(TOPICS)
        for index, topic in enumerate(TOPICS):
            vector[index] = float(text.count(topic))
        if not any(vector):
            vector[0] = 1.0
        vectors.append(vector)
    return vectors


def pages_from(*page_texts):
    return [{"page": i, "text": text} for i, text in enumerate(page_texts, start=1)]


def sentence(topic, n, filler=8):
    """One sentence carrying a topic marker and enough words to be worth counting."""
    return f"{topic} sentence {n} " + " ".join(f"word{w}" for w in range(filler)) + "."


# ---------------------------------------------------------------- sentence splitting
print("\nsentences_with_pages()")

pages = pages_from("First one. Second one.\n\nThird one.", "Fourth one.")
found = sentences_with_pages(pages)
check("splits on sentence boundaries and paragraph breaks", len(found) == 4, found)
check("keeps each sentence's page number",
      [page for _, page in found] == [1, 1, 1, 2],
      [p for _, p in found])
check("carries the sentence text intact", found[0][0] == "First one.", found[0])

huge = "word " * 900
check("caps a runaway unpunctuated block at max_words",
      all(len(text.split()) <= 100 for text, _ in
          sentences_with_pages(pages_from(huge), max_words=100)))
check("empty pages produce no sentences", sentences_with_pages([]) == [])
check("whitespace-only page produces no sentences",
      sentences_with_pages(pages_from("   \n\n  ")) == [])


# ---------------------------------------------------------------- percentile / distance
print("\nmaths helpers")

check("percentile interpolates like numpy's default",
      abs(chunking._percentile([0.0, 1.0, 2.0, 3.0], 50) - 1.5) < 1e-9,
      chunking._percentile([0.0, 1.0, 2.0, 3.0], 50))
check("p100 is the maximum", chunking._percentile([0.1, 0.9, 0.4], 100) == 0.9)
check("p0 is the minimum", chunking._percentile([0.1, 0.9, 0.4], 0) == 0.1)
check("a single value is its own percentile", chunking._percentile([0.7], 95) == 0.7)
check("no values is 0.0, not a crash", chunking._percentile([], 95) == 0.0)

check("identical vectors are distance 0",
      chunking._cosine_distance([1.0, 0.0], [1.0, 0.0]) < 1e-9)
check("orthogonal vectors are distance 1",
      abs(chunking._cosine_distance([1.0, 0.0], [0.0, 1.0]) - 1.0) < 1e-9)
check("a long vector pair is still distance 0 (normalised, not a raw dot product)",
      abs(chunking._cosine_distance([1.0, 0.0], [9.0, 0.0])) < 1e-9)
# Short vectors are the case a raw dot product gets wrong in the other direction: dot is
# 0.01, so an unnormalised implementation would call two IDENTICAL sentences distance 0.99
# and cut between them. The clamp in _cosine_distance hides the long-vector case, so this
# is the check that actually pins normalisation down.
check("a short vector pair is also distance 0",
      abs(chunking._cosine_distance([0.1, 0.0], [0.1, 0.0])) < 1e-9,
      chunking._cosine_distance([0.1, 0.0], [0.1, 0.0]))
check("distance never leaves [0, 2]",
      0.0 <= chunking._cosine_distance([1.0, 2.0], [-3.0, 1.0]) <= 2.0)
check("a zero vector reads as 'no topic change', not as a boundary",
      chunking._cosine_distance([0.0, 0.0], [1.0, 0.0]) == 0.0)


# ---------------------------------------------------------------- semantic boundaries
print("\nsemantic_chunks() - boundary placement")

# Three topics, five sentences each, one per paragraph so sentence splitting is trivial.
body = "\n\n".join(
    sentence(topic, n) for topic in ("T1", "T2", "T3") for n in range(5)
)
doc = pages_from(body)

# buffer_size=0: judge each sentence on its own, so the one-hot stub gives exactly two
# non-zero distances - at the two real topic changes.
chunks = chunking.semantic_chunks(
    doc, percentile=90, buffer_size=0, min_words=1, max_words=10_000, embed=stub_embed)
check("cuts once per topic change and nowhere else", len(chunks) == 3, len(chunks))
check("chunk 1 is entirely topic 1",
      "T1" in chunks[0]["text"] and "T2" not in chunks[0]["text"], chunks[0]["text"][:60])
check("chunk 2 is entirely topic 2",
      "T2" in chunks[1]["text"] and "T1" not in chunks[1]["text"], chunks[1]["text"][:60])
check("chunk 3 is entirely topic 3",
      "T3" in chunks[2]["text"] and "T2" not in chunks[2]["text"], chunks[2]["text"][:60])
check("no sentence is lost",
      sum(c["text"].count("sentence") for c in chunks) == 15,
      sum(c["text"].count("sentence") for c in chunks))
check("no sentence is duplicated (semantic chunks do not overlap)",
      len(" ".join(c["text"] for c in chunks).split("T1 sentence 0")) == 2)

# A document with no topic change at all must not be shredded into one chunk per sentence.
flat = pages_from("\n\n".join(sentence("T1", n) for n in range(12)))
flat_chunks = chunking.semantic_chunks(
    flat, percentile=95, buffer_size=0, min_words=1, max_words=10_000, embed=stub_embed)
check("a uniform document stays one chunk (flat distances are not boundaries)",
      len(flat_chunks) == 1, len(flat_chunks))


# ---------------------------------------------------------------- size bounds
print("\nsemantic_chunks() - size bounds")

sized = chunking.semantic_chunks(
    doc, percentile=90, buffer_size=0, min_words=1, max_words=60, embed=stub_embed)
check("no chunk exceeds max_words",
      all(len(c["text"].split()) <= 60 for c in sized),
      [len(c["text"].split()) for c in sized])
check("an oversized topic is split rather than dropped",
      sum(c["text"].count("sentence") for c in sized) == 15)

# A page with no sentence punctuation at all (a table dump, an OCR failure) is one
# "sentence" thousands of words long. If the word cap is not passed down into sentence
# splitting, it becomes one chunk the embedding model silently truncates.
runaway = pages_from("T1 " + " ".join(f"word{w}" for w in range(600)))
capped = chunking.semantic_chunks(
    runaway, percentile=95, buffer_size=0, min_words=1, max_words=50, embed=stub_embed)
check("an unpunctuated block is still capped at max_words",
      capped and all(len(c["text"].split()) <= 50 for c in capped),
      [len(c["text"].split()) for c in capped][:5])

# min_words larger than any single topic block: the merger must fold them together, but
# never past max_words.
merged = chunking.semantic_chunks(
    doc, percentile=90, buffer_size=0, min_words=200, max_words=250, embed=stub_embed)
check("undersized chunks are merged up towards min_words",
      len(merged) < 3, len(merged))
check("merging still respects max_words",
      all(len(c["text"].split()) <= 250 for c in merged),
      [len(c["text"].split()) for c in merged])
check("merging loses no text",
      sum(c["text"].count("sentence") for c in merged) == 15)

# An over-long group must be cut at its WEAKEST internal seam, not at an arbitrary one.
# Six sentences, no distance above the p100 threshold, so they start as a single group;
# max_words then forces exactly one cut. The only sensible place is the T1/T2 seam, and
# cutting anywhere else leaves a chunk that mixes both topics - which is the failure this
# whole strategy exists to avoid.
mixed = pages_from("\n\n".join(
    [sentence("T1", n) for n in range(3)] + [sentence("T2", n) for n in range(3)]))
seamed = chunking.semantic_chunks(
    mixed, percentile=100, buffer_size=0, min_words=1, max_words=35, embed=stub_embed)
check("an over-long group is cut at its weakest internal seam", len(seamed) == 2, len(seamed))
check("...so neither half mixes the two topics",
      len(seamed) == 2
      and "T2" not in seamed[0]["text"] and "T1" not in seamed[1]["text"],
      [c["text"][:40] for c in seamed])

# A heading-sized fragment between two topics should attach to the topic it matches. The
# fragment below shares a marker with T2, so it is much closer to the section it
# introduces than to the one that just ended - and it must land on that side.
heading_doc = pages_from(
    "\n\n".join([sentence("T1", 0), sentence("T1", 1), "T2 T4 heading.",
                 sentence("T2", 0), sentence("T2", 1)])
)
attached = chunking.semantic_chunks(
    heading_doc, percentile=60, buffer_size=0, min_words=15, max_words=10_000,
    embed=stub_embed)
holder = next((c for c in attached if "T4 heading." in c["text"]), None)
check("a short fragment merges into its more similar neighbour",
      holder is not None and "T2 sentence 0" in holder["text"]
      and "T1 sentence 0" not in holder["text"],
      holder["text"][:80] if holder else None)


# ---------------------------------------------------------------- pages and edge cases
print("\nsemantic_chunks() - pages and edge cases")

multi = pages_from(sentence("T1", 0), sentence("T1", 1), sentence("T2", 0))
spanning = chunking.semantic_chunks(
    multi, percentile=90, buffer_size=0, min_words=1, max_words=10_000, embed=stub_embed)
check("a chunk spanning two pages reports the full range",
      spanning[0]["page_start"] == 1 and spanning[0]["page_end"] == 2,
      (spanning[0]["page_start"], spanning[0]["page_end"]))
check("page_start <= page_end for every chunk",
      all(c["page_start"] <= c["page_end"] for c in spanning))
check("every chunk carries the keys the vector store writes",
      all({"text", "page_start", "page_end"} <= set(c) for c in spanning))

check("no pages produce no chunks",
      chunking.semantic_chunks([], embed=stub_embed) == [])
one = chunking.semantic_chunks(pages_from("Only one sentence here."), embed=stub_embed)
check("a single sentence produces a single chunk", len(one) == 1, one)

# A provider that returns the wrong number of vectors must fall back, not misalign.
fallback = chunking.semantic_chunks(
    doc, percentile=90, buffer_size=0, min_words=1, max_words=10_000,
    embed=lambda texts: stub_embed(texts)[:-2])
check("a short vector list falls back to fixed chunking instead of misaligning",
      len(fallback) >= 1 and sum(c["text"].count("sentence") for c in fallback) == 15,
      len(fallback))

# Determinism: same input, same output. Resumable ingest depends on this.
first = chunking.semantic_chunks(doc, percentile=90, buffer_size=0, min_words=1,
                                 max_words=60, embed=stub_embed)
second = chunking.semantic_chunks(doc, percentile=90, buffer_size=0, min_words=1,
                                  max_words=60, embed=stub_embed)
check("chunking is deterministic (resumed ingest slices depend on it)", first == second)


# ---------------------------------------------------------------- buffer smoothing
print("\nsemantic_chunks() - neighbour buffer")

windows = chunking._windows([("a.", 1), ("b.", 1), ("c.", 1)], 1)
check("buffer 1 widens each sentence to its neighbours",
      windows == ["a. b.", "a. b. c.", "b. c."], windows)
check("buffer 0 leaves sentences alone",
      chunking._windows([("a.", 1), ("b.", 1)], 0) == ["a.", "b."])
noisy = pages_from("\n\n".join([
    sentence("T1", 0), sentence("T1", 1), "It does not.",
    sentence("T1", 2), sentence("T1", 3),
]))
smoothed = chunking.semantic_chunks(
    noisy, percentile=90, buffer_size=1, min_words=1, max_words=10_000, embed=stub_embed)
check("the buffer stops a contentless sentence being read as a topic change",
      len(smoothed) == 1, len(smoothed))


# ---------------------------------------------------------------- dispatch and cache
print("\nchunk_pages() dispatch")

chunking.clear_cache()
# Against the configured size, not a literal 300/50: this must pass on a machine whose
# .env has already been tuned, otherwise it fails for a reason that is not a bug.
check("CHUNK_STRATEGY=fixed dispatches to the fixed packer",
      chunking.chunk_pages(doc)
      == chunk_document(doc, chunking.CHUNK_SIZE_WORDS, chunking.CHUNK_OVERLAP_WORDS))
check("strategy_name() reports the configured strategy",
      chunking.strategy_name() == "fixed", chunking.strategy_name())

# Flip the module-level constant to exercise the semantic path and its cache. (Reloading
# config would be cleaner but would take every importer with it.)
calls = {"n": 0}


def counting_embed(texts):
    calls["n"] += 1
    return stub_embed(texts)


original_strategy = chunking.CHUNK_STRATEGY
original_semantic = chunking.semantic_chunks
try:
    chunking.CHUNK_STRATEGY = "semantic"
    chunking.semantic_chunks = lambda pages: original_semantic(
        pages, percentile=90, buffer_size=0, min_words=1, max_words=10_000,
        embed=counting_embed)
    chunking.clear_cache()
    a = chunking.chunk_pages(doc)
    b = chunking.chunk_pages(doc)
    check("CHUNK_STRATEGY=semantic dispatches to the semantic chunker", len(a) == 3, len(a))
    check("a repeated call is served from cache (a resumed slice must not re-embed)",
          calls["n"] == 1, calls["n"])
    check("the cached result is identical, not merely similar", a == b)
    other = pages_from("\n\n".join(sentence("T4", n) for n in range(4)))
    chunking.chunk_pages(other)
    check("a different document is not served the cached chunks", calls["n"] == 2, calls["n"])
    chunking.clear_cache()
    chunking.chunk_pages(doc)
    check("clear_cache() forces a re-chunk", calls["n"] == 3, calls["n"])
finally:
    chunking.CHUNK_STRATEGY = original_strategy
    chunking.semantic_chunks = original_semantic
    chunking.clear_cache()


# ---------------------------------------------------------------- fixed chunker, unchanged
print("\nfixed chunker is untouched")

fixed = chunk_document(doc, 60, 10)
check("still packs to roughly the requested size",
      all(len(c["text"].split()) <= 75 for c in fixed),
      [len(c["text"].split()) for c in fixed])
check("still overlaps between neighbours",
      len(fixed) > 1 and fixed[0]["text"].split()[-1] in fixed[1]["text"])
check("still reports page ranges",
      all(c["page_start"] <= c["page_end"] for c in fixed))


print("\n" + "=" * 60)
print(f"{PASSED} passed, {FAILED} failed")
print("=" * 60)
sys.exit(1 if FAILED else 0)
