"""
Offline end-to-end test.

Run from the backend/ folder:   python tests/test_pipeline_offline.py

CI sandboxes can't reach huggingface.co to download the real sentence-transformers model,
so this test substitutes a small deterministic "fake" embedding model (hashed
bag-of-words, cosine-comparable) via sys.modules BEFORE importing any project module.
Every other line of real project code runs completely unmodified.

This proves: PDF extraction, structure-aware chunking, page-range tagging, change
detection, ChromaDB storage/retrieval, the background ingestion job, and every FastAPI
endpoint. It does NOT prove that the real embedding model downloads (needs open internet)
or that a live Groq call succeeds (needs a real API key).
"""
import hashlib
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---- 1. Install a fake sentence_transformers module before anything imports it --------
DIM = 128


def _fake_vector(text: str):
    vec = np.zeros(DIM, dtype=np.float32)
    for word in text.lower().split():
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class FakeSentenceTransformer:
    max_seq_length = 512

    def __init__(self, model_name):
        print(f"[fake embeddings] pretending to load '{model_name}'")

    def encode(self, texts, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, normalize_embeddings=False):
        return np.array([_fake_vector(t) for t in texts])


_STOPWORDS = {
    "a", "an", "the", "и", "of", "to", "in", "on", "at", "for", "with", "and", "or", "is",
    "are", "was", "were", "be", "do", "does", "did", "how", "what", "which", "who", "when",
    "where", "why", "i", "you", "it", "this", "that", "these", "those", "can", "could",
    "would", "should", "much", "many", "receive", "use", "used",
}


class FakeCrossEncoder:
    """
    Stands in for the re-ranker. Scores on content-word overlap so ordering is
    deterministic and inspectable; the real model reads the pair semantically.

    Stopwords are excluded deliberately: a real cross-encoder gives no credit for a shared
    "a" or "how", and a stub that did would let any off-topic question clear the relevance
    floor - masking exactly the behaviour these tests exist to check.
    """

    def __init__(self, model_name):
        print(f"[fake reranker] pretending to load '{model_name}'")

    @staticmethod
    def _content(text):
        return {w.strip(".,;:()[]?!\"'") for w in text.lower().split()} - _STOPWORDS

    def predict(self, pairs):
        scores = []
        for question, text in pairs:
            overlap = len(self._content(question) & self._content(text))
            # Map overlap onto a logit-ish range: no overlap lands below the default floor.
            scores.append(-8.0 + 3.0 * overlap)
        return np.array(scores, dtype=np.float32)


fake_module = types.ModuleType("sentence_transformers")
fake_module.SentenceTransformer = FakeSentenceTransformer
fake_module.CrossEncoder = FakeCrossEncoder
sys.modules["sentence_transformers"] = fake_module

# ---- 2. Point every path at isolated temp folders, never the real data/ or chroma_db --
# NOTE: config.py reads these via os.getenv() AT IMPORT TIME and other modules do
# `from src.core.config import X`, binding the value then - so env vars must be set BEFORE
# `import src.core.config`, not after.
TEST_DATA_DIR = Path("/tmp/rag_test_data") if os.name != "nt" else Path(os.environ["TEMP"]) / "rag_test_data"
TEST_CHROMA_DIR = Path("/tmp/rag_test_chroma") if os.name != "nt" else Path(os.environ["TEMP"]) / "rag_test_chroma"
shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ["DATA_DIR"] = str(TEST_DATA_DIR)
os.environ["CHROMA_DIR"] = str(TEST_CHROMA_DIR)
os.environ["GROQ_API_KEY"] = ""            # force the "no key" code path in llm.py
os.environ["CHUNK_SIZE_WORDS"] = "50"      # sample PDF is ~90 words/page, so small chunks
os.environ["CHUNK_OVERLAP_WORDS"] = "10"   # to get more than one chunk per page
os.environ["EMBEDDING_QUERY_PREFIX"] = ""
os.environ["LOG_LEVEL"] = "WARNING"        # keep the test output readable

from scripts.make_test_pdf import make_pdf  # noqa: E402

make_pdf(str(TEST_DATA_DIR / "sample.pdf"))

from fastapi.testclient import TestClient  # noqa: E402

from src.ml import reranker  # noqa: E402
from src.services import bm25, manifest, retrieval, vectorstore  # noqa: E402
from src.services.pdf import chunk_document, format_pages  # noqa: E402
from src.main import app  # noqa: E402

PASSED = 0


def check(label, condition):
    global PASSED
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    assert condition, label
    PASSED += 1


def wait_for_ingest(client, timeout=120):
    """The ingest job is a background thread now - poll until it finishes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get("/ingest/status").json()
        if body["state"] != "running":
            return body
        time.sleep(0.05)
    raise TimeoutError("ingestion job did not finish in time")


# ---- 3. Unit checks on the chunker, independent of any PDF ----------------------------
print("\n--- chunker (structure awareness) ---")

pages = [
    {"page": 1, "text": "Alpha heading.\n\n" + " ".join(["alpha"] * 40)},
    {"page": 2, "text": " ".join(["beta"] * 40) + "\n\nGamma tail paragraph."},
]
chunks = chunk_document(pages, chunk_size_words=50, overlap_words=10)
check("chunker produced chunks", len(chunks) > 0)
check("every chunk carries a page range",
      all("page_start" in c and "page_end" in c for c in chunks))
check("page_end is never before page_start",
      all(c["page_end"] >= c["page_start"] for c in chunks))
check("no chunk exceeds the size budget by more than one paragraph",
      all(len(c["text"].split()) <= 50 + 40 for c in chunks))
check("a page's paragraphs are not split across chunks when they fit",
      all(c["page_start"] == c["page_end"] for c in chunks))

# Small paragraphs on consecutive pages DO pack into one chunk - that chunk must record
# the whole range it covers, which is the bug page_start/page_end exists to fix.
short_pages = [
    {"page": 7, "text": " ".join(["seven"] * 20)},
    {"page": 8, "text": " ".join(["eight"] * 20)},
]
spanning = chunk_document(short_pages, chunk_size_words=50, overlap_words=10)
check("a chunk spanning two pages records both",
      any(c["page_start"] == 7 and c["page_end"] == 8 for c in spanning))
check("format_pages renders a single page", format_pages(3, 3) == "page 3")
check("format_pages renders a range", format_pages(3, 4) == "pages 3-4")

huge = [{"page": 1, "text": " ".join(["word"] * 500)}]
check("an oversized single paragraph is still split",
      len(chunk_document(huge, chunk_size_words=50, overlap_words=10)) > 1)
check("empty input yields no chunks", chunk_document([], 50, 10) == [])
check("overlap >= chunk size does not explode",
      len(chunk_document(huge, chunk_size_words=50, overlap_words=999)) > 1)

# ---- 4. Drive the real HTTP endpoints exactly as the frontend would -------------------
# `with TestClient(app)` runs the lifespan startup, which kicks off the background
# ingestion job - the "server boots and picks up the data folder" path, for real.
with TestClient(app) as client:
    print("\n--- /health (must answer while ingestion runs in the background) ---")
    r = client.get("/health")
    check("health returns 200 immediately", r.status_code == 200)

    print("\n--- startup ingestion ---")
    job = wait_for_ingest(client)
    check("startup job finished without error", job["state"] == "idle")
    r = client.get("/stats")
    print(r.json())
    check("startup ingested sample.pdf", "sample.pdf" in r.json()["sources"])
    check("startup stored chunks", r.json()["total_chunks"] > 0)
    check("stats reports per-document detail",
          r.json()["documents"][0]["pages"] == 3 and r.json()["documents"][0]["chunks"] > 0)
    chunks_after_first = r.json()["total_chunks"]

    print("\n--- /ingest re-run (nothing new, must not re-embed) ---")
    r = client.post("/ingest")
    check("ingest returns 202", r.status_code == 202)
    job = wait_for_ingest(client)
    check("re-running ingest finds nothing new",
          all(res["status"] == "already_stored" for res in job["results"]))
    check("chunk count unchanged", client.get("/stats").json()["total_chunks"] == chunks_after_first)

    print("\n--- /ingest picks up a newly-added file without a restart ---")
    make_pdf(str(TEST_DATA_DIR / "sample2.pdf"))
    client.post("/ingest")
    job = wait_for_ingest(client)
    check("second file got ingested", any(
        res["filename"] == "sample2.pdf" and res["status"] == "ingested"
        for res in job["results"]
    ))

    print("\n--- change detection (edited PDF is re-ingested, not duplicated) ---")
    before = client.get("/stats").json()["total_chunks"]
    sha_before = manifest.get("sample2.pdf")["sha256"]
    # Overwrite sample2 with different content so its hash changes.
    (TEST_DATA_DIR / "sample2.pdf").write_bytes((TEST_DATA_DIR / "sample.pdf").read_bytes())
    padding = b"\n%% cache-busting comment to change the file hash\n"
    with open(TEST_DATA_DIR / "sample2.pdf", "ab") as fh:
        fh.write(padding)
    client.post("/ingest")
    job = wait_for_ingest(client)
    check("changed file was re-ingested", any(
        res["filename"] == "sample2.pdf" and res["status"] == "ingested"
        and res.get("reason") == "changed"
        for res in job["results"]
    ))
    check("manifest hash was updated", manifest.get("sample2.pdf")["sha256"] != sha_before)
    check("re-ingest replaced chunks instead of duplicating them",
          client.get("/stats").json()["total_chunks"] == before)

    print("\n--- /chat (retrieval quality check) ---")
    r = client.post("/chat", json={"question": "How much funding did the project receive?"})
    data = r.json()
    print("Answer:", data["answer"][:120])
    print("Sources:", [(s["source"], s["pages"], s["similarity"]) for s in data["sources"]])
    check("chat returns 200", r.status_code == 200)
    check("chat returns sources", len(data["sources"]) > 0)
    check("sources carry a page label", all(s["pages"].startswith("page") for s in data["sources"]))
    # Neighbour chunks are fetched by id, not by search, so they carry no similarity.
    check("similarity is never negative",
          all(s["similarity"] >= 0 for s in data["sources"] if s["similarity"] is not None))
    check("at least one source carries a similarity score",
          any(s["similarity"] is not None for s in data["sources"]))
    all_snippets = " ".join(s["snippet"] for s in data["sources"])
    check("retrieved evidence actually contains the funding figure", "1.2 million" in all_snippets)
    check("no GROQ_API_KEY path returns the setup message instead of crashing",
          "GROQ_API_KEY" in data["answer"])

    print("\n--- /chat (different topic) ---")
    r = client.post("/chat", json={"question": "What stopover sites did the cranes use?", "top_k": 8})
    all_snippets = " ".join(s["snippet"] for s in r.json()["sources"])
    check("retrieved evidence mentions the Yellow River Delta finding",
          "Yellow River Delta" in all_snippets)

    print("\n--- hybrid retrieval (BM25 + vector, fused) ---")
    lexical = bm25.search("Poyang", limit=10)
    check("BM25 finds an exact rare term", len(lexical) > 0)
    check("BM25 hit actually contains the term",
          any("poyang" in chunk["text"].lower() for chunk, _ in lexical))
    check("BM25 keeps technical tokens intact", bm25.tokenize("A* search k-means f1") ==
          ["a*", "search", "k-means", "f1"])
    check("BM25 scoping to one source filters the rest",
          all(c["source"] == "sample.pdf"
              for c, _ in bm25.search("Poyang", limit=10, source="sample.pdf")))

    fused = retrieval.fuse([
        [{"source": "a", "chunk_index": 1}, {"source": "a", "chunk_index": 2}],
        [{"source": "a", "chunk_index": 2}, {"source": "a", "chunk_index": 3}],
    ])
    check("RRF ranks the chunk both lists agree on first", fused[0]["chunk_index"] == 2)
    check("RRF deduplicates across lists", len(fused) == 3)

    print("\n--- cross-encoder re-ranking ---")
    check("re-ranker is available (stubbed in this test)", reranker.available())
    hits = retrieval.retrieve("How much funding did the project receive?", top_k=3)
    check("re-ranked hits carry a score", all("rerank_score" in c or "neighbor_of" in c for c in hits))
    ranked_scores = [c["rerank_score"] for c in hits if "rerank_score" in c]
    check("re-rank scores are in descending order",
          ranked_scores == sorted(ranked_scores, reverse=True))

    print("\n--- relevance floor (unanswerable question is not answered) ---")
    off_topic = retrieval.retrieve("How do I configure a Kubernetes ingress controller?", top_k=4)
    check("nothing clears the floor for an off-topic question", off_topic == [])
    r = client.post("/chat", json={"question": "How do I configure a Kubernetes ingress controller?"})
    check("off-topic question returns no sources", r.json()["sources"] == [])
    check("off-topic question declines instead of guessing",
          "couldn't find" in r.json()["answer"].lower())

    print("\n--- neighbour expansion ---")
    expanded = retrieval.retrieve("How much funding did the project receive?", top_k=1, expand=1)
    unexpanded = retrieval.retrieve("How much funding did the project receive?", top_k=1, expand=0)
    check("expansion adds neighbouring chunks", len(expanded) > len(unexpanded))
    check("neighbours are flagged as such", any("neighbor_of" in c for c in expanded))
    check("neighbours come from the same document",
          len({c["source"] for c in expanded}) == 1)

    print("\n--- per-document scoping ---")
    r = client.post("/chat", json={"question": "How much funding did the project receive?",
                                   "source": "sample2.pdf"})
    check("scoped chat only cites the requested document",
          all(s["source"] == "sample2.pdf" for s in r.json()["sources"]))
    check("scoped chat still finds the answer",
          "1.2 million" in " ".join(s["snippet"] for s in r.json()["sources"]))

    print("\n--- conversation history + follow-up rewriting ---")
    r = client.post("/chat", json={
        "question": "And how long do they rest there?",
        "history": [{"question": "What stopover sites did the cranes use?",
                     "answer": "The Yellow River Delta and Poyang Lake."}],
    })
    check("chat accepts history", r.status_code == 200)
    # Rewriting needs a Groq key; without one the raw question is used unchanged.
    check("no rewrite is reported when the LLM is unconfigured",
          r.json()["search_query"] is None)

    print("\n--- /chat/stream (SSE) ---")
    with client.stream("POST", "/chat/stream",
                       json={"question": "How much funding did the project receive?"}) as stream:
        check("stream returns 200", stream.status_code == 200)
        body = "".join(stream.iter_text())
    check("stream emits a sources event", "event: sources" in body)
    check("stream emits token events", "event: token" in body)
    check("stream terminates with done", body.rstrip().endswith('event: done\ndata: {}'))
    payload = json.loads(body.split("event: sources\ndata: ")[1].split("\n\n")[0])
    check("streamed sources match the non-streamed shape",
          len(payload["sources"]) > 0 and "pages" in payload["sources"][0])

    print("\n--- input validation ---")
    check("empty question rejected", client.post("/chat", json={"question": "   "}).status_code == 422)
    check("absurd top_k rejected",
          client.post("/chat", json={"question": "hi", "top_k": 10000}).status_code == 422)
    check("negative top_k rejected",
          client.post("/chat", json={"question": "hi", "top_k": 0}).status_code == 422)

    print("\n--- edge cases: ingestion robustness ---")
    # A corrupt PDF must not take the rest of the corpus down with it. Named to sort
    # BETWEEN the two good files, so a job that aborts on it leaves the third un-indexed.
    (TEST_DATA_DIR / "mmm_corrupt.pdf").write_bytes(b"%PDF-1.4\nnot actually a pdf\n")
    make_pdf(str(TEST_DATA_DIR / "zzz_after_corrupt.pdf"))
    client.post("/ingest")
    job = wait_for_ingest(client)
    by_name = {r["filename"]: r for r in job["results"]}
    check("a corrupt PDF is reported as failed, not raised",
          by_name["mmm_corrupt.pdf"]["status"] == "failed")
    check("the ingest job survives a corrupt PDF", job["state"] == "idle")
    check("files after the corrupt one still get ingested",
          by_name["zzz_after_corrupt.pdf"]["status"] == "ingested")

    # Nested folders under data/ used to be invisible.
    (TEST_DATA_DIR / "textbooks").mkdir(exist_ok=True)
    make_pdf(str(TEST_DATA_DIR / "textbooks" / "nested.pdf"))
    client.post("/ingest")
    job = wait_for_ingest(client)
    check("a PDF in a subfolder is found",
          any(r["filename"] == "textbooks/nested.pdf" and r["status"] == "ingested"
              for r in job["results"]))
    check("nested documents are keyed by their relative path",
          "textbooks/nested.pdf" in client.get("/stats").json()["sources"])

    # Byte-identical duplicates would otherwise compete with themselves for every slot.
    shutil.copy(TEST_DATA_DIR / "sample.pdf", TEST_DATA_DIR / "sample_copy.pdf")
    client.post("/ingest")
    job = wait_for_ingest(client)
    duplicate = next(r for r in job["results"] if r["filename"] == "sample_copy.pdf")
    check("a byte-identical duplicate is skipped", duplicate["status"] == "skipped")
    check("the duplicate names the file it duplicates", "duplicate of" in duplicate["reason"])

    print("\n--- edge cases: deleted documents are pruned ---")
    before_sources = set(client.get("/stats").json()["sources"])
    (TEST_DATA_DIR / "zzz_after_corrupt.pdf").unlink()
    client.post("/ingest")
    job = wait_for_ingest(client)
    check("a deleted file is reported as removed",
          any(r["filename"] == "zzz_after_corrupt.pdf" and r["status"] == "removed"
              for r in job["results"]))
    after = client.get("/stats").json()
    check("the deleted document leaves the manifest",
          "zzz_after_corrupt.pdf" in before_sources and "zzz_after_corrupt.pdf" not in after["sources"])
    check("the deleted document is no longer retrievable",
          all(c["source"] != "zzz_after_corrupt.pdf"
              for c in retrieval.retrieve("funding project budget", top_k=20, use_rerank=False)))

    print("\n--- edge cases: manifest durability ---")
    from src.core.config import MANIFEST_PATH  # noqa: E402
    intact = MANIFEST_PATH.read_text(encoding="utf-8")
    MANIFEST_PATH.write_text(intact[: len(intact) // 2], encoding="utf-8")
    check("a truncated manifest degrades to empty instead of raising", manifest.sources() == [])
    MANIFEST_PATH.write_text(intact, encoding="utf-8")
    check("the manifest reads back correctly once restored", len(manifest.sources()) > 0)
    manifest.put("probe.pdf", sha256="abc", mtime=1.0, size=2, pages=3, chunks=4)
    check("writes are atomic (no partial file left behind)",
          not any(p.name.startswith(".manifest-") for p in MANIFEST_PATH.parent.iterdir()))
    check("a written entry reads back", manifest.get("probe.pdf")["chunks"] == 4)
    manifest.remove("probe.pdf")
    check("a removed entry is gone", manifest.get("probe.pdf") is None)

    print("\n--- edge cases: context budget and query prefix ---")
    from src.ml.llm import build_context  # noqa: E402
    oversized = [
        {"source": "big.pdf", "page_start": 1, "page_end": 1, "text": "Z" * 5000},
        {"source": "small.pdf", "page_start": 2, "page_end": 2, "text": "relevant detail"},
    ]
    context = build_context(oversized, max_chars=1000)
    check("a chunk larger than the whole budget is truncated, not dropped", context != "")
    check("the truncated context respects the budget", len(context) <= 1000)
    check("the truncated context keeps its citation label", "big.pdf" in context)

    import src.core.config as app_config  # noqa: E402
    check("the query prefix always ends in a separator (dotenv strips trailing spaces)",
          app_config.EMBEDDING_QUERY_PREFIX == "" or app_config.EMBEDDING_QUERY_PREFIX.endswith(" "))

    print("\n--- edge cases: retrieval guards ---")
    call_count = {"n": 0}
    original_get = vectorstore._collection.get

    def counting_get(*args, **kwargs):
        call_count["n"] += 1
        return original_get(*args, **kwargs)

    vectorstore._collection.get = counting_get
    try:
        hits = retrieval.retrieve("funding budget project", top_k=6, use_rerank=False, expand=1)
    finally:
        vectorstore._collection.get = original_get
    check("neighbour expansion batches into one query per source", call_count["n"] <= 3)
    check("expansion still returns neighbours", any("neighbor_of" in c for c in hits))

    # A top_k above RETRIEVAL_CANDIDATES must widen the candidate pool, not silently
    # return fewer hits than asked for. The relevance floor is neutralised here so the
    # only thing under test is the pool size.
    original_candidates = retrieval.RETRIEVAL_CANDIDATES
    original_floor = retrieval.MIN_SIMILARITY
    retrieval.RETRIEVAL_CANDIDATES = 2
    retrieval.MIN_SIMILARITY = -1.0
    try:
        wide = retrieval.retrieve("project", top_k=10, use_rerank=False, expand=0)
    finally:
        retrieval.RETRIEVAL_CANDIDATES = original_candidates
        retrieval.MIN_SIMILARITY = original_floor
    check("top_k larger than RETRIEVAL_CANDIDATES still fills up",
          len(wide) == min(10, vectorstore.count()))

    print("\n--- /reset (wipes, then re-ingests in the background) ---")
    r = client.post("/reset")
    check("reset returns 202", r.status_code == 202)
    job = wait_for_ingest(client)
    # The folder now deliberately contains a corrupt PDF and a duplicate, so "every file
    # ingested" is the wrong bar: every *usable, unique* file must be, and the bad ones
    # must be classified rather than silently dropped or fatal.
    statuses = {res["filename"]: res["status"] for res in job["results"]}
    check("reset re-ingested every usable file",
          statuses.get("sample.pdf") == "ingested"
          and statuses.get("textbooks/nested.pdf") == "ingested")
    check("reset still classifies the corrupt file as failed",
          statuses.get("mmm_corrupt.pdf") == "failed")
    check("reset still skips the duplicate", statuses.get("sample_copy.pdf") == "skipped")
    stats = client.get("/stats").json()
    check("store is populated again after reset", stats["total_chunks"] > 0)
    check("manifest rebuilt after reset", len(stats["sources"]) >= 2)

    print("\n--- an interrupted ingest does not leave duplicate chunks ---")
    # Simulate a job killed mid-file: chunks land in Chroma, but the manifest entry (which
    # ingest_one writes only after the whole file finishes) never does. The next run sees
    # the file as "new", so it must clear the orphans before re-embedding.
    from src.services import ingestion as ingestion_mod

    before = vectorstore.count()
    partial = [{"text": "orphaned partial chunk about robotics", "page_start": 1, "page_end": 1}]
    vectorstore.add_chunks("sample.pdf", partial)
    check("partial chunks are in the store before the retry",
          vectorstore.count() == before + len(partial))

    manifest.remove("sample.pdf")   # what an interrupted run leaves behind
    result = ingestion_mod.ingest_one("sample.pdf", reason="new")
    check("re-ingest after an interruption reports 'ingested'", result["status"] == "ingested")

    stored = [c for c in vectorstore.all_chunks() if c["source"] == "sample.pdf"]
    check("no orphan chunks survive the retry", len(stored) == result["chunks_stored"])
    check("the orphan text itself is gone",
          all("orphaned partial chunk" not in c["text"] for c in stored))

print(f"\nAll {PASSED} checks passed.")
