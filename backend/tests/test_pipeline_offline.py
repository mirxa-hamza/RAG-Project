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
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

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


fake_module = types.ModuleType("sentence_transformers")
fake_module.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = fake_module

# ---- 2. Point every path at isolated temp folders, never the real data/ or chroma_db --
# NOTE: config.py reads these via os.getenv() AT IMPORT TIME and other modules do
# `from app.config import X`, binding the value then - so env vars must be set BEFORE
# `import app.config`, not after.
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

from app import manifest  # noqa: E402
from app.pdf_utils import chunk_document, format_pages  # noqa: E402
from app.main import app  # noqa: E402

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
    check("similarity is never negative", all(s["similarity"] >= 0 for s in data["sources"]))
    all_snippets = " ".join(s["snippet"] for s in data["sources"])
    check("retrieved evidence actually contains the funding figure", "1.2 million" in all_snippets)
    check("no GROQ_API_KEY path returns the setup message instead of crashing",
          "GROQ_API_KEY" in data["answer"])

    print("\n--- /chat (different topic) ---")
    r = client.post("/chat", json={"question": "What stopover sites did the cranes use?", "top_k": 8})
    all_snippets = " ".join(s["snippet"] for s in r.json()["sources"])
    check("retrieved evidence mentions the Yellow River Delta finding",
          "Yellow River Delta" in all_snippets)

    print("\n--- input validation ---")
    check("empty question rejected", client.post("/chat", json={"question": "   "}).status_code == 422)
    check("absurd top_k rejected",
          client.post("/chat", json={"question": "hi", "top_k": 10000}).status_code == 422)
    check("negative top_k rejected",
          client.post("/chat", json={"question": "hi", "top_k": 0}).status_code == 422)

    print("\n--- /reset (wipes, then re-ingests in the background) ---")
    r = client.post("/reset")
    check("reset returns 202", r.status_code == 202)
    job = wait_for_ingest(client)
    check("reset re-ingested every file",
          all(res["status"] == "ingested" for res in job["results"]))
    stats = client.get("/stats").json()
    check("store is populated again after reset", stats["total_chunks"] > 0)
    check("manifest rebuilt after reset", len(stats["sources"]) == 2)

print(f"\nAll {PASSED} checks passed.")
