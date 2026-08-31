"""
Offline end-to-end test.

This sandbox can't reach huggingface.co to download the real sentence-transformers
model, so this test substitutes a small deterministic "fake" embedding model
(hashed bag-of-words, cosine-comparable) in its place via sys.modules, BEFORE
importing any of the real project modules. Every other line of real project code
(pdf_utils, embeddings, vectorstore, ingest, llm, main) runs completely unmodified.

This proves: PDF extraction, chunking, ChromaDB storage/retrieval, startup ingestion,
POST /ingest, and every other FastAPI endpoint work correctly. It does NOT prove the
real HuggingFace model downloads correctly (needs open internet - test that on your
own machine) or that the Groq call succeeds (needs a real API key).
"""
import sys
import hashlib
import numpy as np
import types

# ---- 1. Install a fake sentence_transformers module before anything imports it ----
DIM = 128

def _fake_vector(text: str):
    vec = np.zeros(DIM, dtype=np.float32)
    for word in text.lower().split():
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

class FakeSentenceTransformer:
    def __init__(self, model_name):
        print(f"[fake embeddings] pretending to load '{model_name}'")

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
        return np.array([_fake_vector(t) for t in texts])

fake_module = types.ModuleType("sentence_transformers")
fake_module.SentenceTransformer = FakeSentenceTransformer
sys.modules["sentence_transformers"] = fake_module

# ---- 2. Point DATA_DIR at an isolated test folder, never the real data/ folder ----
# NOTE: config.py reads these via os.getenv() AT IMPORT TIME, and main.py/vectorstore.py/
# ingest.py do `from config import X`, binding the value then - so env vars must be set
# BEFORE `import config`, not after (setting config.X afterwards would be too late).
import os
import shutil

TEST_DATA_DIR = "/tmp/test_data_dir"
shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)

os.environ["CHROMA_DIR"] = "/tmp/test_chroma_db"      # isolated so it doesn't touch a real run
os.environ["DATA_DIR"] = TEST_DATA_DIR                # isolated - never the real data/ folder
os.environ["GROQ_API_KEY"] = ""                        # force the "no key" code path in llm.py
os.environ["CHUNK_SIZE_WORDS"] = "50"                  # our sample PDF is only ~90 words/page,
os.environ["CHUNK_OVERLAP_WORDS"] = "10"               # so use small chunks to get >1 chunk/page

shutil.rmtree("/tmp/test_chroma_db", ignore_errors=True)

from make_test_pdf import make_pdf
make_pdf(os.path.join(TEST_DATA_DIR, "sample.pdf"))

import config

from fastapi.testclient import TestClient
from main import app

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    assert condition, label


# ---- 3. Run through the real HTTP endpoints exactly as the frontend would ----
# `with TestClient(app)` runs FastAPI's startup event, which is what triggers the first
# ingestion pass over TEST_DATA_DIR - the "server restart picks up new PDFs" behavior,
# exercised for real, not simulated.
with TestClient(app) as client:
    print("\n--- /health ---")
    r = client.get("/health")
    check("health returns 200", r.status_code == 200)

    print("\n--- startup ingestion (checked via /stats) ---")
    r = client.get("/stats")
    print(r.json())
    check("startup ingested sample.pdf", "sample.pdf" in r.json()["sources"])
    check("startup stored chunks", r.json()["total_chunks"] > 0)

    print("\n--- /ingest re-run (nothing new, should not re-embed) ---")
    r = client.post("/ingest")
    print(r.json())
    check("ingest returns 200", r.status_code == 200)
    check("re-running ingest finds nothing new",
          all(res["status"] == "already_stored" for res in r.json()["results"]))

    print("\n--- /ingest picks up a newly-added file without a restart ---")
    make_pdf(os.path.join(TEST_DATA_DIR, "sample2.pdf"))
    r = client.post("/ingest")
    print(r.json())
    check("second file got ingested", any(
        res["filename"] == "sample2.pdf" and res["status"] == "ingested"
        for res in r.json()["results"]
    ))

    print("\n--- /chat (retrieval quality check) ---")
    r = client.post("/chat", json={"question": "How much funding did the project receive?"})
    data = r.json()
    print("Answer:", data["answer"][:200])
    print("Sources:", data["sources"])
    check("chat returns 200", r.status_code == 200)
    check("chat returns sources", len(data["sources"]) > 0)
    # Content-based check (robust to the fake test embedding and to page-boundary chunk labeling):
    # the actual funding figure should be in the retrieved evidence somewhere in the top results.
    all_snippets = " ".join(s["snippet"] for s in data["sources"])
    check("retrieved evidence actually contains the funding figure",
          "1.2 million" in all_snippets)
    check("no GROQ_API_KEY path returns the setup message instead of crashing",
          "GROQ_API_KEY" in data["answer"])

    print("\n--- /chat (different topic) ---")
    # top_k raised: sample.pdf and sample2.pdf are duplicate content, so the default
    # top_k=4 can fill up with near-identical top matches from both files and miss this
    # chunk - ask for more to make sure it's still findable in the store.
    r = client.post("/chat", json={"question": "What stopover sites did the cranes use?", "top_k": 8})
    data = r.json()
    print("Sources:", [(s["page"], s["similarity"]) for s in data["sources"]])
    all_snippets = " ".join(s["snippet"] for s in data["sources"])
    check("retrieved evidence mentions the Yellow River Delta finding",
          "Yellow River Delta" in all_snippets)

    print("\n--- /reset (wipes then re-ingests from the data folder) ---")
    r = client.post("/reset")
    print(r.json())
    check("reset returns 200", r.status_code == 200)
    check("reset re-ingested both fixture files",
          all(res["status"] == "ingested" for res in r.json()["results"]))
    r = client.get("/stats")
    check("store is non-empty again after reset (re-ingested from data folder)",
          r.json()["total_chunks"] > 0)

print("\nAll checks passed.")
