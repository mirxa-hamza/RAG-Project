"""
Offline end-to-end test.

This sandbox can't reach huggingface.co to download the real sentence-transformers
model, so this test substitutes a small deterministic "fake" embedding model
(hashed bag-of-words, cosine-comparable) in its place via sys.modules, BEFORE
importing any of the real project modules. Every other line of real project code
(pdf_utils, embeddings, vectorstore, llm, main) runs completely unmodified.

This proves: PDF extraction, chunking, ChromaDB storage/retrieval, and every
FastAPI endpoint work correctly. It does NOT prove the real HuggingFace model
downloads correctly (needs open internet - test that on your own machine) or
that the Groq call succeeds (needs a real API key).
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

# ---- 2. Now import the REAL project code - it'll pick up the fake backend above ----
# NOTE: config.py reads these via os.getenv() AT IMPORT TIME, and main.py/vectorstore.py
# do `from config import X`, binding the value then - so env vars must be set BEFORE
# `import config`, not after (setting config.X afterwards would be too late).
import os
os.environ["CHROMA_DIR"] = "/tmp/test_chroma_db"      # isolated so it doesn't touch a real run
os.environ["GROQ_API_KEY"] = ""                        # force the "no key" code path in llm.py
os.environ["CHUNK_SIZE_WORDS"] = "50"                  # our sample PDF is only ~90 words/page,
os.environ["CHUNK_OVERLAP_WORDS"] = "10"               # so use small chunks to get >1 chunk/page

import shutil
shutil.rmtree("/tmp/test_chroma_db", ignore_errors=True)

import config

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    assert condition, label


# ---- 3. Run through the real HTTP endpoints exactly as the frontend would ----
print("\n--- /health ---")
r = client.get("/health")
check("health returns 200", r.status_code == 200)

print("\n--- /upload ---")
with open("../data/sample.pdf", "rb") as f:
    r = client.post("/upload", files={"file": ("sample.pdf", f, "application/pdf")})
print(r.json())
check("upload returns 200", r.status_code == 200)
check("upload extracted 3 pages", r.json()["pages"] == 3)
check("upload stored chunks", r.json()["chunks_stored"] > 0)

print("\n--- /stats ---")
r = client.get("/stats")
print(r.json())
check("stats shows sample.pdf", "sample.pdf" in r.json()["sources"])

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
r = client.post("/chat", json={"question": "What stopover sites did the cranes use?"})
data = r.json()
print("Sources:", [(s["page"], s["similarity"]) for s in data["sources"]])
all_snippets = " ".join(s["snippet"] for s in data["sources"])
check("retrieved evidence mentions the Yellow River Delta finding",
      "Yellow River Delta" in all_snippets)

print("\n--- /reset ---")
r = client.post("/reset")
check("reset returns 200", r.status_code == 200)
r = client.get("/stats")
check("store is empty after reset", r.json()["total_chunks"] == 0)

print("\nAll checks passed.")
