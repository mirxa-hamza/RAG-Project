# Document Q&A — a from-scratch RAG system

PDF in → chunked & embedded locally → stored in ChromaDB → retrieved at question time →
answered by an LLM on Groq, grounded only in what's in the PDF.

No LangChain/LlamaIndex — every step is plain Python so you can see exactly what's happening
at each stage. That's deliberate, given this is a learning project.

**Documents come only from the backend's `data/` folder.** There is no upload endpoint —
the frontend can only ask questions, never add or change what's in the vector store. Drop
PDFs into `data/`, restart the server (or hit `POST /ingest`), and they're searchable.

## Stack

| Stage             | Tool                                  |
|-------------------|----------------------------------------|
| PDF text extraction | `PyMuPDF` (`pymupdf`)                 |
| Chunking           | custom sliding-window chunker (`pdf_utils.py`) |
| Embeddings         | `sentence-transformers` (`all-MiniLM-L6-v2`, runs locally, free) |
| Vector store       | ChromaDB (persistent, local folder, no server to run) |
| LLM                | Groq API (`openai/gpt-oss-20b` by default) |
| Backend            | FastAPI + Uvicorn |
| Frontend           | Plain HTML/CSS/JS (no build step) |

> **License note:** `PyMuPDF` is licensed AGPL-3.0 (unlike the permissively-licensed
> libraries above). Fine for personal/learning use; if you ever open-source or sell this,
> either comply with AGPL (source stays open) or buy Artifex's commercial PyMuPDF license.

## Project layout

```
rag-project/
├── backend/
│   ├── main.py                    FastAPI app: /ingest, /chat, /stats, /reset, /health
│   ├── config.py                  loads all settings from .env
│   ├── ingest.py                  the ONLY path documents enter the system - scans data/
│   ├── pdf_utils.py                PDF → pages → overlapping chunks
│   ├── embeddings.py               wraps the sentence-transformers model
│   ├── vectorstore.py              ChromaDB: add_chunks / query_chunks
│   ├── llm.py                      builds the prompt, calls Groq
│   ├── requirements.txt
│   ├── requirements-dev.txt        adds test-only deps (httpx, reportlab)
│   ├── .env.example                copy to .env and fill in your Groq key
│   ├── make_test_pdf.py            generates a fictional PDF into test_fixtures/ for testing
│   ├── test_fixtures/              test-only PDFs (gitignored, never mixed with data/)
│   └── test_pipeline_offline.py    automated end-to-end test (see below)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── data/
    └── (your real PDFs go here — this folder is the only way to add documents)
```

## How a question actually gets answered (the RAG loop)

**At ingestion time (server startup, and whenever `POST /ingest` runs):**
1. The backend scans `data/` for `.pdf` files not already stored.
2. `PyMuPDF` extracts text page by page from each new one.
3. The text is flattened into a stream of words and sliced into overlapping
   chunks (default: 300 words per chunk, 50-word overlap) — overlap exists so a
   sentence that matters doesn't get cut in half between two chunks and lost.
4. Each chunk is embedded (turned into a vector of numbers that captures its
   meaning) by a local model — no API call, no cost.
5. The chunk text + its vector + its page number get stored in ChromaDB.

**At question time (every chat message):**
1. Your question is embedded with the *same* model.
2. ChromaDB returns the chunks whose vectors are closest to the question's vector
   (cosine similarity) — this is the "retrieval" in Retrieval-Augmented Generation.
3. Those chunks get stuffed into a prompt as CONTEXT, along with a system prompt
   that instructs the model to answer only from that context.
4. Groq's LLM generates the answer, and the app returns it along with which
   page(s) it drew from.

## Setup

```bash
cd rag-project/backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set `GROQ_API_KEY` (free key: https://console.groq.com/keys).
The default model is `openai/gpt-oss-20b` — Groq deprecates models periodically,
so if you get a "model not found" error, check https://console.groq.com/docs/models
and update `GROQ_MODEL` in `.env`.

**Windows on ARM64 (Snapdragon/Copilot+ PCs):** `requirements.txt` pins `chromadb==1.5.9`
specifically because older `chromadb` (0.5.x) requires `numpy<2.0`, and numpy has no
official wheel below 2.0 for Windows ARM64 — that combination crashes at import time with
`AttributeError: np.float_ was removed in the NumPy 2.0 release`. If you still see a numpy
`MINGW-W64 ... experimental` warning on startup, that's just numpy's own build note for
this platform (no official MSVC/OpenBLAS build yet) — harmless on its own.

`requirements.txt` also pins `groq==1.7.0` (not the older `groq==0.11.0`) — the old version
breaks under `httpx>=0.28` with `TypeError: Client.__init__() got an unexpected keyword
argument 'proxies'`, since `requirements-dev.txt` pins `httpx==0.28.1` directly.

## Adding documents

Put PDFs directly in the `data/` folder — that's the only way documents get in. Then either:

- **Restart the server** — ingestion runs automatically on startup, or
- **Call `POST /ingest`** (or click "Sync from data folder" in the frontend sidebar) to pick
  up new files without a restart.

Either way, files already ingested are skipped (matched by filename), so it's always safe
to re-run.

## Running it

```bash
# from rag-project/backend, with venv active
uvicorn main:app --reload --port 8000
```

First run will download the embedding model (~90MB) — that needs a normal internet
connection and takes a few seconds to a minute, then it's cached locally for good. It will
also ingest whatever is already in `data/` before it starts serving requests — for a large
PDF this can take a while the first time (embedding runs locally, no API cost, but it's CPU
work), so give it a minute on a big document.

Then open `frontend/index.html` directly in a browser (double-click it, or use
VS Code's Live Server extension). It talks to `http://localhost:8000` by default —
change `API_BASE` at the top of `frontend/script.js` if you serve the backend
elsewhere.

**Try it:**
1. Ask a question about a document's content in the chat box.
2. The answer appears with a "Sources" line showing which page(s) and how
   similar each retrieved chunk was to your question (0–1, higher = closer match).
3. Added a new PDF to `data/` while the server's running? Click "Sync from data folder"
   in the sidebar (or restart the server) to make it searchable.

## Testing

Two layers of testing are included:

**1. Automated pipeline test (`test_pipeline_offline.py`)**
Generates fictional test PDFs into `backend/test_fixtures/` (never `data/`) and drives
every FastAPI endpoint (`/ingest`, `/chat`, `/stats`, `/reset`, startup ingestion) through
`TestClient`, checking that PDF parsing, chunking, and ChromaDB storage/retrieval all
behave correctly.

```bash
pip install -r requirements-dev.txt
python3 test_pipeline_offline.py
```

(`test_pipeline_offline.py` generates its own fixture PDFs via `make_test_pdf.py` — no need
to run that separately.) This swaps in a lightweight fake embedding model instead of the
real one, purely so it can run without a live download in restricted environments; on your
machine you can freely rely on it since your internet access is normal. It also doesn't
call Groq (no API key needed) — it just confirms the app correctly reports "no key set"
instead of crashing, so you know that failure path is handled.

**2. Manual smoke test (do this once you have a real GROQ_API_KEY)**

```bash
# terminal 1
uvicorn main:app --reload --port 8000

# terminal 2 - if data/ already has a document loaded, just ask about it:
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'
```

Then try the same question through the actual frontend in a browser.

**3. Things worth testing deliberately, once it's running:**
- Ask something *not* in the PDF → the model should say it isn't there, not
  make something up. If it hallucinates, tighten the system prompt in `llm.py`.
- Put two different PDFs in `data/` and ask a question — check `sources` in the
  response to confirm retrieval is pulling from the right document.
- Try a scanned/image-only PDF → `/ingest` should mark it `"status": "skipped"`
  (this pipeline doesn't do OCR; that'd be a `pytesseract` addition later).

## Known limitations (fine for a learning project, worth knowing)

- **Page tagging is approximate for chunks that straddle a page break.** A
  chunk is labeled with the page its *first* word came from — if the actually
  relevant sentence is near the end of that chunk, it may technically be on the
  next page. Good enough for "roughly where to look," not pixel-exact citation.
- **No OCR** — scanned/image PDFs won't extract any text.
- **In-memory chat, no conversation history** — each question is answered
  independently; there's no multi-turn memory yet. (Natural next feature: pass
  recent Q&A pairs into the prompt.)
- **Single global collection** — all ingested PDFs go into one ChromaDB
  collection, so questions retrieve across every ingested document. Fine for
  one user testing locally; for multi-user you'd add per-user or per-session
  collections.
- **No authentication on `/ingest` or `/reset`** — anyone who can reach the API can
  trigger a re-ingest or wipe the store. Fine for local/personal use; add auth before
  exposing this beyond your own machine.
- **Windows ARM64 numpy warning** — you may still see `Numpy built with MINGW-W64 on
  Windows 64 bits is experimental` on startup even with the fixes above. NumPy doesn't yet
  publish an MSVC/OpenBLAS build for Windows ARM64, so pip installs a community MINGW-W64
  build — the warning alone is expected and generally harmless for this project's vector
  math. `pip install --upgrade numpy` gets the newest available ARM64 wheel if you see
  actual numeric errors, not just the warning.

## Natural next steps, if you want to extend it

- Add conversation history (pass last N turns into the prompt).
- Swap `all-MiniLM-L6-v2` for a larger embedding model if retrieval quality
  needs improving on more complex documents.
- Add OCR (`pytesseract`) for scanned PDFs.
- Move from ChromaDB's local persistence to a hosted vector DB if you deploy
  this rather than run it locally.
- Add simple auth in front of `/ingest` and `/reset` before exposing this beyond localhost.
