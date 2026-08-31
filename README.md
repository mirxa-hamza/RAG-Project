# Document Q&A — a from-scratch RAG system

PDF in → chunked & embedded locally → stored in ChromaDB → retrieved at question time →
answered by an LLM on Groq, grounded only in what's in the PDF.

No LangChain/LlamaIndex — every step is plain Python so you can see exactly what's happening
at each stage. That's deliberate, given this is a learning project.

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
│   ├── main.py                    FastAPI app: /upload, /chat, /stats, /reset, /health
│   ├── config.py                  loads all settings from .env
│   ├── pdf_utils.py                PDF → pages → overlapping chunks
│   ├── embeddings.py               wraps the sentence-transformers model
│   ├── vectorstore.py              ChromaDB: add_chunks / query_chunks
│   ├── llm.py                      builds the prompt, calls Groq
│   ├── requirements.txt
│   ├── requirements-dev.txt        adds test-only deps (httpx, reportlab)
│   ├── .env.example                copy to .env and fill in your Groq key
│   ├── make_test_pdf.py            generates a sample PDF for testing
│   └── test_pipeline_offline.py    automated end-to-end test (see below)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── data/
    └── sample.pdf                  generated test document
```

## How a question actually gets answered (the RAG loop)

**At upload time (once per PDF):**
1. `PyMuPDF` extracts text page by page.
2. The text is flattened into a stream of words and sliced into overlapping
   chunks (default: 300 words per chunk, 50-word overlap) — overlap exists so a
   sentence that matters doesn't get cut in half between two chunks and lost.
3. Each chunk is embedded (turned into a vector of numbers that captures its
   meaning) by a local model — no API call, no cost.
4. The chunk text + its vector + its page number get stored in ChromaDB.

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

## Running it

```bash
# from rag-project/backend, with venv active
uvicorn main:app --reload --port 8000
```

First run will download the embedding model (~90MB) — that needs a normal internet
connection and takes a few seconds to a minute, then it's cached locally for good.

Then open `frontend/index.html` directly in a browser (double-click it, or use
VS Code's Live Server extension). It talks to `http://localhost:8000` by default —
change `API_BASE` at the top of `frontend/script.js` if you serve the backend
elsewhere.

**Try it:**
1. Upload a PDF using the sidebar.
2. Ask a question about its content in the chat box.
3. The answer appears with a "Sources" line showing which page(s) and how
   similar each retrieved chunk was to your question (0–1, higher = closer match).

## Testing

Two layers of testing are included:

**1. Automated pipeline test (`test_pipeline_offline.py`)**
Generates a small sample PDF and drives every FastAPI endpoint (`/upload`,
`/chat`, `/stats`, `/reset`) through `TestClient`, checking that PDF parsing,
chunking, and ChromaDB storage/retrieval all behave correctly.

```bash
pip install -r requirements-dev.txt
python3 make_test_pdf.py
python3 test_pipeline_offline.py
```

This has already been run during development — all checks pass. It swaps in a
lightweight fake embedding model instead of the real one, purely so it can run
without a live download in restricted environments; on your machine you can
freely rely on it since your internet access is normal. It also doesn't call
Groq (no API key needed) — it just confirms the app correctly reports "no key
set" instead of crashing, so you know that failure path is handled.

**2. Manual smoke test (do this once you have a real GROQ_API_KEY)**

```bash
# terminal 1
uvicorn main:app --reload --port 8000

# terminal 2
curl -X POST http://localhost:8000/upload -F "file=@../data/sample.pdf"
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question": "How much funding did the project receive?"}'
```

You should get back a JSON answer citing the $1.2 million figure and page 2.
Then try the same question through the actual frontend in a browser.

**3. Things worth testing deliberately, once it's running:**
- Ask something *not* in the PDF → the model should say it isn't there, not
  make something up. If it hallucinates, tighten the system prompt in `llm.py`.
- Upload two different PDFs and ask a question — check `sources` in the
  response to confirm retrieval is pulling from the right document.
- Try a scanned/image-only PDF → `/upload` should return a clear 422 error
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
- **Single global collection** — all uploaded PDFs go into one ChromaDB
  collection, so questions retrieve across every uploaded document. Fine for
  one user testing locally; for multi-user you'd add per-user or per-session
  collections.

## Natural next steps, if you want to extend it

- Add conversation history (pass last N turns into the prompt).
- Swap `all-MiniLM-L6-v2` for a larger embedding model if retrieval quality
  needs improving on more complex documents.
- Add OCR (`pytesseract`) for scanned PDFs.
- Move from ChromaDB's local persistence to a hosted vector DB if you deploy
  this rather than run it locally.
