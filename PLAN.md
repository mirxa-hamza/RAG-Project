# PLAN.md

Roadmap for the Document Q&A RAG project. Status as of 2026-08-31.

## Status: working end-to-end, backend-only ingestion

The core pipeline (ingest from `data/` → chunk → embed → store → retrieve → answer) is
implemented and verified: `test_pipeline_offline.py` drives startup ingestion, `/ingest`,
`/chat`, `/stats`, and `/reset` through a fake embedding model and all checks pass. You've
already dropped two real PDFs (AI: A Modern Approach, Pattern Classification) into `data/`
for real use. A manual smoke test with a real `GROQ_API_KEY` still needs to be run on your
machine (needs live internet + a real key) — see README.md's "Manual smoke test" section.

## Just fixed (this pass)

- [x] **Removed the upload endpoint entirely.** Documents now only enter the system via the
      backend's own `data/` folder — no way for a frontend user to add, replace, or remove
      documents. `POST /upload` is gone; `backend/ingest.py` is the new single entry point,
      called on FastAPI startup and via `POST /ingest` (re-scans without a restart).
      `POST /reset` now wipes the store and immediately re-ingests from `data/`, rather than
      leaving it empty.
- [x] **Frontend upload UI removed.** The drag-and-drop box, file picker, and `/upload` call
      are gone from `index.html`/`script.js`/`style.css`, replaced with a "Sync from data
      folder" button that calls `POST /ingest`.
- [x] **Swapped `pypdf` for `PyMuPDF`** (see prior entry, still in effect) — note PyMuPDF is
      AGPL-3.0 licensed, unlike the rest of the stack.
- [x] **Test suite rewritten** for the new flow: generates fixture PDFs into an isolated
      `/tmp` directory (never `data/`), exercises startup ingestion, `/ingest` idempotency
      (re-running finds nothing new), `/ingest` picking up a newly-added file without a
      restart, and `/reset`'s wipe-then-re-ingest behavior.
- [x] **`make_test_pdf.py` output moved out of `data/`** into `backend/test_fixtures/`
      (gitignored) — `data/` is now the live ingestion source for your real documents and
      must never get a synthetic test PDF mixed into it.
- [x] **Upgraded `chromadb==0.5.5` to `chromadb==1.5.9`.** On Windows ARM64, numpy has no
      wheel below 2.0, but chromadb 0.5.5 hard-requires `numpy<2.0` — that combination
      crashed at import time with `AttributeError: np.float_ was removed in the NumPy 2.0
      release`. (0.5.5 also had a separate posthog telemetry incompatibility, previously
      worked around with a `posthog==2.4.2` pin.) chromadb 1.5.9 drops both the numpy pin
      and the posthog dependency — verified clean with the full offline test suite, no
      extra pins needed, no telemetry spam either.
- [x] **Upgraded `groq==0.11.0` to `groq==1.7.0`.** The old version's HTTP client
      construction passed `proxies=` to `httpx.Client(...)`, which `httpx>=0.28` (already
      pinned in `requirements-dev.txt`) no longer accepts —
      `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` at
      startup. Verified `groq==1.7.0` constructs fine and `chat.completions.create(...)`
      keeps the same call shape `llm.py` relies on.
- [x] **`.gitignore` covers `backend/venv/`, `__pycache__/`, `backend/chroma_db/`,
      `backend/test_fixtures/`, `data/*.pdf`, and common OS/editor junk.**

## Next steps, in rough priority order

1. **Run the manual smoke test for real**, now against your actual textbooks in `data/` —
   confirm the real `all-MiniLM-L6-v2` download + a live Groq call work end-to-end. Note:
   embedding a large book (the AI: A Modern Approach PDF is ~38MB) on first startup will
   take noticeably longer than the small fictional test fixture did — that's expected, not
   a hang.
2. **Windows ARM64 numpy warning**, if you still see it after the chromadb upgrade above —
   see the note in README.md and CLAUDE.md. Try `pip install --upgrade numpy` if you hit
   real numerical issues; so far this is just a noisy warning, not a confirmed correctness
   bug in this project.
3. **Conversation history.** Right now each question is answered independently. Natural
   next step: pass the last N question/answer pairs into the prompt in `llm.py` so
   follow-up questions ("what about the second one?") work.
4. **Per-document retrieval scoping.** With two full textbooks now in the store, a question
   can retrieve chunks from *either* book even if you only care about one. Add a `source`
   filter option to `/chat` (e.g. `{"question": "...", "source": "Pattern Classification..."}`)
   so you can scope a question to one document.
5. **OCR for scanned PDFs.** `pdf_utils.extract_pages` returns nothing for image-only PDFs,
   and `ingest_one()` correctly reports `"status": "skipped"` rather than silently storing
   nothing — but there's no path forward for those documents yet. Add `pytesseract` as an
   opt-in fallback when `extract_text()` comes back empty.
6. **Retrieval quality tuning**, now that you're testing on real technical textbooks rather
   than the fictional sample:
   - Try a larger embedding model if `all-MiniLM-L6-v2` misses on dense technical content.
   - Tune `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` / `TOP_K` in `.env` — textbook prose
     with equations/figures may chunk differently than the fictional test document did.
7. **If you ever deploy this beyond your own machine:**
   - Move ChromaDB from local persistence to a hosted vector DB.
   - Tighten `allow_origins=["*"]` in `main.py`'s CORS middleware to your real frontend
     origin.
   - Add auth in front of `/ingest` and `/reset` — there's currently none; anyone who can
     reach the API can trigger a re-ingest or wipe the store.

## Explicitly out of scope for now

- A build step for the frontend (it's deliberately plain HTML/CSS/JS).
- Swapping in LangChain/LlamaIndex — the point of this project is seeing every step.
- Any way for the frontend/a user to add, replace, or delete documents — this is a
  deliberate design constraint, not a missing feature. If it ever needs to change, that's a
  product decision to make explicitly, not something to slip back in incidentally.

## How to keep this file useful

Update the "Just fixed" section into a changelog-style history as you go, and re-prioritize
"Next steps" as items get done or new gaps get found. Treat CLAUDE.md as the stable
reference (architecture, conventions) and this file as the living to-do list.
