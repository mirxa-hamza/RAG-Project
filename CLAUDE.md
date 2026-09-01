# CLAUDE.md

Guidance for Claude (or any AI assistant) working in this repository.

## What this project is

A from-scratch Retrieval-Augmented Generation (RAG) app: add PDFs (upload them in the UI or
drop them in the backend's `data/` folder), ask questions about them, get answers grounded
only in that content. Built
deliberately without LangChain/LlamaIndex so every pipeline step is plain, readable
Python — this is a learning project first, a working app second.

**Documents live in `data/` on the backend's own filesystem, and everything is indexed
from there.** They get into that folder two ways: copied in by whoever runs the server, or
uploaded through `POST /upload` from the web UI. Nothing is ever embedded straight out of a
request body — `/upload` validates and writes the file, then the normal ingestion job picks
it up off disk like any other PDF.

Browser upload was added deliberately (2026-09-01), replacing an earlier rule that there
should be no upload endpoint at all. The trade-off that rule protected is still real and
worth restating: **there is no authentication anywhere in this app**, so anyone who can
reach it can now add documents to the corpus and delete them, PDF and all, from the
server's disk. That is acceptable on localhost and not acceptable on a public address. Put
it behind auth before exposing it, or set `UPLOAD_ENABLED=false` if that switch has been
added by then.

Pipeline:

```
ingest:  PDF -> PyMuPDF extraction (paragraphs preserved, optional OCR fallback)
              -> structure-aware chunking (page ranges recorded)
              -> local embeddings -> ChromaDB

query:   question -> follow-up rewrite (if there's history)
                  -> vector search  ─┐
                  -> BM25 search    ─┴─> Reciprocal Rank Fusion
                  -> cross-encoder re-rank
                  -> relevance floor (nothing clears it => decline, no LLM call)
                  -> neighbour expansion
                  -> Groq answers strictly from those chunks (streamed)
```

## Layout

Purpose-based `src/` package at the project root - one folder per concern, mirroring the
structure used in the owner's other FastAPI projects.

```
src/
  main.py              app assembly: router, CORS, lifespan, static mount
  api/                 HTTP layer - thin handlers, one module per endpoint group
    __init__.py        aggregates the routers into `api_router`
    chat.py            /chat, /chat/stream (SSE)
    documents.py       /ingest, /ingest/status, /stats, /reset, /upload, DELETE /documents/{name}
    system.py          /health, /info
  core/                cross-cutting: imported by everything else
    config.py          every setting; paths resolve against PROJECT_ROOT, not the CWD
    logging.py         logging config + `timed()` context manager
  models/
    schemas.py         pydantic request/response shapes
  ml/                  model wrappers
    embeddings.py      sentence-transformers singleton, query prefix, truncation guard
    reranker.py        lazy cross-encoder singleton, degrades gracefully if unavailable
    llm.py             grounded prompt, Groq call (sync + streaming), follow-up rewriting
  services/            pipeline logic
    ingestion.py       single entry point for documents + background job state machine
    uploads.py         validates + writes browser uploads into DATA_DIR; deletes documents
    manifest.py        JSON sidecar: what's ingested, with content hashes
    pdf.py             extraction (keeps paragraph breaks, optional OCR) + chunker
    vectorstore.py     ChromaDB add/query/neighbours/delete/reset, batched
    bm25.py            lazy in-memory BM25 index (keyword half of hybrid search)
    retrieval.py       the retrieval pipeline: fusion, re-rank, floor, neighbour expansion
  static/              the web UI, served by FastAPI at "/" - index.html / style.css / script.js
scripts/
  ingest.py            CLI index builder (--force, --status)
  run.py               starts uvicorn, opens the browser once /health answers
  make_test_pdf.py     fixture generator -> tests/fixtures/, never data/
eval/
  golden_questions.json  golden set (currently fixture questions - replace with real ones)
  run_eval.py            hit-rate@k, MRR, refusal rate, optional LLM-as-judge
tests/
  test_pipeline_offline.py   126 checks, offline, no Groq key needed
data/                  real PDFs the user drops here - the live ingestion source, gitignored
storage/chroma_db/     generated index state, gitignored
requirements.txt / requirements-dev.txt / .env.example / .env   (project root)
refrence/              third-party reference project kept for study; gitignored
```

## Conventions

- **Every module imports settings from `src.core.config`**, never `os.getenv()` directly.
  New setting → add it there with a sensible default, and to `.env.example`.
- **Dependencies point one way: `api` → `services`/`ml` → `core`.** Route handlers stay
  thin (validate, call a service, shape the response); nothing in `services/` or `ml/`
  imports from `api/`, and `core/` imports from neither.
- **The web UI is served by FastAPI itself** from `src/static/`, mounted at `/` in
  `main.py`. That mount is registered LAST, after `api_router`, so an API path always wins
  over a same-named file. `script.js` uses a relative `API_BASE` because it is same-origin.
- **`config.py` reads env vars at import time**, and resolves relative paths against
  `PROJECT_ROOT`. Tests that need different values must set `os.environ[...]` *before*
  `app.config` is first imported anywhere in the chain.
- **No LangChain/LlamaIndex, no hidden abstractions.** Keep new pipeline code plain and
  readable — that's the point of this project.
- **`src/services/ingestion.py` is the single entry point for adding documents.** Called
  from the CLI, from the FastAPI lifespan startup, from `POST /ingest` and from
  `POST /upload`. All four go through `ingest_data_folder()`, which fingerprints files by
  SHA-256 and skips unchanged ones. **Uploads are not a second pipeline**: `/upload` writes
  the file into `DATA_DIR` and then runs the ordinary job. Nothing is ever embedded straight
  out of a request body.
- **Ingestion never blocks a request.** `start_job()` runs it on a background thread;
  `/ingest`, `/reset` and `/upload` return 202 and the client polls `/ingest/status`.
- **`start_job()` called during a run flags a re-scan; it never starts a second thread.**
  The running job listed the folder when it started, so a PDF uploaded mid-run was invisible
  to it and sat unindexed until someone pressed sync. `_run()` drains
  `_consume_rescan_request()` after the scan finishes.
- **A document's manifest entry is written only after its last chunk is stored**, and the
  UI's document list is built from the manifest. That is what makes "only fully-processed
  documents appear in the library" true; don't write the entry earlier to make progress
  reporting easier.
- **`ingest_one()` calls `delete_source()` for `new` files too, not just `changed` ones.**
  A job killed mid-file leaves stored chunks with no manifest entry, so the retry treats the
  file as new and re-embeds it under a fresh `run_id` — without the delete, every
  interrupted attempt piles up another duplicate copy.
- **`src/services/retrieval.py` owns the query path.** Endpoints call `retrieve()`; they don't talk
  to `vectorstore`/`bm25`/`reranker` directly. Every stage is switchable through both
  config and keyword arguments — that's what lets `eval/run_eval.py` A/B them.
- **An empty retrieval result is a real answer**, not an error: it means nothing cleared
  the relevance floor, and `ml.llm.generate_answer()` returns the "not in these documents"
  message *before* checking for an API key, since that answer needs no LLM call.
- **Heavy models are LAZY singletons — never module-level.** The embedding model, the
  cross-encoder and the BM25 index all load on first use, behind a lock. This is not a
  micro-optimisation: uvicorn imports the app *before* it binds the port, so loading the
  embedding model at import kept the port closed for ~18s and the browser answered
  `ERR_CONNECTION_REFUSED`. `main.py`'s lifespan starts `embeddings.warm_up()` on a thread
  so the wait happens behind the loading screen instead. Never move a model load to import
  time, and never into a per-request path either.
- **Anything that changes stored chunks must call `_invalidate_keyword_index()`** (already
  wired into `add_chunks`, `delete_source`, `reset_collection`) — the BM25 index caches the
  corpus in memory and would otherwise go stale.
- **One bad document must never stop the others.** `ingest_data_folder()` catches per
  file and records `{"status": "failed", "error": ...}`; anything that adds a new failure
  mode inside the per-file path must keep that contract. A corrupt PDF used to abort the
  whole job, leaving every file after it silently un-indexed.
- **`data/` is scanned recursively** and documents are keyed by their POSIX-style relative
  path (`textbooks/norvig.pdf`), so the same identity works on Windows and Linux.
- **The manifest is written atomically** (temp file + `os.replace`) under a lock. A plain
  write leaves a window where a reader sees half a file, decides the store is empty, and
  re-embeds the entire corpus.
- **`src/services/uploads.py` is the only place that turns request bytes into a file.**
  It trusts nothing the client sends: the filename is reduced to a bare, scrubbed segment
  (so `../../.env.pdf` and `C:\Users\me\book.pdf` cannot escape `DATA_DIR`), the content
  must actually start with `%PDF-` regardless of extension or content type, the size cap is
  enforced *while streaming* with the partial file deleted, a colliding name is renamed
  rather than overwritten, and the write is atomic (temp file + `os.replace`) so the
  ingestion thread can never read a half-uploaded PDF. Any new intake path must reuse this
  module rather than reimplementing part of it.
- **Deleting a document must delete the PDF too** (`DELETE /documents/{name}`). Removing
  only the vectors leaves the file in `data/`, and the next startup ingest faithfully
  re-indexes it.
- **HTML is served `no-cache`; CSS and JS are cache-busted with `?v=N` in `index.html`.**
  Chrome served a cached `index.html` against a freshly updated `style.css` once and the new
  markup rendered completely unstyled. Bump the `?v=` number whenever you change either
  static asset.
- `data/` and `tests/fixtures/` are strictly separate: `data/` is real user documents;
  `tests/fixtures/` is synthetic PDFs from `make_test_pdf.py`. Never point the fixture
  generator's default output at `data/`.

## Frontend (`src/static/`)

Three files, no build step, no framework. Conventions that matter:

- **`renderMarkdown()` escapes first, then re-introduces a fixed tag set.** Model output is
  untrusted text; never swap this for an innerHTML pass over raw model output.
- **Answers stream over SSE** (`readSSE()`), repainted once per animation frame rather than
  per token. While retrieval runs the bubble shows a "Searching your documents…" indicator,
  switched to "Writing the answer…" when the `sources` event lands and removed on the first
  token — the bubble is never empty.
- **The sidebar is two different things.** On desktop it is a grid column that collapses to
  a 64px icon rail (`.app.is-collapsed`, labels visually hidden but kept in the
  accessibility tree). Under 860px it is an off-canvas drawer (`.sidebar.is-open` + scrim)
  that hides completely — the rail rules are explicitly reverted in that media query.
  `syncMenuButton()` keeps the toggle's `aria-expanded`/`aria-label` honest for both.
- **Upload lives in a native `<dialog>`**, so focus trapping, Escape and top-layer stacking
  come from the browser. Closing it does not cancel anything: the transfer and indexing
  carry on and the sidebar keeps reporting.
- **Two progress phases, from two sources.** The transfer has real byte counts (XHR
  `upload.progress` — `fetch()` has no equivalent, which is why it is XHR); indexing is
  polled from `/ingest/status`, which reports `stage`, `chunks_done` and `chunks_total` so a
  900-page book is not a bar frozen at 0% for five minutes.
- **`MAX_UPLOAD_MB` comes from `/stats`,** not a hard-coded copy — the client-side size
  check must not drift from the server's real limit.

## Running it

Everything runs from the **project root** now (there is no `backend/` folder):

```bash
python -m venv venv && venv\Scripts\activate   # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then set GROQ_API_KEY
python scripts/ingest.py               # build the index (or let startup do it)
python scripts/run.py                  # or: uvicorn src.main:app --port 8000
```

`scripts/run.py` waits for `/health` before opening the browser, so a cold start never shows
`ERR_CONNECTION_REFUSED`. Add `--reload` only while editing code: the reloader restarts on
every save, and a restart kills an ingest in flight.

Then open **http://localhost:8000** - the UI is served by the same process. There is no
separate HTML file to open. `http://localhost:8000/docs` is the API reference.

The module path is `src.main:app`. First question loads the cross-encoder (~80MB, one-off
download).

## Testing and evaluation

Two different questions, two different tools — don't confuse them:

```bash
python tests/test_pipeline_offline.py    # does the plumbing work?  (126 checks, offline)
python eval/run_eval.py                  # are the answers any good? (needs an index)
```

`tests/` stubs `sentence_transformers` (both `SentenceTransformer` and `CrossEncoder`) via
`sys.modules`, points `DATA_DIR`/`CHROMA_DIR` at temp folders, and drives real HTTP
endpoints through `TestClient`. It does **not** prove the real models download or that a
live Groq call succeeds — do a manual smoke test with a real `GROQ_API_KEY` before
considering a change done.

`eval/run_eval.py` measures **retrieval hit-rate@k** and **MRR** (did the expected page get
retrieved, and how highly ranked?), plus **refusal rate** on deliberately unanswerable
questions. All three need no API key. `--judge` adds LLM-as-judge correctness /
groundedness / relevance via Groq. **Any change to chunking, the embedding model, BM25, or
the re-ranker must be measured here before and after** — that's the whole reason it exists:

```bash
python eval/run_eval.py                       # baseline
python eval/run_eval.py --no-rerank           # what is the cross-encoder worth?
python eval/run_eval.py --no-hybrid           # what is BM25 worth?
python eval/run_eval.py --top-k 8 --out runs/topk8.json
```

`eval/golden_questions.json` currently holds questions about the *fixture* PDF so the
harness runs out of the box. Replacing them with 20–30 questions about the real documents
in `data/` is the single highest-value thing left to do.

## Known gotchas (already fixed, keep them fixed)

- **Embedding truncation.** `all-MiniLM-L6-v2` reads only 256 tokens; the project's
  ~300-word chunks were being silently truncated, so the tail of every chunk never
  influenced retrieval. Now on `BAAI/bge-small-en-v1.5` (512 tokens), and
  `embeddings.warn_if_truncated()` logs a warning if chunks ever exceed the window again.
  If you switch models, check `max_seq_length` and set `EMBEDDING_QUERY_PREFIX`
  accordingly (bge/e5 want a query-side prefix; MiniLM wants none).
- **Paragraph structure must survive extraction.** `_normalize()` deliberately keeps
  `\n\n`. Collapsing all whitespace (`" ".join(text.split())`) destroys the boundaries the
  chunker packs on and measurably worsened page attribution.
- **BM25 tokenisation keeps technical characters.** `bm25.tokenize()` uses a custom regex,
  not `\w+`, so `A*`, `k-means` and `f1` survive as tokens. A plain `\w+` split turns "A*"
  into "a" — a token that matches everything and ranks nothing.
- **Cross-encoder scores are raw logits, not probabilities.** Compare them against
  `MIN_RERANK_SCORE`; don't display them as confidences or normalise them to 0..1.
- **Neighbour chunks carry no `similarity`/`rerank_score`** — they're fetched by
  `chunk_index`, not by search. Any code reading those fields must tolerate `None`.
- **`EMBEDDING_QUERY_PREFIX` must end in a space, and `config.py` re-adds it.**
  python-dotenv strips trailing whitespace from unquoted `.env` values, which silently
  glued the prefix to the question (`...passages:What is A* search?`) and degraded every
  query embedding. Never remove the normalisation in `config.py`.
- **`build_context()` never returns an empty string.** If the first chunk is larger than
  the whole `MAX_CONTEXT_CHARS` budget it is truncated, not dropped — the system prompt
  orders the model to answer only from CONTEXT, so handing it nothing is an invitation to
  answer from its own weights.
- **Deleted documents are pruned on every ingest** (`prune_deleted()`). Without it, a PDF
  removed from `data/` stayed searchable and citable forever.
- **Neighbour lookups are batched** (`vectorstore.get_neighbors_bulk`) — one query per
  source, not one per hit. Don't reintroduce a per-hit `get()` inside a loop.
- **The re-ranker retries after a failed load** instead of latching off permanently; a
  transient failure on the first question used to disable the biggest quality stage for
  the life of the process.
- **Use `chromadb>=1.0`, not 0.5.x.** 0.5.5 hard-requires `numpy<2.0`, which has no Windows
  ARM64 wheel — pip is forced onto numpy 2.x and 0.5.5 then crashes at import
  (`np.float_` removed). 1.x also dropped the posthog telemetry dependency and its log spam.
- **Use `groq>=1.x`, not 0.11.0.** The old client passes a `proxies` kwarg that
  `httpx>=0.28` rejects: `TypeError: Client.__init__() got an unexpected keyword argument`.
- `pdf_utils.py` imports `pymupdf`, not `fitz` (the old name is deprecated), and silences
  MuPDF's per-image `cmsOpenProfileFromMem` stderr spam.
- **The scrim must stack BELOW the mobile drawer.** `.scrim` at `z-index: 25` sat over the
  sidebar's `20`, so every button inside the open drawer looked enabled and did nothing on
  phones. Scrim is 15, sidebar 20, topbar 30, dialog/boot above that.
- **The composer textarea sets `overflow-y` from JS, and CSS keeps it `hidden`.** Left on
  the browser default, a single-line box showed a scrollbar gutter because `scrollHeight`
  rounds past `clientHeight` at fractional line heights. `autoGrow()` switches it to `auto`
  only once the content really exceeds `max-height` (4 lines).
- **The boot overlay starts `hidden` and is revealed only when the app is unusable** —
  server not answering, model still loading, or an ingest running. It must never block on an
  empty index: that would hide the Add documents button that fixes exactly that.
- **PyMuPDF is AGPL-3.0**, unlike every other dependency here. Fine for personal/local use;
  comply with AGPL or buy a commercial license before shipping this closed-source.

## Known limitations

- **The golden question set is still the fixture one** — until it covers the real documents,
  the eval numbers describe a fictional 3-page PDF, not the textbooks.
- **OCR is off by default** and needs the Tesseract binary installed separately; without it
  scanned PDFs are reported `"status": "skipped"`.
- **Conversation history is client-side.** The backend is stateless: the frontend sends the
  recent turns with each question. Nothing is persisted across browser reloads.
- **No auth on `/ingest`, `/reset`, `/upload` or `DELETE /documents/{name}`** — fine
  locally, unacceptable publicly: two of those write to and delete from the server's disk.
- **The BM25 index is per-process and in memory.** Multiple workers each build their own;
  it rebuilds on restart (one O(corpus) read).
- **Uploads are unauthenticated and unthrottled.** Size is capped per file, but nothing
  limits how many requests arrive; on a public address that is a disk-filling vector.
- **Which of two byte-identical copies wins is scan order, not recency.** Upload the same
  book twice and the alphabetically-first name is indexed while the other is `skipped` —
  even if the skipped one was already in the store.
- **Duplicate detection is byte-exact only.** Two PDFs with identical bytes are caught; the
  same book re-exported or re-scanned is not, and will compete with itself in retrieval.
- **`prune_deleted()` only reconciles the manifest against disk.** Chunks orphaned by an
  interrupted ingest (killed between `delete_source` and `add_chunks`) are not detected;
  `python scripts/ingest.py --force` is the repair.
- **Follow-up rewriting costs an extra Groq call** per question that has history. Set
  `REWRITE_FOLLOWUPS=false` to trade follow-up quality for latency and tokens.

## Before you commit

- Never commit `.env` (real API key) or documents in `data/` — both gitignored.
- Run `tests/test_pipeline_offline.py`; all checks must pass.
- If you touched anything in the retrieval path, run `eval/run_eval.py` before and after
  and put the numbers in the commit message or PLAN.md.
