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

**The app is multi-tenant.** Accounts live in MongoDB; every route that touches documents
requires a signed-in user; every document belongs to exactly one account and is invisible
to every other. Uploads land in `data/users/<user_id>/`, and every stored chunk carries a
`user_id` in its metadata. Read the isolation rules in Conventions before touching
retrieval, ingestion or any endpoint — a missed filter there is a data breach, not a bug.

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
    documents.py       /ingest, /ingest/status, /stats, /api/documents, /reset, /upload,
                       DELETE /documents/{name}   (all require a signed-in user)
    system.py          /health, /info, /api/health/auth (is MongoDB reachable?)
  api/deps.py          get_current_user - the single auth gate every document route uses
  api/auth.py          /api/signup, /api/login, /api/me
  core/                cross-cutting: imported by everything else
    config.py          every setting; paths resolve against PROJECT_ROOT, not the CWD
    logging.py         text or JSON logging, request ids, `timed()` context manager
    ratelimit.py       in-process sliding-window limiter (single-worker only)
  models/
    schemas.py         pydantic request/response shapes
  ml/                  model wrappers
    embeddings.py      sentence-transformers singleton, query prefix, truncation guard
    reranker.py        lazy cross-encoder singleton, degrades gracefully if unavailable
    llm.py             grounded prompt, Groq call (sync + streaming), follow-up rewriting
  services/            pipeline logic
    database.py        Motor client + the users collection (accounts only live here)
    security.py        Argon2id hashing (pwdlib) + JWT encode/decode
    ownership.py       adoption of pre-auth documents, owner of record, account cleanup
    answer_cache.py    per-user LRU of finished answers, invalidated by document changes
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
  backup.py            archives data/ + MongoDB; --verify reads a backup back
  verify_index.py      finds orphan/ownerless/missing chunks; --fix repairs them
  draft_golden.py      drafts eval questions from the real corpus for a human to edit
Dockerfile / docker-compose.yml   app + MongoDB + volumes, models baked into the image
  make_test_pdf.py     fixture generator -> tests/fixtures/, never data/
eval/
  golden_questions.json  golden set (currently fixture questions - replace with real ones)
  run_eval.py            hit-rate@k, MRR, refusal rate, optional LLM-as-judge
tests/
  test_pipeline_offline.py   220 checks, offline; no Groq key and no MongoDB needed
data/                  the live ingestion source, gitignored
  users/<user_id>/     one folder per account: everything uploaded through the web UI
  <anything else>      hand-copied PDFs; owned by the "owner of record" (first account)
storage/chroma_db/     generated index state, gitignored
requirements.txt / requirements-dev.txt / .env.example / .env   (project root)
refrence/              third-party reference project kept for study; gitignored
```

## Isolation (read this before touching retrieval or any endpoint)

A user must never see, search, or delete another user's document. Four rules make that
true, and all four are load-bearing:

1. **Every stored chunk carries `user_id` in its Chroma metadata**, written by
   `vectorstore.add_chunks`. A chunk without one is owned by nobody and matches no filter.
2. **Three separate stages can reach stored text, and each filters independently:**
   - `vectorstore.query_chunks` — Chroma `where` clause (isolation point 1)
   - `bm25.search` — the BM25 index is built over the WHOLE corpus for cost reasons, so it
     filters the ranking in memory, *before* applying the limit (isolation point 2)
   - `vectorstore.get_neighbors_bulk` — fetches by `chunk_index`, bypassing search
     entirely, so it needs its own owner clause (isolation point 3)
   Filtering only the first is the classic mistake: the answer still leaks through BM25 or
   through the neighbour of a legitimate hit.
3. **`retrieval.retrieve()` re-checks ownership on the way out** and logs an error if
   anything slipped through. That assertion is a smoke alarm, not the fix.
4. **Ownership is derived from the document's path** (`ingestion.owner_from_path`), not
   from a field a request can set. Uploads go to `data/users/<user_id>/`, and the API
   attaches the *authenticated* caller's id, never anything from the request body.

Endpoint rules that follow from it:

- **Every document route depends on `get_current_user`.** There is deliberately no
  "optional user" dependency: a route that can run without one is a route that can leak.
- **Someone else's document is a 404, never a 403.** A 403 confirms it exists.
- **`/stats`, `/api/documents` and `/ingest/status` are all filtered.** The job is global -
  one thread indexes everybody's uploads - so `job_status(user_id)` redacts `current_file`
  and `results`; another user's file in progress shows as busy with no name.
- **`/reset` rebuilds only the caller's documents.** It used to wipe the whole store, which
  with several accounts would throw away everyone else's work.
- **Deduplication is per owner.** Globally, the second person to upload a given book got a
  `skipped` document they could never see, because the only stored copy was someone else's.

## Limits, and why each one exists

Every one of these was added after an audit found a way to spend somebody else's money or
CPU. Removing one re-opens the specific hole named beside it.

| Limit | Setting | Without it |
|---|---|---|
| History field lengths | `Turn` in schemas.py | one request measured at 4,000,671 characters to Groq |
| Assembled history | `MAX_HISTORY_CHARS` | several legal-sized turns add up to the same thing |
| Login attempts | `ratelimit.LOGIN` | password guessing, and Argon2 becomes a CPU-exhaustion primitive |
| Signups per IP | `ratelimit.SIGNUP` | unbounded account creation |
| Uploads per account | `ratelimit.UPLOAD` | minutes of CPU per request, on demand |
| Questions per account | `ratelimit.CHAT` | unbounded Groq spend |
| Per-user storage | `MAX_USER_STORAGE_MB` | one account fills the disk and stops MongoDB for everyone |
| Job result list | `MAX_JOB_RESULTS` | unbounded memory on a long-lived server |

`src/core/ratelimit.py` and `src/services/answer_cache.py` hold state **in process**. That
is correct only because the app is single-worker (below); with N workers every limit is
effectively N times larger.

## Deployment invariants

- **ONE uvicorn worker. Not negotiable.** The ingestion job, the BM25 cache, the rate
  limiter and the answer cache are all in-process, and ChromaDB's persistent client is
  single-process. With `--workers 4`: progress bars hang (a random worker answers
  `/ingest/status`), BM25 goes stale in three workers out of four, rate limits quadruple,
  and two processes write the same SQLite file. To scale, move the job to a queue and the
  vectors to a server-based store first.
- **`/health` is liveness, `/ready` is readiness.** `/health` answers as soon as the process
  is up - which is before the embedding model has loaded and regardless of MongoDB.
  Anything that routes traffic must probe `/ready`, which is 503 until both are true.
- **`data/` is the only copy of user documents.** `storage/` is derived and rebuildable;
  `data/` is not, and neither is MongoDB. `scripts/backup.py` covers both, and
  `--verify` exists because a backup nobody has read back is a hope, not a backup.
- **Models are baked into the Docker image and `HF_HUB_OFFLINE=1` is set**, so a running
  container never depends on Hugging Face being up, and a model repository cannot change
  under a pinned name.

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
  so the wait happens in the background, with the status dot reporting it. Never move a model load to import
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
  static asset (currently `?v=12`).
- **Auth code: `pwdlib` with Argon2id, never `passlib`.** passlib is unmaintained and
  breaks against recent bcrypt releases. `PasswordHash.recommended()` also gives hash
  migration for free.
- **`JWT_SECRET` is generated once and persisted to `.env`.** A key regenerated per restart
  signs everyone out on every reload; a hard-coded default lets anyone mint a token. The
  decode call pins the algorithm - accepting the token header's `alg` is the classic JWT
  forgery.
- **Login failures are indistinguishable.** "No such user" and "wrong password" return the
  same 401 with the same message, and a missing user still pays for a hash so the timing
  matches. Anything else enumerates accounts.
- **Username uniqueness is enforced by a unique Mongo index**, created at startup. The
  pre-check in the handler is a courtesy; two simultaneous signups both pass it.
- **Confirmation failures are 403, never 401.** A wrong *current* password on the change
  form, or a wrong confirmation on account deletion, is not an authentication failure - the
  token is fine. Returning 401 made the frontend's "any 401 ends the session" rule sign the
  user out for a typo, which is how this was found.
- **`token_version` is what makes a JWT revocable.** It is stamped into every token and
  compared on every request; a password change or "sign out everywhere" increments it and
  every older token dies immediately. Any new place that mints a token must pass the
  account's current version.
- **Only accounts live in MongoDB.** Documents, vectors and the manifest stay in Chroma and
  on disk. Splitting one document's identity across two databases means a delete can
  half-succeed.
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
  `syncMenuButton()` keeps the toggle's `aria-expanded`/`aria-label` honest for both. It
  holds the conversation, not the library: one **New chat** button.
- **Everything about documents lives in one slide-over** (`#docsPanel`, opened from the
  Documents button in the top bar or the composer's Attach file): the dropzone, the upload
  cards, and the library list with its delete buttons and its scope tick boxes. Processing
  status and Rebuild index live in the top bar instead - occasional maintenance actions
  that were competing with the document list for space. It is deliberately NOT a `<dialog>` — the chat has to stay readable behind
  it — so `openDocs()`/`closeDocs()` do by hand what a dialog gives for free: move focus in
  and back out, trap Tab inside the panel, close on Escape and on a backdrop click. The
  panel is `visibility: hidden` when closed, not merely translated off-screen, or its
  buttons would still be tab stops over the chat.
- **The search scope is a multi-select, and an empty selection means EVERYTHING.** Ticking
  documents in the panel narrows retrieval; the chip at the bottom right of the composer
  says what is currently searched and opens the panel. `ChatRequest.sources` carries the
  list (`source` still works for one), and both vector backends turn it into an `$in`
  filter. "Nothing ticked = search nothing" would answer "not in these documents" to every
  question and look exactly like a broken index, which is why it is not a reachable state.
  A document deleted while selected is dropped from the selection by `renderScope()`,
  otherwise the filter keeps narrowing to something the server no longer has.
- **Closing the panel cancels nothing:** the transfer and the indexing carry on, and the
  panel picks the state back up (it calls `refreshSources()` on every open).
- **Two progress phases, from two sources.** The transfer has real byte counts (XHR
  `upload.progress` — `fetch()` has no equivalent, which is why it is XHR); indexing is
  polled from `/ingest/status`, which reports `stage`, `chunks_done` and `chunks_total` so a
  900-page book is not a bar frozen at 0% for five minutes.
- **`MAX_UPLOAD_MB` comes from `/stats`,** not a hard-coded copy — the client-side size
  check must not drift from the server's real limit.
- **Every request goes through `authFetch()`**, which attaches the bearer token and treats
  any 401 as "the session is over". The one exception is the XHR upload (fetch has no
  upload-progress events), which sets the header by hand and handles 401 itself — if you
  add a request, use `authFetch`.
- **The SSE stream carries the token** only because it is read with `fetch()`; `EventSource`
  cannot set headers. Don't "simplify" it to `EventSource`.
- **`resetAppState()` runs on every sign-in and sign-out.** Without it the previous
  account's chat transcript and document rows sit on screen until the first refresh lands,
  which looks exactly like a leak even though the server sent none of it.
- **Ownership of a JOB is `scope`, not `current_file`.** `job_status(user_id)` reveals
  progress when the run was started for that user, or when the file in flight is theirs.
  Deciding from `current_file` alone was a bug: it is None at the start of a run, between
  files and for the whole finished state, so the counts were blanked exactly when the UI
  needed them and the progress bar never moved.
- **The activity trail (`_state["events"]`) is per user and bounded.** It is what the
  Processing status panel shows - the steps the pipeline actually took, tagged with whose
  document they concern, so another account's filenames never appear in it.
- **The app is `hidden` until `/api/me` confirms the stored token.** A token in
  localStorage is not proof of a session - it may be expired or signed with a rotated key.
  The consequence to remember: while it is hidden, nothing inside it can be measured (see
  the `autoGrow()` gotcha), so anything that sizes itself from layout must run, or re-run,
  in `startSession()`.

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
python tests/test_pipeline_offline.py    # does the plumbing work?  (220 checks, offline)
python eval/run_eval.py                  # are the answers any good? (needs an index)
```

`tests/` stubs `sentence_transformers` (both `SentenceTransformer` and `CrossEncoder`) via
`sys.modules`, substitutes an in-memory fake for the Mongo users collection
(`database.set_users_collection`), points `DATA_DIR`/`CHROMA_DIR` at temp folders, and
drives real HTTP endpoints through `TestClient`. Hashing, JWTs, the auth dependency and
every isolation filter are the real code.

**The isolation checks are the ones that matter.** They sign up two accounts, upload a
byte-identical document to each, and assert that neither can reach the other's chunks
through vector search, through BM25, or through neighbour expansion - each verified
separately, because a single end-to-end check passes even when two of the three filters
are missing. Sabotaging any one of them makes a specific check fail; if you change
retrieval, confirm that is still true. It does **not** prove the real models download or that a
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

`eval/golden_questions.json` still holds questions about the *fixture* PDF so the harness
runs out of the box. Replacing them with 20-30 questions about the real documents remains
the single highest-value thing left to do; `scripts/draft_golden.py` does the tedious half
(finding candidate passages and recording their pages) and marks every entry
`"reviewed": false` so nobody mistakes a draft for a measurement - `run_eval.py` prints a
warning when it sees them.

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
- **Never measure a hidden element and write the result back.** `autoGrow()` runs at
  startup, and once the app began starting out `hidden` behind the sign-in screen, its
  `scrollHeight` was 0 - so it wrote `height: 0px` and the composer stayed collapsed to its
  padding after login. It now bails out when the element has no layout, refuses to write a
  zero height, and runs again from `startSession()`; CSS carries a `min-height` floor as
  the backstop. The same trap applies to any layout measurement taken before sign-in.
- **A background loop MUST stop when the session ends.** `endSession()` bumps
  `sessionEpoch`; every polling loop captures it and exits when it changes, and `authFetch`
  tags its 401 error `signedOut` so a loop can tell "stop" from "retry". Without both,
  `watchStartup()` kept polling `/stats` with a dead token during model warm-up: each 401
  called `endSession()` again, which re-rendered the sign-in screen about once a second -
  clearing the password field and yanking focus back to the username box, so the form was
  literally impossible to type into. `showAuthScreen()` also forces the form back to "Sign
  in", or an expiry lands on a "Create an account" form that then reports "Incorrect
  username or password" for an account that exists.
- **There is no loading screen.** The overlay that used to hold the app back during model
  warm-up and the first ingest is gone: sign in and you are in the chat. `watchStartup()`
  polls `/stats` in the background and reports those two states through the top bar's
  status dot instead ("Warming up", "Indexing", with the detail in the tooltip). A question
  asked during warm-up simply waits for the model rather than failing, which is why
  blocking the UI was never worth it. The one thing the dot must keep saying is that an
  ingest is still running, because an answer given then may be missing passages.
- **A database outage is a 503 with instructions, not a 500 with a traceback.**
  `database._guard()` converts pymongo's `ServerSelectionTimeoutError`/`ConnectionFailure`
  into `DatabaseUnavailable`, and an exception handler in `main.py` renders it as a 503
  whose message says how to start MongoDB; the sign-in screen shows that text verbatim.
  Only connection-level failures are converted - `DuplicateKeyError` must keep propagating,
  or a taken username stops returning 409. Every new Mongo call goes through `_guard`.
- **A document's owner comes from its path, never from the request.** An upload's owner is
  the authenticated caller; `owner_from_path()` reads it back off `users/<id>/...`. Trusting
  a `user_id` field in a request body would let anyone claim anyone's documents.
- **Deletes by document name are deliberately NOT owner-scoped** (after the ownership check
  passes). The name is the path, so it can only ever match one user's file - and scoping it
  would strand chunks written before the document had an owner.
- **PyMuPDF is AGPL-3.0**, unlike every other dependency here. Fine for personal/local use;
  comply with AGPL or buy a commercial license before shipping this closed-source.

## Known limitations

- **The golden question set is still the fixture one** — until it covers the real documents,
  the eval numbers describe a fictional 3-page PDF, not the textbooks.
- **OCR is off by default** and needs the Tesseract binary installed separately; without it
  scanned PDFs are reported `"status": "skipped"`.
- **Conversation history is client-side.** The backend is stateless: the frontend sends the
  recent turns with each question. Nothing is persisted across browser reloads.
- **A JWT is revocable now, but only through `token_version`.** Signing out in one browser
  still only clears that browser; "sign out on all devices", a password change, or deleting
  the account are what invalidate outstanding tokens.
- **The API is plain HTTP.** A token on a shared network is sniffable. Localhost only,
  unless you put TLS in front of it.
- **Signup is open.** Anyone who can reach the page can create an account and upload.
- **MongoDB is a hard dependency.** Without it nobody can sign in. The app still starts,
  logs how to fix it, and answers auth requests with a 503 that says the same - it does not
  pretend to work. `GET /api/health/auth` reports whether it is reachable.
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
