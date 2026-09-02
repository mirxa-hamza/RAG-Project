# CLAUDE.md

Guidance for Claude (or any AI assistant) working in this repository.

## What this project is

**Marginalia** — a from-scratch Retrieval-Augmented Generation (RAG) app: add PDFs (upload
them in the UI or drop them in the backend's `data/` folder), ask questions about them, get
answers grounded only in that content. The name is the product's argument: an answer is a
note in the margin of a page you own, and it points at the line it came from. The product
name lives in exactly four places - `index.html` (title, wordmark, favicon, auth card),
`main.py`'s FastAPI title, `README.md`, and `TOKEN_KEY` in `script.js` - so renaming again
is a small, findable job. Built
deliberately without LangChain/LlamaIndex so every pipeline step is plain, readable
Python — this is a learning project first, a working app second.

**The app runs in two modes, chosen by `RAG_MODE` in `src/core/config.py`.** `local`
(default) is the development shape this project started as: models on your own CPU, vectors
in a Chroma folder, PDFs in `data/` on disk, job/rate-limit/cache state in process memory —
costs nothing, needs nothing provisioned. `cloud` is the production shape this app also now
runs in, aimed at a Vercel deployment: Pinecone for embeddings and re-ranking, Chroma Cloud
for vectors, Cloudinary for PDFs, MongoDB for both accounts and everything that used to live
in process memory. Almost every module reads a setting (`EMBEDDINGS_PROVIDER`,
`VECTOR_STORE`, `DOCUMENT_STORE`, `STATE_STORE`, ...) that defaults from `RAG_MODE` but can
be overridden individually — see `core/config.py`'s `_mode_default()` — so you can mix, e.g.
run cloud-mode vector/embedding providers against a local disk of PDFs while developing.
**The two modes are the same application code end to end** — no local-only or cloud-only
branch of the *pipeline* logic exists; only the storage/state backends underneath it differ.
See "Cloud mode (RAG_MODE=cloud)" below for the specifics, and `CLOUD_MIGRATION_PLAN.md` for
the reasoning behind how it was built.

**Documents live in `data/` on the backend's own filesystem in local mode, and everything is
indexed from there.** They get into that folder two ways: copied in by whoever runs the
server, or uploaded through `POST /upload` from the web UI. Nothing is ever embedded straight
out of a request body — `/upload` validates and writes the file, then the normal ingestion job
picks it up off disk like any other PDF. **In cloud mode there is no `data/` folder at all** —
PDFs live in Cloudinary and are tracked in a Mongo registry instead; see below.

**The app is multi-tenant.** Accounts live in MongoDB; every route that touches documents
requires a signed-in user; every document belongs to exactly one account and is invisible
to every other. In local mode, uploads land in `data/users/<user_id>/`, and every stored
chunk carries a `user_id` in its metadata. In cloud mode the same `users/<user_id>/filename.pdf`
identity is used as the Mongo registry's document key, so every rule below applies unchanged
regardless of mode. Read the isolation rules in Conventions before touching
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

In cloud mode, "local embeddings"/"ChromaDB"/"cross-encoder" above are swapped for their
provider-backed equivalents (Pinecone embeddings, Chroma Cloud, Pinecone rerank) via the
`EMBEDDINGS_PROVIDER`/`VECTOR_STORE`/`RERANKER_PROVIDER` switches — the pipeline *shape*
above is identical either way.

## Layout

Purpose-based `src/` package at the project root - one folder per concern, mirroring the
structure used in the owner's other FastAPI projects.

```
src/
  main.py              app assembly: router, CORS, lifespan, static mount
  api/                 HTTP layer - thin handlers, one module per endpoint group
    __init__.py        aggregates the routers into `api_router`
    chat.py            /chat, /chat/stream (SSE)
    documents.py       /ingest, /ingest/status, /ingest/continue, /stats, /api/documents,
                       /reset, /upload, /upload/sign, /upload/complete,
                       DELETE /documents/{name}   (all require a signed-in user)
    system.py          /health, /info, /api/health/auth (is MongoDB reachable?)
  api/deps.py          get_current_user - the single auth gate every document route uses
  api/auth.py          /api/signup, /api/login, /api/me
  core/                cross-cutting: imported by everything else
    config.py          every setting; paths resolve against PROJECT_ROOT, not the CWD
    logging.py         text or JSON logging, request ids, `timed()` context manager
    ratelimit.py       sliding-window limiter; in-process (local) or Mongo-backed (cloud)
  models/
    schemas.py         pydantic request/response shapes
  ml/                  model wrappers
    embeddings.py      sentence-transformers singleton, query prefix, truncation guard
    reranker.py        lazy cross-encoder singleton, degrades gracefully if unavailable
    llm.py             grounded prompt, Groq call (sync + streaming), follow-up rewriting
    providers.py       Pinecone/Cohere/Jina embedding+rerank HTTP clients (cloud mode)
  services/            pipeline logic
    database.py        Motor (async, accounts) + pymongo (sync, cloud-mode state) clients
    security.py        Argon2id hashing (pwdlib) + JWT encode/decode
    ownership.py       adoption of pre-auth documents, owner of record, account cleanup
    answer_cache.py    per-user LRU of finished answers; in-process or Mongo-backed
    ingestion.py       single entry point for documents + local-thread/cloud-per-request job models
    uploads.py         validates + writes browser uploads (DATA_DIR or Cloudinary); deletes documents
    manifest.py        what's ingested, with content hashes; JSON sidecar or Mongo-backed
    cloudinary_store.py  cloud mode only: signed direct-upload payloads, public_id ownership
                       check, fetch-by-URL, destroy - raw REST + httpx, no Cloudinary SDK
    cloud_documents.py cloud mode only: Mongo registry of cloud-stored PDFs, same
                       users/<id>/name.pdf identity local mode uses
    pdf.py             extraction (keeps paragraph breaks, optional OCR) + chunker
    vectorstore.py      ChromaDB (local/Chroma Cloud) add/query/neighbours/delete/reset, batched
    vector_pinecone.py  Pinecone vector store backend (VECTOR_STORE=pinecone)
    vector_chroma.py    Chroma backend, disk or Chroma Cloud (CHROMA_BACKEND=disk|cloud)
    bm25.py             lazy in-memory BM25 index (keyword half of hybrid search) - per-user,
                       runs in both modes
    retrieval.py        the retrieval pipeline: fusion, re-rank, floor, neighbour expansion
  static/              the web UI, served by FastAPI at "/" - index.html / style.css / script.js
scripts/
  ingest.py            CLI index builder (--force, --status)
  run.py               starts uvicorn, opens the browser once /health answers
  backup.py            archives data/ + MongoDB; --verify reads a backup back
  verify_index.py      finds orphan/ownerless/missing chunks; --fix repairs them
  draft_golden.py      drafts eval questions from the real corpus for a human to edit
  check_cloud.py       cloud preflight: are the credentials, quotas and dimensions real?
Dockerfile / docker-compose.yml   app + MongoDB + volumes, models baked into the image
  make_test_pdf.py     fixture generator -> tests/fixtures/, never data/
eval/
  golden_questions.json  golden set (currently fixture questions - replace with real ones)
  run_eval.py            hit-rate@k, MRR, refusal rate, optional LLM-as-judge
tests/
  test_pipeline_offline.py   offline checks (local pipeline + cloud-state Mongo fakes);
                             no Groq key, no live MongoDB, no live Cloudinary/Pinecone needed
data/                  the live ingestion source in LOCAL MODE ONLY, gitignored
  users/<user_id>/     one folder per account: everything uploaded through the web UI
  <anything else>      hand-copied PDFs; owned by the "owner of record" (first account)
storage/chroma_db/     generated index state (local disk Chroma), gitignored
requirements.txt / requirements-dev.txt        full local-mode dependencies (incl. torch)
requirements-cloud.txt                         cloud-mode dependencies (no torch/sentence-transformers)
api/index.py / api/requirements.txt            Vercel ASGI entrypoint + its own copy of
                                               requirements-cloud.txt (@vercel/python looks
                                               for requirements.txt next to the entrypoint)
vercel.json            routes all traffic to api/index.py, sets maxDuration
.env.example / .env   (project root)
CLOUD_MIGRATION_PLAN.md  the plan this cloud mode was implemented from, kept as a record of
                         what shipped, what deviated from the original plan, and what is
                         still unverified against live cloud credentials
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
   from a field a request can set. In local mode, uploads go to `data/users/<user_id>/`; in
   cloud mode, `cloud_documents.register()` writes the same `users/<user_id>/filename.pdf`
   key into Mongo. Either way the API attaches the *authenticated* caller's id, never
   anything from the request body. **The cloud-mode-specific version of this rule**: a
   client can claim any Cloudinary `public_id` it wants when calling `POST
   /upload/complete` — `cloudinary_store.public_id_belongs_to(public_id, user_id)` is what
   stops that request from being used to make the server fetch and index a *different*
   user's Cloudinary file. It checks that the public_id is actually inside that user's own
   folder (`{CLOUDINARY_FOLDER}/users/{user_id}/...`), not merely that it starts with the
   same characters — a folder named `{CLOUDINARY_FOLDER}-evil/...` must NOT match. This
   check has its own dedicated tests; treat it with the same weight as the three isolation
   points above.

Endpoint rules that follow from it:

- **Every document route depends on `get_current_user`.** There is deliberately no
  "optional user" dependency: a route that can run without one is a route that can leak.
- **Someone else's document is a 404, never a 403.** A 403 confirms it exists.
- **`/stats`, `/api/documents` and `/ingest/status` are all filtered.** The job is global -
  one thread (local mode) or one per-user Mongo job document (cloud mode) indexes
  everybody's uploads - so `job_status(user_id)` redacts `current_file`
  and `results`; another user's file in progress shows as busy with no name.
- **`/reset` rebuilds only the caller's documents.** It used to wipe the whole store, which
  with several accounts would throw away everyone else's work.
- **Deduplication is per owner.** Globally, the second person to upload a given book got a
  `skipped` document they could never see, because the only stored copy was someone else's.

## Cloud mode (`RAG_MODE=cloud`)

Built to run this app on Vercel: Pinecone for embeddings + re-ranking, Chroma Cloud for
vectors, Cloudinary for PDFs, MongoDB for accounts *and* everything that used to live in
process memory. The reasoning and step-by-step build log live in
`CLOUD_MIGRATION_PLAN.md` — read that before changing anything here. The short version:

- **`DOCUMENT_STORE` (`local` | `cloudinary`) picks where PDF bytes live.** Cloud mode's
  value follows `RAG_MODE` by default. Cloudinary was chosen over accepting uploads through
  the app's own request body because **Vercel rejects any request over 4.5MB**, and
  `MAX_UPLOAD_MB` defaults to 100 - the browser has to upload the PDF straight to
  Cloudinary and only tell the app the result.
  - `POST /upload/sign` (rate-limited via `ratelimit.UPLOAD_SIGN`) returns a signed
    payload for the browser to POST directly to Cloudinary - the server never sees the raw
    bytes for this leg. The signature's lifespan is enforced by Cloudinary itself (its own
    timestamp window), not by a setting in this app.
  - `POST /upload/complete` is called by the frontend afterwards with the resulting
    `public_id`. It re-checks ownership (`public_id_belongs_to`, see Isolation above),
    fetches the bytes back (`cloudinary_store.fetch_bytes`), runs them through the SAME
    validation local uploads get (`uploads.validate_uploaded_bytes` - magic-byte check,
    per-account storage cap), registers the document (`cloud_documents.register`), and
    kicks off ingestion. **Nothing is ever indexed from a URL without this validation
    step** - skipping it would let an arbitrary file type in disguised as a PDF, which is
    exactly the hole local mode's `uploads.py` docstring says is closed.
  - `POST /upload` (the local-mode multipart endpoint) returns 400 immediately when
    `DOCUMENT_STORE=cloudinary` - it is not a fallback path in cloud mode, it is disabled.
  - Deleting a document in cloud mode calls `cloudinary_store.destroy()` in addition to the
    usual manifest/vector cleanup, mirroring local mode's "delete the PDF too" rule.
- **`STATE_STORE` (`memory` | `mongo`) picks where job/rate-limit/cache/manifest state
  lives.** Cloud mode's value follows `RAG_MODE` by default, and **must** be `mongo` for any
  serverless deployment: a cold start gets a brand-new empty process every time, so anything
  that only lives in a module-level dict silently resets on every single request.
  - `src/core/ratelimit.py`, `src/services/answer_cache.py`, and `src/services/manifest.py`
    each kept their original in-memory implementation *and* added a Mongo-backed one side
    by side, dispatching on `STATE_STORE` at the top of every public function. The
    in-memory code path is untouched by this - it is still exactly what local mode runs.
  - **The Mongo-backed state code uses a plain, synchronous `pymongo.MongoClient`**
    (`database.sync_collection()`, new), not the `motor` async client the rest of this app
    (accounts) uses. This is a deliberate exception to "everything Mongo goes through
    `database._guard()`" - call sites for rate limiting and the ingestion job span both the
    local background thread (no event loop to await into) and async request handlers, and a
    full async refactor of `ingestion.py` wasn't worth doing to avoid one synchronous
    driver. The tradeoff is a bit of added latency per Mongo round-trip, not a correctness
    issue, and it's consistent with this app's documented single-worker, non-concurrent-load
    design (see Deployment invariants).
- **Ingestion has two job models, chosen the same way.** Local mode's threading-based
  background job (`start_job`/`job_status`, unchanged) only works because the process stays
  alive between requests. A Vercel function is killed the instant it responds - there is no
  "meanwhile, in the background." Cloud mode instead does **one file's worth of work per
  request**:
  - `POST /ingest` (or `/upload/complete`, for the single-file case) computes the queue of
    work and writes a job document to the `ingestion_jobs` Mongo collection, one per user.
  - `POST /ingest/continue` embeds at most `INGEST_CHUNKS_PER_REQUEST` chunks (default 400)
    of the file at the head of the queue and returns the updated status, including whether
    more work remains (`done`). It is a bounded SLICE, not a whole file: a serverless
    function is killed at `maxDuration`, and a 636-page book takes about two minutes, so
    "one file per request" meant a large upload timed out, lost its partial work, and was
    retried from scratch forever - one doomed invocation and one wasted batch of embedding
    tokens per poll. `ingest_one()` takes `start_chunk`/`max_chunks` and returns status
    `"partial"` with `next_chunk`; the job document keeps that offset on the queued item so
    the next call resumes where this one stopped. Two invariants make it safe:
    `delete_source()` runs only on the first slice (`start_chunk == 0`), or a resumed call
    would wipe everything before it; and `add_chunks(..., index_offset=start_chunk)` keeps
    `chunk_index` contiguous across slices, without which each slice would renumber from 0
    and neighbour expansion - which addresses chunks BY that index - would fetch the wrong
    passages. Both have dedicated tests that fail if either is removed. Re-extracting the
    PDF on every slice is deliberate: chunking is deterministic, so slice N sees exactly the
    list slice N-1 saw, and a few seconds of extraction is cheap next to embedding. The frontend's polling loop
    (`pollUntilDone()` in `script.js`) calls this in a loop instead of `GET /ingest/status`
    when `document_store === "cloudinary"` (surfaced by `/stats`).
  - `main.py`'s `lifespan` skips the startup background scan and the embedding warm-up
    thread entirely when `IS_CLOUD` is true - there is no local embedding model to warm up
    in cloud mode (Pinecone does that work), and no long-lived process to kick a scan off
    in.
  - Both job models are built on the SAME planning function, `ingestion._plan_ingest()` -
    see the "result-ordering" note under Conventions below for why that had to be one
    ordered list rather than two.
- **File discovery** (`ingestion._pdf_filenames`) and **fingerprinting**
  (`ingestion._fingerprint_any`) dispatch on `DOCUMENT_STORE` the same way retrieval
  dispatches on `VECTOR_STORE`: cloud mode reads the Mongo registry
  (`cloud_documents.list_for_user`/`list_all`) instead of walking `DATA_DIR`, and reads a
  cached sha256/size off that registry instead of re-hashing. Extraction
  (`ingestion._extract_pages_any`) fetches the PDF bytes by URL into a temp file
  (`tempfile.mkstemp`, deleted in a `finally`) rather than opening a local path.
- **What is genuinely identical between modes**: chunking, embedding call shape (through the
  `EMBEDDINGS_PROVIDER` switch), retrieval, re-ranking, the LLM call, and every isolation
  rule above. Cloud mode is a different set of backends behind the same pipeline, not a
  parallel implementation of it.
- **Not yet verified against live credentials.** Everything above was built and tested
  against `tests/test_pipeline_offline.py`'s fakes (a `FakeSyncCollection` standing in for
  pymongo, `database.set_sync_collection()` as the injection point - the same pattern
  `database.set_users_collection()` already used for accounts). No live Cloudinary,
  Pinecone, Chroma Cloud, or MongoDB Atlas call has been made from this code yet. Do one
  manual end-to-end smoke test with real keys (sign up, upload, watch `/ingest/continue`
  drain the queue, ask a question) before trusting a Vercel deployment.

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
| Cloudinary sign requests | `ratelimit.UPLOAD_SIGN` | a stolen/replayed signature becomes an unbounded direct-to-Cloudinary upload channel (cloud mode) |
| Questions per account | `ratelimit.CHAT` | unbounded Groq spend |
| Per-user storage | `MAX_USER_STORAGE_MB` | one account fills the disk (local) or the Cloudinary quota (cloud) and stops the app for everyone |
| Job result list | `MAX_JOB_RESULTS` | unbounded memory (local) or an unbounded Mongo document (cloud) on a long-lived job |

`src/core/ratelimit.py` and `src/services/answer_cache.py` hold state **in process** when
`STATE_STORE=memory` (local mode). That is correct only because the app is single-worker
(below); with N workers every limit is effectively N times larger. In cloud mode
(`STATE_STORE=mongo`) this limitation doesn't apply the same way - state is shared through
Mongo - but the *rate limit values themselves* still assume roughly this app's traffic
shape; revisit them if cloud usage looks different from local.

## Deployment invariants

- **Local mode: ONE uvicorn worker. Not negotiable.** The ingestion job, the BM25 cache, the
  rate limiter and the answer cache are all in-process when `STATE_STORE=memory`, and
  ChromaDB's persistent disk client is single-process. With `--workers 4`: progress bars
  hang (a random worker answers `/ingest/status`), BM25 goes stale in three workers out of
  four, rate limits quadruple, and two processes write the same SQLite file. To scale
  local mode, move the job to a queue and the vectors to a server-based store first - or
  just run cloud mode, which was built to solve exactly this.
- **Cloud mode has the opposite constraint: no persistent process at all.** Every Vercel
  function invocation is a fresh, short-lived process - nothing may be assumed to survive
  between requests. That's why `STATE_STORE=mongo` is mandatory there and why ingestion is a
  per-request slice (`POST /ingest/continue`) instead of a background thread. BM25 still
  rebuilds per-process (per cold start) in cloud mode too - see the "per-process, in memory"
  known limitation below, now true in both modes for different reasons.
- **`/health` is liveness, `/ready` is readiness.** `/health` answers as soon as the process
  is up - which is before the embedding model has loaded and regardless of MongoDB.
  Anything that routes traffic must probe `/ready`, which is 503 until both are true. In
  cloud mode there is no embedding model to load, so this resolves faster there.
- **`data/` is the only copy of user documents in local mode.** `storage/` is derived and
  rebuildable; `data/` is not, and neither is MongoDB. `scripts/backup.py` covers both, and
  `--verify` exists because a backup nobody has read back is a hope, not a backup. **In
  cloud mode, Cloudinary is the only copy of PDF bytes** and the `cloud_documents` Mongo
  collection is the only record of what exists and who owns it - back up both if you rely
  on cloud mode in production; `scripts/backup.py` does not cover Cloudinary/cloud-mode
  Mongo state today.
- **Models are baked into the Docker image and `HF_HUB_OFFLINE=1` is set** for local mode,
  so a running container never depends on Hugging Face being up, and a model repository
  cannot change under a pinned name. Cloud mode has no local model to bake in - embeddings
  and re-ranking are HTTP calls to Pinecone instead (see `requirements-cloud.txt`, which
  drops `torch`/`sentence-transformers` entirely - they don't fit a serverless bundle and
  aren't needed).

## Conventions

- **Every module imports settings from `src.core.config`**, never `os.getenv()` directly.
  New setting → add it there with a sensible default, and to `.env.example`. Cloud-mode
  settings (`DOCUMENT_STORE`, `STATE_STORE`, `CLOUDINARY_*`) follow the same
  `_mode_default()` pattern as the existing `EMBEDDINGS_PROVIDER`/`VECTOR_STORE` switches -
  blank means "follow `RAG_MODE`", set means "override it."
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
  readable — that's the point of this project. This is also why `cloudinary_store.py` talks
  to Cloudinary's REST API directly with `httpx` and a hand-rolled SHA-1 signature instead
  of adding the `cloudinary` SDK as a dependency.
- **`src/services/ingestion.py` is the single entry point for adding documents.** Called
  from the CLI, from the FastAPI lifespan startup (local mode only), from `POST /ingest`,
  from `POST /upload` (local mode) and from `POST /upload/complete` (cloud mode). All of
  these go through `ingest_data_folder()` (local) or the cloud job builder (`start_job`
  cloud branch / `continue_job`), which both fingerprint files by SHA-256 (or, in cloud
  mode, read a cached fingerprint off the `cloud_documents` registry) and skip unchanged
  ones. **Uploads are not a second pipeline**: local's `/upload` writes the file into
  `DATA_DIR` and cloud's `/upload/complete` registers it in Mongo, then both run the
  ordinary job. Nothing is ever embedded straight out of a request body.
- **Ingestion never blocks a request.** In local mode, `start_job()` runs it on a background
  thread; `/ingest`, `/reset` and `/upload` return 202 and the client polls
  `/ingest/status`. In cloud mode, `start_job()` only plans the work and returns
  immediately; the actual embedding happens one file at a time inside `POST
  /ingest/continue`, which the client calls in a loop.
- **`start_job()` called during a run flags a re-scan; it never starts a second thread
  (local) or a second job document (cloud).** The running job listed the folder/registry
  when it started, so a PDF uploaded mid-run was invisible to it and sat unindexed until
  someone pressed sync. Local's `_run()` drains `_consume_rescan_request()` after the scan
  finishes; cloud's job document carries the same `rescan_requested`/`rescan_force` fields.
- **The planning step (`ingestion._plan_ingest`) returns ONE ordered list, not two.** It
  used to return a `(decided, pending)` tuple, with callers appending all decided results
  first and all pending (to-be-ingested) results second. That reorders the results relative
  to the original filename-scan order whenever a pending file happens to sort before a
  decided one (ASCII: space `0x20` sorts below period `0x2e`, so `"book (2).pdf"` sorts
  before `"book.pdf"`) - which silently broke the "byte-identical content is skipped as a
  duplicate" guarantee's test coverage. `_plan_ingest()` now returns entries in scan order,
  each either a terminal result or a `{"pending": True, ...}` marker, and both
  `ingest_data_folder()` (local) and `_start_job_cloud()` (cloud) consume that single list.
  Don't reintroduce the two-list split.
- **A document's manifest entry is written only after its last chunk is stored**, and the
  UI's document list is built from the manifest. That is what makes "only fully-processed
  documents appear in the library" true; don't write the entry earlier to make progress
  reporting easier. This holds in both the JSON-sidecar (local) and Mongo-backed (cloud)
  manifest implementations.
- **`ingest_one()` calls `delete_source()` for `new` files too, not just `changed` ones.**
  A job killed mid-file (locally: Ctrl+C or a reload; in cloud mode: the function simply
  being killed between slices) leaves stored chunks with no manifest entry, so the retry
  treats the file as new and re-embeds it under a fresh `run_id` — without the delete,
  every interrupted attempt piles up another duplicate copy. This is exactly what makes
  cloud mode's per-request slicing safe to interrupt at any point.
- **`src/services/retrieval.py` owns the query path.** Endpoints call `retrieve()`; they don't talk
  to `vectorstore`/`bm25`/`reranker` directly. Every stage is switchable through both
  config and keyword arguments — that's what lets `eval/run_eval.py` A/B them, and what lets
  a single retrieval code path serve both `VECTOR_STORE=chroma` and `VECTOR_STORE=pinecone`
  unchanged.
- **An empty retrieval result is a real answer**, not an error: it means nothing cleared
  the relevance floor, and `ml.llm.generate_answer()` returns the "not in these documents"
  message *before* checking for an API key, since that answer needs no LLM call.
- **Heavy local models are LAZY singletons — never module-level.** The embedding model, the
  cross-encoder and the BM25 index all load on first use, behind a lock (local mode; cloud
  mode's embedding/rerank are HTTP calls with no local model to load). This is not a
  micro-optimisation: uvicorn imports the app *before* it binds the port, so loading the
  embedding model at import kept the port closed for ~18s and the browser answered
  `ERR_CONNECTION_REFUSED`. `main.py`'s lifespan starts `embeddings.warm_up()` on a thread
  so the wait happens in the background, with the status dot reporting it - but only in
  local mode; `IS_CLOUD` skips this block entirely (see "Cloud mode" above). Never move a
  model load to import time, and never into a per-request path either.
- **Anything that changes stored chunks must call `_invalidate_keyword_index()`** (already
  wired into `add_chunks`, `delete_source`, `reset_collection`) — the BM25 index caches the
  corpus in memory and would otherwise go stale. True in both modes; cloud mode's BM25 index
  is just rebuilt more often, on every cold start.
- **One bad document must never stop the others.** `ingest_data_folder()`/cloud's job loop
  catch per file and record `{"status": "failed", "error": ...}`; anything that adds a new
  failure mode inside the per-file path must keep that contract. A corrupt PDF used to abort
  the whole job, leaving every file after it silently un-indexed.
- **`data/` is scanned recursively in local mode** and documents are keyed by their
  POSIX-style relative path (`textbooks/norvig.pdf`), so the same identity works on Windows
  and Linux. **Cloud mode uses the identical key shape** (`users/<id>/name.pdf`) as the
  Mongo document `_id` in `cloud_documents`, so `owner_from_path()` and every isolation
  filter work unchanged across both modes - this was a deliberate design choice, not an
  accident.
- **The manifest is written atomically** (temp file + `os.replace`, local mode) under a
  lock. A plain write leaves a window where a reader sees half a file, decides the store is
  empty, and re-embeds the entire corpus. Cloud mode's Mongo-backed manifest gets the
  equivalent atomicity for free from Mongo's per-document write guarantees.
- **`src/services/uploads.py` is the only place that turns request bytes into a file
  (local) or validates fetched bytes before registering them (cloud).**
  `validate_uploaded_bytes()` is the cloud-mode entry point, sharing the same checks
  `save_pdf()` enforces locally: the content must actually start with `%PDF-` regardless of
  extension or content type, and the size cap is enforced against the actual byte count. For
  local uploads specifically: the filename is reduced to a bare, scrubbed segment (so
  `../../.env.pdf` and `C:\Users\me\book.pdf` cannot escape `DATA_DIR`), the size cap is
  enforced *while streaming* with the partial file deleted, a colliding name is renamed
  rather than overwritten, and the write is atomic (temp file + `os.replace`) so the
  ingestion thread can never read a half-uploaded PDF. Any new intake path must reuse this
  module rather than reimplementing part of it.
- **Deleting a document must delete the PDF too** (`DELETE /documents/{name}`). Removing
  only the vectors leaves the file behind (in `data/` locally, in Cloudinary in cloud mode),
  and the next ingest/startup faithfully re-indexes it. Cloud mode's delete calls
  `cloudinary_store.destroy()` in addition to the manifest/vector cleanup both modes share.
- **HTML is served `no-cache`; CSS and JS are cache-busted with `?v=N` in `index.html`.**
  Chrome served a cached `index.html` against a freshly updated `style.css` once and the new
  markup rendered completely unstyled. Bump the `?v=` number whenever you change either
  static asset (currently `?v=22`).
- **Auth code: `pwdlib` with Argon2id, never `passlib`.** passlib is unmaintained and
  breaks against recent bcrypt releases. `PasswordHash.recommended()` also gives hash
  migration for free.
- **`JWT_SECRET` is generated once and persisted to `.env`.** A key regenerated per restart
  signs everyone out on every reload; a hard-coded default lets anyone mint a token. The
  decode call pins the algorithm - accepting the token header's `alg` is the classic JWT
  forgery. In cloud mode, generate this once yourself and set it as a Vercel env var - there
  is no local `.env` file for the running process to persist it back into.
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
- **Only accounts live in MongoDB via the async `motor` client (`database.py`'s existing
  functions).** Documents, vectors and the manifest stay in Chroma/Pinecone/Cloudinary and,
  in local mode, on disk. **Cloud-mode state (rate limits, answer cache, ingestion jobs,
  manifest) also lives in MongoDB, but through a separate synchronous `pymongo.MongoClient`**
  (`database.sync_collection()`/`get_sync_client()`) - see "Cloud mode" above for why sync,
  not async. Splitting one document's identity across two databases means a delete can
  half-succeed, so this sync/async split is about the *driver*, not about splitting data
  across two different databases - it's still all one MongoDB instance/cluster.
- `data/` and `tests/fixtures/` are strictly separate: `data/` is real user documents (local
  mode only); `tests/fixtures/` is synthetic PDFs from `make_test_pdf.py`. Never point the
  fixture generator's default output at `data/`.

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
- **`documentStore` (module var, set from `/stats`'s `document_store` field by
  `refreshSources()`) is what the frontend uses to branch between the two upload/ingest
  flows.** `uploadFiles(fileList)` dispatches to `uploadFilesCloud(files)` when
  `documentStore === "cloudinary"` before falling through to the original local-mode
  direct-upload logic; `uploadFilesCloud` does the sign → direct-POST-to-Cloudinary →
  complete round trip per file (`signCloudUpload()`, `sendToCloudinary()` via XHR for
  progress events, then `POST /upload/complete`), sequentially. Similarly,
  `pollUntilDone()` calls `POST /ingest/continue` in a loop in cloud mode instead of
  polling `GET /ingest/status` - the endpoint IS the work in cloud mode, not just a status
  read.
- **Two progress phases, from two sources (local mode).** The transfer has real byte counts
  (XHR `upload.progress` — `fetch()` has no equivalent, which is why it is XHR); indexing is
  polled from `/ingest/status`, which reports `stage`, `chunks_done` and `chunks_total` so a
  900-page book is not a bar frozen at 0% for five minutes. Cloud mode's `sendToCloudinary`
  reuses the same XHR-progress approach for the direct-to-Cloudinary leg.
- **`MAX_UPLOAD_MB` comes from `/stats`,** not a hard-coded copy — the client-side size
  check must not drift from the server's real limit.
- **Every request goes through `authFetch()`**, which attaches the bearer token and treats
  any 401 as "the session is over". The exceptions are the XHR uploads (fetch has no
  upload-progress events) - local mode's direct multipart XHR and cloud mode's
  `sendToCloudinary` - which set headers by hand and handle 401 themselves; if you add a
  request, use `authFetch`.
- **The SSE stream carries the token** only because it is read with `fetch()`; `EventSource`
  cannot set headers. Don't "simplify" it to `EventSource`.
- **`resetAppState()` runs on every sign-in and sign-out.** Without it the previous
  account's chat transcript and document rows sit on screen until the first refresh lands,
  which looks exactly like a leak even though the server sent none of it.
- **Ownership of a JOB is `scope`, not `current_file`.** `job_status(user_id)` reveals
  progress when the run was started for that user, or when the file in flight is theirs.
  Deciding from `current_file` alone was a bug: it is None at the start of a run, between
  files and for the whole finished state, so the counts were blanked exactly when the UI
  needed them and the progress bar never moved. This applies to both the local-thread job
  status and the cloud per-user job document.
- **The activity trail (`_state["events"]` locally, a Mongo `events` array in cloud mode) is
  per user and bounded.** It is what the Processing status panel shows - the steps the
  pipeline actually took, tagged with whose document they concern, so another account's
  filenames never appear in it.
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
download) in local mode.

**To run cloud mode locally** (validates Pinecone + Chroma Cloud + Atlas without deploying):
set `RAG_MODE=cloud` (or the individual `EMBEDDINGS_PROVIDER`/`RERANKER_PROVIDER`/
`VECTOR_STORE`/`CHROMA_BACKEND`/`STATE_STORE` overrides) plus the corresponding API
keys/URIs in `.env`, then run `python scripts/run.py` exactly as above. Everything works
except browser uploads if `DOCUMENT_STORE` is also cloud (Cloudinary needs to be configured
too, or hand-copy a PDF into `data/` and run `python scripts/ingest.py` while leaving
`DOCUMENT_STORE=local` to sidestep it). **To deploy to Vercel**: push with `vercel.json` and
`api/index.py` in place, set every env var from `CLOUD_MIGRATION_PLAN.md`'s Step 5 list in
the Vercel dashboard (Production + Preview), and set `RAG_MODE=cloud`.

## Testing and evaluation

Three different questions, three different tools — don't confuse them:

```bash
python tests/test_pipeline_offline.py    # does the plumbing work?  (offline, no live services)
python eval/run_eval.py                  # are the answers any good? (needs an index)
python scripts/check_cloud.py            # do the cloud credentials work? (live, no app needed)
```

`scripts/check_cloud.py` is the pre-deploy gate, and it only checks what lives OUTSIDE the
process, because that is the half the offline suite deliberately fakes. It verifies the Groq
key *and* that `GROQ_MODEL` still exists (Groq retires models on their own schedule, and the
failure otherwise arrives as a 404 on a user's first question), that MongoDB accepts a real
write and not merely a ping (an Atlas user with read-only rights pings happily and then
fails at signup), that the live embedding dimension matches both `PINECONE_EMBED_DIM` and
whatever dimension the existing index/collection was created with, and that a signed
Cloudinary upload actually round-trips — a real upload-then-destroy through
`cloudinary_store`'s own hand-rolled signature, because a wrong signature is rejected with a
generic error that no amount of "the keys are set" will predict. It exits non-zero on any
failure, so it drops into CI or a pre-deploy hook.

Its first section is a pure-configuration pass that needs no network, and it exists because
the most expensive cloud bugs are not bad credentials but settings that are *valid yet wrong
for a serverless host*: `STATE_STORE=memory` resets on every request, `DOCUMENT_STORE=local`
writes PDFs to a filesystem that is thrown away, a blank `JWT_SECRET` signs everyone out on
every cold start. Each of those deploys cleanly and then misbehaves in production. Pinecone
re-ranking is behind `--rerank` because testing it spends one of the free tier's 500 monthly
requests.

`tests/` stubs `sentence_transformers` (both `SentenceTransformer` and `CrossEncoder`) via
`sys.modules`, substitutes an in-memory fake for the Mongo users collection
(`database.set_users_collection`), points `DATA_DIR`/`CHROMA_DIR` at temp folders, and
drives real HTTP endpoints through `TestClient`. Hashing, JWTs, the auth dependency and
every isolation filter are the real code. **Cloud-mode state code is tested the same way**:
`FakeSyncCollection` is an in-memory stand-in for a pymongo `Collection` (supports
`find_one`, `find`, `insert_one`, `update_one` with `$set`/`$setOnInsert`/`$inc`/`$push`
with `$each`/`$slice`, `delete_one`, `delete_many`, `count_documents`, `create_index`),
injected via `database.set_sync_collection(name, fake)` - the sync-client equivalent of
`set_users_collection`. This covers the Mongo-backed rate limiter, answer cache, manifest,
cloud document registry, and the Cloudinary `public_id_belongs_to` ownership check, but it
does **not** prove a live Cloudinary/Pinecone/Chroma Cloud/Atlas call actually succeeds -
see "Cloud mode" above.

**The isolation checks are the ones that matter.** They sign up two accounts, upload a
byte-identical document to each, and assert that neither can reach the other's chunks
through vector search, through BM25, or through neighbour expansion - each verified
separately, because a single end-to-end check passes even when two of the three filters
are missing. Sabotaging any one of them makes a specific check fail; if you change
retrieval, confirm that is still true. It does **not** prove the real models download or that a
live Groq call succeeds — do a manual smoke test with a real `GROQ_API_KEY` before
considering a change done. The same "offline tests pass, live smoke test still pending"
caveat applies doubly to cloud mode - see "Cloud mode" above.

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
  accordingly (bge/e5 want a query-side prefix; MiniLM wants none). Doesn't apply to
  `EMBEDDINGS_PROVIDER=pinecone` (cloud mode) - the provider's own model handles this.
- **Paragraph structure must survive extraction.** `_normalize()` deliberately keeps
  `\n\n`. Collapsing all whitespace (`" ".join(text.split())`) destroys the boundaries the
  chunker packs on and measurably worsened page attribution.
- **BM25 tokenisation keeps technical characters.** `bm25.tokenize()` uses a custom regex,
  not `\w+`, so `A*`, `k-means` and `f1` survive as tokens. A plain `\w+` split turns "A*"
  into "a" — a token that matches everything and ranks nothing.
- **Cross-encoder scores are raw logits, not probabilities.** Compare them against
  `MIN_RERANK_SCORE`; don't display them as confidences or normalise them to 0..1. Cohere's
  API-based reranker scores 0..1 instead and needs its own floor
  (`MIN_RERANK_SCORE_API`) - don't reuse `MIN_RERANK_SCORE` for it.
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
  removed from `data/` (or from the cloud registry) stayed searchable and citable forever.
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
  ingest is still running, because an answer given then may be missing passages. (Cloud
  mode has no warm-up phase - see "Cloud mode" above.)
- **A database outage is a 503 with instructions, not a 500 with a traceback.**
  `database._guard()` converts pymongo's `ServerSelectionTimeoutError`/`ConnectionFailure`
  into `DatabaseUnavailable`, and an exception handler in `main.py` renders it as a 503
  whose message says how to start MongoDB; the sign-in screen shows that text verbatim.
  Only connection-level failures are converted - `DuplicateKeyError` must keep propagating,
  or a taken username stops returning 409. Every new Mongo call goes through `_guard` - the
  async accounts path, that is; the sync cloud-state path (`sync_collection`) does not go
  through `_guard` today and is a candidate for the same treatment if cloud-mode Mongo
  outages turn out to need the same friendly-503 handling.
- **A document's owner comes from its path, never from the request.** An upload's owner is
  the authenticated caller; `owner_from_path()` reads it back off `users/<id>/...`. Trusting
  a `user_id` field in a request body would let anyone claim anyone's documents. In cloud
  mode this extends to the Cloudinary `public_id` - see the isolation note above.
- **Deletes by document name are deliberately NOT owner-scoped** (after the ownership check
  passes). The name is the path, so it can only ever match one user's file - and scoping it
  would strand chunks written before the document had an owner.
- **PyMuPDF is AGPL-3.0**, unlike every other dependency here. Fine for personal/local use;
  comply with AGPL or buy a commercial license before shipping this closed-source.
- **The ingestion planning step must return one ordered list, not decided/pending split
  into two.** See the `_plan_ingest` note under Conventions - splitting them reorders
  results whenever a pending filename sorts before a decided one, which broke the
  byte-identical-duplicate test and would just as easily surprise anyone reading a job's
  `results` array expecting scan order.

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
- **The API is plain HTTP in local mode.** A token on a shared network is sniffable.
  Localhost only, unless you put TLS in front of it. A Vercel deployment gets TLS for free.
- **Signup is open.** Anyone who can reach the page can create an account and upload.
- **MongoDB is a hard dependency in both modes.** Without it nobody can sign in, and in
  cloud mode nothing else works either (state store). The app still starts, logs how to fix
  it, and answers auth requests with a 503 that says the same - it does not pretend to
  work. `GET /api/health/auth` reports whether it is reachable.
- **The BM25 index is per-process and in memory, in both modes.** Multiple local workers
  each build their own; it rebuilds on restart (one O(corpus) read). In cloud mode it
  rebuilds on every cold start instead - a cost, not a correctness problem, per the
  reasoning in `CLOUD_MIGRATION_PLAN.md`.
- **Uploads are unauthenticated... no, wait - uploads ARE authenticated** (every document
  route requires `get_current_user`), but nothing beyond `ratelimit.UPLOAD`/`UPLOAD_SIGN`
  limits how many requests one account can make; on a public address that is still a
  disk-filling (local) or Cloudinary-quota-filling (cloud) vector, just a rate-limited one.
- **Which of two byte-identical copies wins is scan order, not recency.** Upload the same
  book twice and the alphabetically-first name is indexed while the other is `skipped` —
  even if the skipped one was already in the store.
- **Duplicate detection is byte-exact only.** Two PDFs with identical bytes are caught; the
  same book re-exported or re-scanned is not, and will compete with itself in retrieval.
- **`prune_deleted()` only reconciles the manifest against what's currently listed** (disk
  in local mode, the Mongo registry in cloud mode). Chunks orphaned by an interrupted
  ingest (killed between `delete_source` and `add_chunks`) are not detected;
  `python scripts/ingest.py --force` is the repair for local mode - cloud mode has no
  equivalent CLI repair script yet.
- **Follow-up rewriting costs an extra Groq call** per question that has history. Set
  `REWRITE_FOLLOWUPS=false` to trade follow-up quality for latency and tokens.
- **Cloud mode is implemented but not live-tested.** See "Cloud mode" and "Testing and
  evaluation" above - everything has offline test coverage against Mongo/Cloudinary fakes,
  nothing has been run against real Cloudinary/Pinecone/Chroma Cloud/Atlas credentials from
  this code yet. Treat a first real deployment as needing its own smoke test, not as
  "already proven."
- **Cloudinary delivery is PUBLIC, and that is a deliberate, known gap.** Cloudinary
  restricts `raw`/PDF delivery by default; that restriction was lifted in the dashboard so
  `cloudinary_store.fetch_bytes()` - a plain unauthenticated GET on the `secure_url` - could
  read an upload back at `/upload/complete`. The consequence is that **every uploaded PDF is
  readable by anyone holding its URL**, with no session and no ownership check. The URL
  contains a ~20-character random segment so it is not guessable, but this is capability-URL
  security, and the URL is stored in Mongo, appears in logs, and is handled by the browser.
  That sits awkwardly beside the four isolation rules above, which the rest of the app
  enforces carefully. It is acceptable for a portfolio deployment and NOT acceptable for real
  user documents. The fix is to re-restrict delivery in the Cloudinary dashboard and have
  `fetch_bytes()` authenticate - either a signed delivery URL or an upload with
  `type=authenticated` - and `scripts/diagnose_cloudinary.py` exists to find out which of
  those a given account actually accepts, since a wrong signature is rejected with a generic
  error that no amount of reading the docs will predict.
- **`scripts/backup.py` does not cover cloud-mode state.** It backs up `data/` and MongoDB
  (which, in cloud mode, includes the `cloud_documents` registry and Mongo-backed state
  collections) but has no Cloudinary-specific backup/restore path - the PDFs themselves
  would need their own backup strategy in a production cloud deployment.

## Before you commit

- Never commit `.env` (real API key) or documents in `data/` — both gitignored.
- Run `tests/test_pipeline_offline.py`; all checks must pass.
- If you touched anything in the retrieval path, run `eval/run_eval.py` before and after
  and put the numbers in the commit message or PLAN.md.
- If you touched anything cloud-mode-specific (`DOCUMENT_STORE`/`STATE_STORE` switches,
  `cloudinary_store.py`, `cloud_documents.py`, the Mongo-backed branches of `ratelimit.py`/
  `answer_cache.py`/`manifest.py`, or either ingestion job model), extend
  `tests/test_pipeline_offline.py`'s `FakeSyncCollection`-based cloud-state tests to cover
  it, and note in the commit message that it's offline-tested only, not verified against
  live Cloudinary/Pinecone/Atlas - see "Cloud mode" above for why that caveat matters.
