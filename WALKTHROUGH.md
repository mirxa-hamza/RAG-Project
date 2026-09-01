# How this project fits together

A guided tour of the codebase: what each folder is for, what happens on each request, and
where to look when you want to change something. Written to be read top to bottom once,
then dipped into.

If you only want one answer: **there is no separate login page** — see the first section.

---

## 1. Where the login and signup screen lives

There is no `login.html`, and no separate signup page. Sign in and sign up are the **same
form**, an overlay inside the single HTML file the app has. It covers the app and hides it
until a token is confirmed; the "Create an account" link only relabels the fields.

| What | File | Where exactly |
|---|---|---|
| The form itself | `src/static/index.html` | `<div class="auth" id="auth">` — **line 44**. Title, username + password fields, error line, and the switch link. |
| How it looks | `src/static/style.css` | `.auth`, `.auth__card` — **line 643**. A fixed overlay at `z-index: 70`, using the same design tokens as the app. |
| What it does | `src/static/script.js` | `authForm` submit handler — **line 1040**. Also `setAuthMode()` (960) to flip between the two modes, `startSession()` (1020), `endSession()` (1032), and `startup()` (1383), which decides which screen you see on load. |
| The endpoints it calls | `src/api/auth.py` | `/api/signup` (51), `/api/login` (95), `/api/me` (123), plus password change, sign-out-everywhere, delete account. |
| Passwords and tokens | `src/services/security.py` | Argon2id hashing (pwdlib), JWT encode/decode. |
| Where accounts are stored | `src/services/database.py` | MongoDB, database `rag_app`, collection `users`. |

**Why one file rather than two pages?** The whole front end is a single page with no router
and no build step. Showing the app is `el.app.hidden = false`; showing the login screen is
the reverse. That is also why the app's markup exists in the DOM but stays hidden until
`/api/me` confirms the stored token — a token in `localStorage` is not proof of a session.

---

## 2. The shape of it, in one paragraph

One Python process does everything: it serves the HTML/CSS/JS, exposes the API, and runs a
background thread that turns PDFs into searchable passages. MongoDB holds user accounts and
nothing else. Nothing is ever indexed straight from a request body — an upload is written to
disk first and the ordinary indexer picks it up, so a file that arrives by upload and a file
you copy into `data/` by hand travel exactly the same path.

```
browser (static/)  →  src/api/  →  src/services/ + src/ml/  →  MongoDB · Chroma · data/
   auth overlay       thin           the actual pipeline          accounts, vectors, PDFs
   sidebar, chat      handlers
```

---

## 3. Folder map

Everything under `src/` is the running application; everything else is tooling around it.

```
src/
├── main.py             assembles the app: router, middleware, lifespan, serves the UI at /
├── api/                HTTP layer — thin handlers only
│   ├── auth.py         signup, login, me, password change, delete account
│   ├── chat.py         /chat and /chat/stream (server-sent events)
│   ├── documents.py    upload, ingest status, stats, delete, rebuild
│   ├── system.py       /health, /ready, /info
│   └── deps.py         get_current_user — the single auth gate every route uses
├── services/           the pipeline and its state
│   ├── pdf.py          text extraction (PyMuPDF) + chunking into passages
│   ├── ingestion.py    the ONLY way documents get in; background job + progress + events
│   ├── uploads.py      validates bytes from the browser, writes them safely to disk
│   ├── vectorstore.py  ChromaDB: add, query, neighbours, delete
│   ├── bm25.py         keyword search index, one per user, LRU-cached
│   ├── retrieval.py    fuse vector + keyword, re-rank, apply floor, expand neighbours
│   ├── manifest.py     small JSON record of what is indexed, with fingerprints
│   ├── database.py     MongoDB: accounts and the audit trail
│   ├── security.py     password hashing + JWTs
│   ├── ownership.py    who owns which document; adoption; account cleanup
│   └── answer_cache.py repeated questions skip the whole pipeline
├── ml/                 model wrappers, all lazily loaded
│   ├── embeddings.py   text → vectors (bge-small-en-v1.5)
│   ├── reranker.py     cross-encoder that decides what actually reaches the LLM
│   └── llm.py          the prompt, and the Groq call (plain + streaming)
├── core/               imported by everything, imports nothing of ours
│   ├── config.py       every setting, read from .env, in one place
│   ├── logging.py      text or JSON logs, request ids, stage timings
│   └── ratelimit.py    login / signup / upload / chat limits
├── models/schemas.py   request and response shapes (Pydantic), and their limits
└── static/             the entire front end — three files
    ├── index.html      auth overlay + app markup + upload/account/status dialogs
    ├── style.css       design tokens first, then components
    └── script.js       session, uploads, chat streaming, panels

scripts/                things you run by hand
├── run.py              starts the server, opens the browser once it answers
├── ingest.py           index the data folder from the terminal (--force, --status)
├── backup.py           archive data/ + MongoDB; --verify reads a backup back
├── verify_index.py     find and repair index drift (orphan / missing chunks)
└── draft_golden.py     draft evaluation questions from your own PDFs

data/                   the PDFs — THE ONLY COPY. Back this up.
└── users/<user_id>/    one folder per account; the path is what proves ownership
storage/chroma_db/      the index + manifest. Derived: rebuildable from data/
tests/                  220 offline checks; no API key, no MongoDB needed
eval/                   measures answer quality, not plumbing
```

---

## 4. Flow: signing in

1. **Page loads** — `script.js · startup()`. A stored token is not proof of a session, so it
   calls `/api/me` before showing anything.
2. **You submit** — `authForm` handler POSTs to `/api/login` or `/api/signup`. These are the
   only two calls in the app that carry no token.
3. **Rate limit** — `core/ratelimit.py`, keyed by address **and** by username: by address
   alone a botnet spreads out; by username alone an attacker can lock out a real user.
4. **Verify** — `services/security.py`, Argon2id. A missing user still pays for a hash, so
   timing cannot reveal which accounts exist.
5. **Token** — a JWT carrying your id, username and `token_version` (the field that makes it
   revocable).
6. **App appears** — `startSession()` stores the token, wipes the previous user's state, and
   fetches your documents.

From then on every request goes through `authFetch()`, which attaches the token and treats
any 401 as "the session is over". The server-side mirror is `get_current_user` in
`api/deps.py`: every route that touches documents depends on it.

---

## 5. Flow: a PDF becomes searchable

This is exactly what the **Processing status** panel narrates.

1. **Upload** (`api/documents.py`) — streamed to disk in 1 MB blocks. The filename is
   scrubbed to a bare name, the contents must really begin with `%PDF-`, and both the size
   cap and your storage quota are enforced *while* writing.
2. **Written** (`services/uploads.py`) — lands at `data/users/<your id>/name.pdf`. That path
   is the ownership record; nothing trusts a `user_id` sent in a request.
3. **Job starts** (`services/ingestion.py`) — a background thread scoped to you. The HTTP
   response returns immediately; the browser polls `/ingest/status`.
4. **Read pages** (`services/pdf.py`) — PyMuPDF extracts text and keeps paragraph breaks,
   which the chunker packs on.
5. **Split** — ~300 words per passage with overlap, then split again by *tokens* so nothing
   is silently truncated at embedding time.
6. **Embed and store** (`ml/embeddings.py` → `vectorstore.py`) — each passage becomes a
   vector, stored with `user_id` in its metadata. This is the slow part: minutes for a
   900-page book on CPU.
7. **Recorded** (`services/manifest.py`) — written *after* the last passage is stored, which
   is why a document appears in your library only when it is genuinely ready.

Files are fingerprinted with SHA-256, so re-running the indexer skips anything unchanged.

---

## 6. Flow: asking a question

Retrieval is deliberately not a single vector search. Each stage exists because the one
before it is wrong in a different way.

1. **Rewrite** (`ml/llm.py · rewrite_question`) — "what about the second one?" carries
   nothing searchable, so with history it becomes a standalone question first.
2. **Vector search** (`vectorstore.query_chunks`) — strong on paraphrase, weak on exact
   terms. Filtered to your documents.
3. **Keyword search** (`services/bm25.py`) — strong on `A*`, `k-means`, notation, exactly
   where vectors fail.
4. **Fuse** (`retrieval.fuse`) — reciprocal rank fusion, so passages both methods rank
   highly win, without comparing incomparable scores.
5. **Re-rank** (`ml/reranker.py`) — a cross-encoder reads question and passage together;
   anything below the relevance floor is dropped.
6. **Expand** (`retrieval._expand_neighbors`) — pulls the passages either side of each hit so
   the model reads continuous prose.
7. **Answer** (`ml/llm.py · stream_answer`) — passages are fenced as quoted data and sent to
   Groq; tokens stream back over SSE.

**If nothing clears the floor, no LLM call happens at all** — the app says the documents
don't cover it. That refusal is the reason answers stay grounded.

---

## 7. The four places data lives

| Store | Where | What | Backup? |
|---|---|---|---|
| Accounts | MongoDB `rag_app.users` | usernames, Argon2 hashes, `token_version`, audit trail | **Yes** |
| The PDFs | `data/users/<user_id>/` | the files themselves — the only copy anywhere | **Yes** |
| The index | `storage/chroma_db/` | every passage as a vector, tagged with its owner | Rebuildable |
| The manifest | `storage/chroma_db/manifest.json` | a few hundred bytes: what is indexed, its fingerprint, its owner | Rebuildable |

`scripts/backup.py` covers the two that matter, and `--verify` reads a backup back — a
backup nobody has restored is a hope, not a backup.

---

## 8. The rules that keep it coherent

**Dependencies point one way.**

```
api/  →  services/ + ml/  →  core/
```

`api/` handlers validate, authorise, call a service and shape a reply. `services/` does the
work and never imports from `api/`. `core/` imports nothing of ours. If you find yourself
importing `api` from a service, the logic belongs in the service.

**Isolation is enforced three times, not once.** Three stages can reach stored text and each
filters independently: the vector query, the keyword ranking, and neighbour expansion (which
fetches by position and would otherwise bypass search entirely). `retrieval.retrieve()` then
re-checks ownership on the way out. The tests fail individually if any one is removed.

**One uvicorn worker, always.** The background job, the keyword cache, the rate limiter and
the answer cache all live in the process, and ChromaDB's client is single-process.
`--workers 4` silently breaks progress reporting and multiplies every rate limit.

**Models load lazily.** Uvicorn imports the app *before* it opens the port, so loading a
model at import time keeps the port shut for ~18 seconds and the browser says "connection
refused". They load in a warm-up thread instead, behind the loading screen.

---

## 9. Where to change what

| You want to… | Open |
|---|---|
| Change wording, colours or layout | `src/static/style.css`, `index.html` — bump `?v=` so browsers pick it up |
| Change how the sign-in screen behaves | `src/static/script.js`, the "Sign in / sign up" block |
| Add or change an endpoint | `src/api/`, then a service for the real work |
| Tune retrieval (top_k, floors, hybrid on/off) | `src/core/config.py`, or `.env` without touching code |
| Change how PDFs are split | `src/services/pdf.py`, then `python scripts/ingest.py --force` |
| Change the prompt or the model | `src/ml/llm.py`; `GROQ_MODEL` in `.env` |
| Add a setting | `src/core/config.py` **and** `.env.example` |
| Understand a past bug before repeating it | `CLAUDE.md`, "Known gotchas" |
| See what still needs doing | `AUDIT.md` §8 |

### Reading order, if you want the whole thing

```
1. src/main.py              how the app is assembled (162 lines)
2. src/core/config.py       every knob that exists
3. src/api/*.py             the surface: what it can be asked to do
4. src/services/pdf.py      where a PDF becomes text
5. src/services/retrieval.py the interesting part: how passages are chosen
6. src/ml/llm.py            the prompt, and why it refuses to guess
7. tests/test_pipeline_offline.py   every rule above, as an executable check
```

Line numbers in this document refer to the files as they stand today and will drift as you
edit. The file names won't.
