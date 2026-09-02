# Deploying to Vercel

Read this once before you start. The short version: the app is not deployable yet, four
pieces of the code still have to change, and there are four accounts to create. The account
work does not depend on the code work, so you can do Part A today.

`RAG_MODE` decides everything. `RAG_MODE=local` is the setup you already run, unchanged.
`RAG_MODE=cloud` is the one Vercel needs. Nothing about local development goes away.

| Piece | `local` | `cloud` |
|---|---|---|
| Embeddings | sentence-transformers on your CPU | Pinecone |
| Re-ranker | cross-encoder on your CPU | Pinecone Rerank |
| Vectors | `storage/chroma_db/` | Pinecone |
| PDFs | `data/users/<id>/` | Cloudinary |
| Accounts | MongoDB on localhost | MongoDB Atlas |
| Job state, rate limits, cache | in this process's memory | MongoDB |

---

## Part A — what you do (about an hour, all free, no card)

### 1. MongoDB Atlas — accounts and server state

1. <https://cloud.mongodb.com> → create a free **M0** cluster.
2. **Database Access** → add a user with a password. Write it down; you cannot read it back.
3. **Network Access** → add `0.0.0.0/0`. Vercel functions have no fixed IP, so there is no
   narrower rule to write. The database is still protected by the password and TLS.
4. **Connect** → *Drivers* → copy the `mongodb+srv://...` string.

That string is `MONGO_URI`. It contains a password: it is a secret, and it never goes in
git.

### 2. Pinecone — vectors, embeddings and re-ranking

<https://app.pinecone.io> → sign up → **Starter** plan. Free, no credit card. Copy the API
key; that is `PINECONE_API_KEY`, and it is the only Pinecone value you must set. You do not
create the index by hand — the app creates it on first use, because a serverless host has
no shell to run a setup step from.

What the free plan gives you, and what it buys in this app:

| Quota | Roughly |
|---|---|
| 2 GB storage | far more chunks than a personal corpus will ever hold |
| 5M embedding tokens/month | about four 300-page books indexed |
| 500 rerank requests/month | 500 questions asked |

The rerank quota is the tightest thing in the system: one request per question. When it
runs out, re-ranking fails *open* — answers keep coming, ranked by fusion alone, slightly
worse. Nothing breaks and nothing charges you.

One setting to leave alone unless you know why: `PINECONE_EMBED_DIM=1024` matches
`llama-text-embed-v2`. An index's dimension is fixed when it is created, so changing the
model without changing this (and deleting the index) makes every upsert fail.

### 3. Cloudinary — the PDFs

You already have this account. From the dashboard: **cloud name**, **API key**, **API
secret** → `CLOUDINARY_CLOUD_NAME`, `CLOUDINARY_API_KEY`, `CLOUDINARY_API_SECRET`.

### 4. Groq — the answer model

The key you already use. Nothing changes.

### 5. Vercel

<https://vercel.com> → sign up with GitHub. Do not import the project yet; the repo is not
ready (Part B).

---

## Part B — what still has to change in the code

Two of these are done and already in your folder. Four are not, and the app will not run on
Vercel until they are. This is the honest list, not a formality.

**Done**

1. **Provider switch.** `EMBEDDINGS_PROVIDER` / `RERANKER_PROVIDER` pick between the local
   models and a hosted one (Pinecone, or Cohere/Jina if you ever want them). A hosted
   re-ranker scores 0..1 while the cross-encoder emits unbounded logits, so the relevance
   floor is now chosen per provider — reusing `-6.0` against a 0..1 score would have kept
   every candidate and quietly disabled the "not in these documents" answer.
2. **Pinecone vector store.** `VECTOR_STORE=pinecone` swaps the whole store. Chunk ids are
   derived from the document name, so re-indexing overwrites instead of duplicating and a
   neighbouring chunk is a plain fetch rather than a second search. The index is opened —
   and created if missing — on first use, not at import, so a missing key cannot stop the
   process before it serves the login page.

**Still to do**

3. **Cloudinary uploads.** Not just "save the file somewhere else": Vercel rejects any
   request body over **4.5MB**, and your upload limit is 100MB. So the browser has to upload
   straight to Cloudinary with a signature this app hands it, and then tell the app the
   URL. The server never sees the bytes on the way in.
4. **State that survives between requests.** The manifest, the ownership record, the
   ingestion job's progress, the rate limiter and the answer cache all live in this
   process's memory or in `storage/`. A Vercel function gets neither: no disk, and a fresh
   process whenever it feels like it. All of it moves into MongoDB.
5. **Ingestion without a background thread.** Vercel kills everything the moment the
   response is sent, so today's "start a thread, poll `/api/activity`" design cannot work.
   Ingestion becomes resumable: each request does one slice of work and returns what is
   left, and the browser keeps asking until it is done. The progress bar you already have
   is what drives it.
6. **The keyword half of hybrid search.** BM25 currently reads the whole corpus into memory
   to build its index. On a machine that stays up, that happens once. On serverless it
   would happen on every cold start, per user. In cloud mode it becomes a Pinecone sparse
   index, which does the same job server-side; `rank-bm25` stays for local. Until then the
   Pinecone backend does the scan with a cap on it, so a large corpus degrades keyword
   ranking rather than hanging a request.

Also in that last step: a `requirements-cloud.txt` with no torch and no
sentence-transformers (they do not fit the bundle and are not used in cloud mode), an
`api/index.py` entrypoint, and `vercel.json`.

---

## Part C — deploying, once Part B is finished

### 1. Push to GitHub

Check before you push:

```bash
git status --porcelain --ignored | grep -E "\.env$|storage/|data/"
```

`.env`, `storage/` and `data/` must all be *ignored*. `.env` holds your Groq key, your
Atlas password and your Cloudinary secret; a public repo with that in it is compromised
within hours, by bots, not by people.

### 2. Import the project on Vercel

New Project → pick the repo → framework preset **Other**. Leave the build command empty.

### 3. Set the environment variables

Vercel dashboard → Settings → Environment Variables. Add every one of these for
**Production** *and* **Preview**:

```
RAG_MODE=cloud
GROQ_API_KEY=...
PINECONE_API_KEY=...
MONGO_URI=mongodb+srv://...
CLOUDINARY_CLOUD_NAME=...
CLOUDINARY_API_KEY=...
CLOUDINARY_API_SECRET=...
JWT_SECRET=<64 random hex characters>
```

Generate the JWT secret yourself and paste it in:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Do not let the app generate it on Vercel. Locally it writes the generated key back into
`.env`; on Vercel there is nothing to write to, so every cold start would invent a new key
and sign everybody out mid-session.

### 4. Deploy, then check

- `https://<your-app>.vercel.app/ready` → should be `{"ready": true, ...}`.
- `https://<your-app>.vercel.app/info` → confirm `"mode": "cloud"` and
  `"embeddings_provider": "pinecone"`. If it says `local`, `RAG_MODE` did not reach the
  function and you are about to get a bundle-size error instead.
- Open the app, create an account, upload one small PDF, ask it something.

Use a *small* PDF for that first test. If something is wrong with the Pinecone key you want
to find out after 3 API calls, not after 300.

---

## What deployment costs you, honestly

None of this is a reason not to deploy. All of it is a reason not to be surprised.

- **Cold starts.** A function that has not run for a while takes a few seconds on the first
  request. There is no way around this on the free tier.
- **Indexing a large book is slow and chatty.** It is many small resumable requests instead
  of one long background job, and the browser tab has to stay open. Closing it pauses the
  work; reopening resumes it.
- **500 questions a month**, and about four books indexed. Fine for a demo and a portfolio
  piece; not fine for a class of 40 students. Past the rerank quota the app keeps answering
  with slightly worse ranking rather than failing.
- **Answers are no longer free.** In local mode the embedding and re-ranking happen on your
  CPU and cost nothing. In cloud mode every question is two API calls against a quota.
- **Local mode stays exactly as it is.** Develop against `RAG_MODE=local`, deploy with
  `cloud`. Same code, same tests, same behaviour — only the backends differ.

## If you want it live sooner

Everything in Part B exists because Vercel functions have no disk and no life after the
response. A container host does have both, and the current code would run there today with
only the Atlas and Pinecone settings changed. You have said you want Vercel, so this
guide targets Vercel — but if a deadline appears, that is the shortcut, and it does not
throw away any of this work.
