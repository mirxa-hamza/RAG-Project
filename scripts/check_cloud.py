"""
Preflight for a cloud deployment: does every external service this configuration depends on
actually answer, with these credentials, right now?

Run this BEFORE `vercel deploy`, and again against the deployed URL's own environment if a
deployment misbehaves. It is deliberately not a test of the app - `tests/test_pipeline_offline.py`
covers the pipeline, and this covers the five things that live outside the process and can
only fail for environmental reasons: a wrong key, a spent quota, an index whose dimension
does not match the embedding model, a Mongo cluster that will not accept a write, a
Cloudinary signature computed by hand and rejected without explanation.

    python scripts/check_cloud.py             # everything this configuration uses
    python scripts/check_cloud.py --rerank     # ALSO spend one of Pinecone's 500 monthly reranks
    python scripts/check_cloud.py --all        # check every backend, even unused ones

Exit code is 0 only when nothing failed, so it drops straight into CI or a pre-deploy hook.

Why the configuration section comes first and can fail on its own: the most expensive
deployment bugs here are not bad credentials, they are a setting that is *valid* but wrong
for a serverless host - STATE_STORE=memory resets on every request, DOCUMENT_STORE=local
writes PDFs to a filesystem that is discarded, a blank JWT_SECRET signs everyone out on
every cold start. Each of those deploys cleanly and then behaves strangely in production,
which is the worst way to find out.
"""
import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import config  # noqa: E402

OK, FAIL, WARN, SKIP = "OK", "FAIL", "WARN", "SKIP"
_COLORS = {OK: "\033[32m", FAIL: "\033[31m", WARN: "\033[33m", SKIP: "\033[90m"}
_RESET = "\033[0m"

_results = []


def report(status: str, label: str, detail: str = "") -> None:
    _results.append(status)
    colour = _COLORS[status] if sys.stdout.isatty() else ""
    reset = _RESET if sys.stdout.isatty() else ""
    print(f"  {colour}[{status:^4}]{reset} {label}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"           {line}")


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _name_of(item) -> str:
    """
    Index/collection listings come back as model objects in some client versions and as
    plain strings or dicts in others. Reading one field three ways is cheaper than pinning
    this script to a client version that the app itself does not pin.
    """
    if isinstance(item, str):
        return item
    name = getattr(item, "name", None)
    if name:
        return str(name)
    try:
        return str(item["name"])
    except (TypeError, KeyError):
        return str(item)


# ------------------------------------------------------------------ 1. configuration


def check_configuration() -> None:
    section("Configuration")

    report(OK, f"RAG_MODE={config.RAG_MODE}", (
        f"vectors     {config.VECTOR_STORE}"
        f"{' (' + config.CHROMA_BACKEND + ')' if config.VECTOR_STORE == 'chroma' else ''}\n"
        f"embeddings  {config.EMBEDDINGS_PROVIDER}\n"
        f"re-ranking  {config.RERANKER_PROVIDER}\n"
        f"documents   {config.DOCUMENT_STORE}\n"
        f"state       {config.STATE_STORE}"
    ))

    if not config.GROQ_API_KEY:
        report(FAIL, "GROQ_API_KEY is not set", "Every answer is a Groq call - nothing works without it.")

    if not config.IS_CLOUD:
        report(SKIP, "serverless suitability checks", "RAG_MODE is not 'cloud' - skipping.")
        return

    # Each of these is valid configuration that a serverless host silently breaks.
    if config.STATE_STORE != "mongo":
        report(FAIL, f"STATE_STORE={config.STATE_STORE} on a serverless host", (
            "A cold start gets a fresh process, so rate limits, the answer cache, the\n"
            "ingestion job and the manifest would reset on essentially every request.\n"
            "Set STATE_STORE=mongo."
        ))
    if config.DOCUMENT_STORE != "cloudinary":
        report(FAIL, f"DOCUMENT_STORE={config.DOCUMENT_STORE} on a serverless host", (
            "There is no persistent disk: uploaded PDFs would be discarded when the\n"
            "function that wrote them exits. Set DOCUMENT_STORE=cloudinary."
        ))
    if config.VECTOR_STORE == "chroma" and config.CHROMA_BACKEND != "cloud":
        report(FAIL, "VECTOR_STORE=chroma with CHROMA_BACKEND=disk", (
            "A folder-backed index is empty again on every cold start.\n"
            "Set CHROMA_BACKEND=cloud, or VECTOR_STORE=pinecone."
        ))
    if config.EMBEDDINGS_PROVIDER == "local":
        report(FAIL, "EMBEDDINGS_PROVIDER=local on a serverless host",
               "torch and sentence-transformers are not in requirements-cloud.txt.")
    if config.RERANKER_PROVIDER == "local":
        report(FAIL, "RERANKER_PROVIDER=local on a serverless host",
               "Same reason: the cross-encoder needs torch, which is not in the bundle.")
    if not config.JWT_SECRET:
        report(FAIL, "JWT_SECRET is blank", (
            "Locally one is generated and written to .env. A serverless filesystem is\n"
            "read-only, so a blank secret means a NEW random key per cold start - every\n"
            "token is invalid almost immediately. Generate one and set it as an env var:\n"
            "  python -c \"import secrets; print(secrets.token_hex(32))\""
        ))
    if config.MONGO_URI.startswith("mongodb://localhost"):
        report(FAIL, "MONGO_URI points at localhost",
               "A deployed function cannot reach your machine. Use the Atlas SRV URI.")


# ------------------------------------------------------------------ 2. Groq


def check_groq() -> None:
    section("Groq (answers)")
    if not config.GROQ_API_KEY:
        report(SKIP, "Groq", "No GROQ_API_KEY set.")
        return

    import httpx

    try:
        response = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            timeout=config.PROVIDER_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        report(FAIL, "Groq unreachable", str(exc))
        return

    if response.status_code == 401:
        report(FAIL, "Groq rejected the API key (401)", "Check GROQ_API_KEY.")
        return
    if response.status_code != 200:
        report(FAIL, f"Groq returned {response.status_code}", response.text[:300])
        return

    names = [m.get("id") for m in response.json().get("data", [])]
    report(OK, f"key accepted ({len(names)} models available)")

    # Worth its own check because Groq retires models on a schedule of their own, and the
    # failure arrives as a 404 on the first question rather than at startup.
    if config.GROQ_MODEL in names:
        report(OK, f"GROQ_MODEL '{config.GROQ_MODEL}' exists")
    else:
        report(FAIL, f"GROQ_MODEL '{config.GROQ_MODEL}' is not offered by this account", (
            "Groq deprecates models regularly. Currently available, for example:\n"
            + ", ".join(sorted(n for n in names if n)[:8])
        ))


# ------------------------------------------------------------------ 3. MongoDB


def check_mongo() -> None:
    section("MongoDB (accounts, and cloud-mode state)")

    from src.services import database

    probe_id = f"preflight-{uuid.uuid4().hex[:8]}"
    try:
        client = database.get_sync_client()
        client.admin.command("ping")
    except Exception as exc:                                    # noqa: BLE001
        report(FAIL, "cannot reach MongoDB", (
            f"{type(exc).__name__}: {exc}\n"
            "If this is Atlas: serverless functions get dynamic IPs, so the cluster's\n"
            "Network Access list needs 0.0.0.0/0 - an allowlist holding only your own\n"
            "address passes locally and then times out once deployed."
        ))
        return
    report(OK, f"connected ({config.MONGO_DB})")

    # Connectivity is not permission: an Atlas user with read-only rights pings happily and
    # then fails on the first write, which here would be a signup.
    try:
        probe = client[config.MONGO_DB]["preflight_probe"]
        probe.insert_one({"_id": probe_id, "at": time.time()})
        found = probe.find_one({"_id": probe_id}) is not None
        probe.delete_one({"_id": probe_id})
        report(OK if found else FAIL, "write / read / delete accepted")
    except Exception as exc:                                    # noqa: BLE001
        report(FAIL, "connected, but this user cannot write", (
            f"{type(exc).__name__}: {exc}\n"
            "Signups and every piece of cloud-mode state need write access."
        ))


# ------------------------------------------------------------------ 4. Pinecone embeddings


def check_pinecone_embeddings(force: bool = False) -> int:
    """Returns the live embedding dimension, or 0 - the Chroma check reuses it."""
    section("Pinecone (embeddings)")
    if config.EMBEDDINGS_PROVIDER != "pinecone" and not force:
        report(SKIP, "Pinecone embeddings", f"EMBEDDINGS_PROVIDER={config.EMBEDDINGS_PROVIDER}.")
        return 0
    if not config.PINECONE_API_KEY:
        report(FAIL, "PINECONE_API_KEY is not set")
        return 0

    from src.ml.providers import PineconeEmbeddings, ProviderError

    try:
        vector = PineconeEmbeddings().embed_query("preflight dimension probe")
    except ProviderError as exc:
        report(FAIL, "embedding call failed", str(exc))
        return 0
    except Exception as exc:                                    # noqa: BLE001
        report(FAIL, f"embedding call raised {type(exc).__name__}", str(exc))
        return 0

    report(OK, f"{config.PINECONE_EMBED_MODEL} answered ({len(vector)} dimensions)")

    # The single most common way this stack breaks: an index or collection is created at
    # one dimension, the embedding model is later changed, and every upsert is rejected.
    if len(vector) != config.PINECONE_EMBED_DIM:
        report(FAIL, "PINECONE_EMBED_DIM does not match the model", (
            f"configured {config.PINECONE_EMBED_DIM}, model returns {len(vector)}.\n"
            "Set PINECONE_EMBED_DIM correctly AND recreate the index/collection - a\n"
            "dimension is fixed when the store is created and cannot be altered."
        ))
    return len(vector)


# ------------------------------------------------------------------ 5. Pinecone index


def check_pinecone_index(live_dim: int, force: bool = False) -> None:
    section("Pinecone (vector index)")
    if config.VECTOR_STORE != "pinecone" and not force:
        report(SKIP, "Pinecone index", f"VECTOR_STORE={config.VECTOR_STORE}.")
        return
    if not config.PINECONE_API_KEY:
        report(FAIL, "PINECONE_API_KEY is not set")
        return

    try:
        from pinecone import Pinecone

        client = Pinecone(api_key=config.PINECONE_API_KEY)
        names = [_name_of(i) for i in client.list_indexes()]
    except Exception as exc:                                    # noqa: BLE001
        report(FAIL, f"could not list indexes ({type(exc).__name__})", str(exc))
        return

    if config.PINECONE_INDEX not in names:
        report(WARN, f"index '{config.PINECONE_INDEX}' does not exist yet", (
            f"existing: {', '.join(names) or 'none'}\n"
            "The app creates it on first use, so this is only a problem if you expected\n"
            "it to be there already."
        ))
        return

    try:
        stats = client.Index(config.PINECONE_INDEX).describe_index_stats()
        dimension = stats.get("dimension")
        vectors = stats.get("total_vector_count", 0)
    except Exception as exc:                                    # noqa: BLE001
        report(FAIL, f"index exists but could not be described ({type(exc).__name__})", str(exc))
        return

    report(OK, f"index '{config.PINECONE_INDEX}' ({dimension} dimensions, {vectors} vectors)")
    if live_dim and dimension and dimension != live_dim:
        report(FAIL, "index dimension does not match the embedding model", (
            f"index {dimension}, model {live_dim}. Every upsert will be rejected.\n"
            "Delete and recreate the index, then re-ingest."
        ))


# ------------------------------------------------------------------ 6. Chroma Cloud


def check_chroma_cloud(live_dim: int, force: bool = False) -> None:
    section("Chroma Cloud (vectors)")
    uses_chroma_cloud = config.VECTOR_STORE == "chroma" and config.CHROMA_BACKEND == "cloud"
    if not uses_chroma_cloud and not force:
        report(SKIP, "Chroma Cloud",
               f"VECTOR_STORE={config.VECTOR_STORE}, CHROMA_BACKEND={config.CHROMA_BACKEND}.")
        return

    missing = [n for n, v in (("CHROMA_API_KEY", config.CHROMA_API_KEY),
                              ("CHROMA_TENANT", config.CHROMA_TENANT),
                              ("CHROMA_DATABASE", config.CHROMA_DATABASE)) if not v]
    if missing:
        report(FAIL, "Chroma Cloud is not fully configured", f"missing: {', '.join(missing)}")
        return

    try:
        import chromadb

        client = chromadb.CloudClient(
            tenant=config.CHROMA_TENANT,
            database=config.CHROMA_DATABASE,
            api_key=config.CHROMA_API_KEY,
        )
        collections = [_name_of(c) for c in client.list_collections()]
    except Exception as exc:                                    # noqa: BLE001
        report(FAIL, f"could not reach Chroma Cloud ({type(exc).__name__})", str(exc))
        return

    report(OK, f"connected to tenant/database ({len(collections)} collections)")

    if config.CHROMA_COLLECTION not in collections:
        report(WARN, f"collection '{config.CHROMA_COLLECTION}' does not exist yet",
               f"existing: {', '.join(collections) or 'none'} - it is created on first ingest.")
        return

    try:
        collection = client.get_collection(config.CHROMA_COLLECTION)
        count = collection.count()
        stored_dim = 0
        if count:
            peek = collection.get(limit=1, include=["embeddings"])
            embeddings = peek.get("embeddings")
            if embeddings is not None and len(embeddings):
                stored_dim = len(embeddings[0])
    except Exception as exc:                                    # noqa: BLE001
        report(FAIL, f"collection exists but could not be read ({type(exc).__name__})", str(exc))
        return

    report(OK, f"collection '{config.CHROMA_COLLECTION}' ({count} chunks"
               f"{f', {stored_dim} dimensions' if stored_dim else ''})")

    # The trap when moving from local mode: bge-small writes 384-dimensional vectors, and
    # llama-text-embed-v2 writes 1024. The collection keeps whichever it was created with.
    if live_dim and stored_dim and stored_dim != live_dim:
        report(FAIL, "this collection was built with a different embedding model", (
            f"stored {stored_dim} dimensions, current model produces {live_dim}.\n"
            "Point CHROMA_COLLECTION at a new name and re-ingest - a collection's\n"
            "dimension is fixed when it is created."
        ))


# ------------------------------------------------------------------ 7. Cloudinary


def check_cloudinary(force: bool = False) -> None:
    section("Cloudinary (PDF storage)")
    if config.DOCUMENT_STORE != "cloudinary" and not force:
        report(SKIP, "Cloudinary", f"DOCUMENT_STORE={config.DOCUMENT_STORE}.")
        return

    missing = [n for n, v in (("CLOUDINARY_CLOUD_NAME", config.CLOUDINARY_CLOUD_NAME),
                              ("CLOUDINARY_API_KEY", config.CLOUDINARY_API_KEY),
                              ("CLOUDINARY_API_SECRET", config.CLOUDINARY_API_SECRET)) if not v]
    if missing:
        report(FAIL, "Cloudinary is not fully configured", f"missing: {', '.join(missing)}")
        return

    import httpx

    from src.services import cloudinary_store

    # A real round trip, through the app's own signing code. The signature is computed by
    # hand (SHA-1 over sorted params + secret), and Cloudinary rejects a wrong one with a
    # generic error - so "the credentials are set" is not evidence that uploads work. This
    # uploads a few bytes to a probe folder and deletes them again.
    probe_user = f"preflight-{uuid.uuid4().hex[:8]}"
    try:
        signed = cloudinary_store.sign_upload(probe_user)
    except cloudinary_store.CloudinaryError as exc:
        report(FAIL, "could not build a signed upload payload", str(exc))
        return

    try:
        response = httpx.post(
            signed["upload_url"],
            data={
                "api_key": signed["api_key"],
                "timestamp": signed["timestamp"],
                "folder": signed["folder"],
                "signature": signed["signature"],
            },
            files={"file": ("preflight.pdf", b"%PDF-1.4 preflight probe\n", "application/pdf")},
            timeout=config.PROVIDER_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        report(FAIL, "upload request failed", str(exc))
        return

    if response.status_code != 200:
        report(FAIL, f"Cloudinary rejected the signed upload ({response.status_code})", (
            f"{response.text[:300]}\n"
            "A 401 here is almost always CLOUDINARY_API_SECRET, since the signature is\n"
            "what the secret produces."
        ))
        return

    public_id = response.json().get("public_id", "")
    report(OK, "signed upload accepted", f"public_id: {public_id}")

    # The isolation guarantee /upload/complete relies on: a real upload must land inside
    # the uploading user's own folder, or the ownership check would reject legitimate files.
    if cloudinary_store.public_id_belongs_to(public_id, probe_user):
        report(OK, "the uploaded public_id is inside the user's own folder")
    else:
        report(FAIL, "uploaded public_id is NOT inside the expected user folder", (
            f"got '{public_id}', expected it under '{cloudinary_store.user_folder(probe_user)}'.\n"
            "POST /upload/complete would reject every genuine upload with a 403."
        ))

    if cloudinary_store.destroy(public_id):
        report(OK, "probe file deleted again")
    else:
        report(WARN, "probe file could not be deleted",
               f"Harmless, but remove '{public_id}' by hand to keep the account tidy.")


# ------------------------------------------------------------------ 8. Pinecone rerank


def check_pinecone_rerank() -> None:
    # No provider guard: reaching here means --rerank was passed, i.e. the caller explicitly
    # asked to spend a request testing that the key and quota work. Skipping because
    # RERANKER_PROVIDER is currently something else would silently ignore the flag - and
    # checking it BEFORE flipping the switch is exactly when you want to know.
    section("Pinecone (re-ranking)")
    if config.RERANKER_PROVIDER != "pinecone":
        report(WARN, "RERANKER_PROVIDER is not 'pinecone' yet",
               f"currently '{config.RERANKER_PROVIDER}' - testing the credential anyway.")
    if not config.PINECONE_API_KEY:
        report(FAIL, "PINECONE_API_KEY is not set")
        return

    from src.ml.providers import PineconeReranker

    ranked = PineconeReranker().rerank(
        "what is a preflight check?",
        [{"text": "A preflight check verifies external services before deploying."},
         {"text": "Unrelated text about gardening in the winter months."}],
    )
    if ranked is None:
        report(FAIL, "rerank returned nothing", (
            "The provider fails OPEN by design, so the app would keep answering with\n"
            "fused ranking instead. Check the log line above for the cause - a spent\n"
            "monthly quota (500 on the free tier) looks exactly like this."
        ))
        return
    report(OK, f"{config.PINECONE_RERANK_MODEL} answered ({len(ranked)} scored)",
           f"top score {ranked[0][1]:.3f} against a floor of {config.MIN_RERANK_SCORE_API}")


# ------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for a cloud deployment.")
    parser.add_argument("--rerank", action="store_true",
                        help="also test Pinecone rerank (spends one of 500 free monthly requests)")
    parser.add_argument("--all", action="store_true",
                        help="check every backend, including ones this configuration does not use")
    args = parser.parse_args()

    print(f"Cloud preflight - {config.PROJECT_ROOT}")

    check_configuration()
    check_groq()
    check_mongo()
    live_dim = check_pinecone_embeddings(force=args.all)
    check_pinecone_index(live_dim, force=args.all)
    check_chroma_cloud(live_dim, force=args.all)
    check_cloudinary(force=args.all)
    if args.rerank:
        check_pinecone_rerank()
    else:
        section("Pinecone (re-ranking)")
        report(SKIP, "rerank", "Costs one of 500 free monthly requests - pass --rerank to test it.")

    failed = _results.count(FAIL)
    warned = _results.count(WARN)
    print(f"\n{'=' * 60}")
    print(f"{_results.count(OK)} passed, {failed} failed, {warned} warnings, "
          f"{_results.count(SKIP)} skipped")
    if failed:
        print("\nNot ready to deploy - fix the failures above.")
    elif warned:
        print("\nNo failures. Read the warnings and decide whether they matter.")
    else:
        print("\nEvery configured service answered. Ready to deploy.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
