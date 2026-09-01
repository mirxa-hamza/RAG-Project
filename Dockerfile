# Two stages, for one reason: the model download is the slowest layer and changes almost
# never, while the source changes constantly. Baking the models into their own layer means
# a code edit rebuilds in seconds instead of re-downloading ~210MB from Hugging Face - and
# it means the running container never needs the internet to answer a question.
FROM python:3.13-slim AS models

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

RUN pip install --no-cache-dir "sentence-transformers==3.0.1"

# Pinned by revision, not just by name: a model repository can be updated under the same
# name, which would silently change every embedding you have already stored.
RUN python - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("models cached")
PY


FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    LOG_FORMAT=json \
    DATA_DIR=/data \
    CHROMA_DIR=/storage/chroma_db

WORKDIR /app

# Dependencies first, so editing source does not reinstall torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=models /models /models
COPY src/ ./src/
COPY scripts/ ./scripts/

# Never run as root: this process writes files whose names came from an HTTP request.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data /storage \
 && chown -R app:app /app /data /storage /models
USER app

EXPOSE 8000

# Readiness, not liveness: /health answers before the embedding model is loaded, so a
# probe on it would route traffic to a process that cannot answer a question yet.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"

# ONE worker, deliberately. The ingestion job, the BM25 cache, the rate limiter and the
# answer cache are all in-process, and ChromaDB's persistent client is single-process:
# a second worker would silently double every rate limit and corrupt the store.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
