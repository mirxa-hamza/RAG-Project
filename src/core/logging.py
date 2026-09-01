"""
One place to configure logging, plus a small timing helper.

Every pipeline stage logs how long it took. Without per-stage timings you can't tell
whether a slow answer came from retrieval or from the LLM, which is the first question
you'll ask when tuning anything.
"""
import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from src.core.config import LOG_FORMAT, LOG_LEVEL

# The id of the request being handled, so every line emitted while serving it can be tied
# together afterwards. A ContextVar rather than a thread local: FastAPI serves async
# handlers, where one thread interleaves many requests.
request_id: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get("")
        return True


class JsonFormatter(logging.Formatter):
    """
    One JSON object per line.

    Prose logs are pleasant to read and impossible to aggregate: "how long do questions take
    at p95" cannot be answered from them without regexes that break the next time a message
    is reworded. The timing data already exists (see `timed()`); this is what stops it being
    thrown away.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if getattr(record, "request_id", ""):
            payload["request_id"] = record.request_id
        # Anything attached via logger.info(..., extra={...}) rides along.
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)

_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler()
    handler.addFilter(_RequestIdFilter())
    if LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(LOG_LEVEL)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)


class _QuietPollFilter(logging.Filter):
    """Drops uvicorn access lines for the endpoints the UI polls while a job runs."""

    NOISY = ("/ingest/status", "/stats")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not (any(p in msg for p in self.NOISY) and " 200 " in msg)


def quiet_access_log() -> None:
    """
    The page polls /ingest/status and /stats about once a second, which buries the real
    ingestion progress under a wall of 200s. Only successful polls are filtered out;
    errors and every other request still log normally.
    """
    logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())


@contextmanager
def timed(logger: logging.Logger, label: str, level: int = logging.INFO):
    """
    `with timed(log, "embed 1781 chunks"):` -> logs "embed 1781 chunks took 12.4s".

    In JSON mode the duration is also emitted as a real number under `duration_ms`, so a
    log aggregator can compute percentiles without parsing the sentence.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s took %.2fs", label, elapsed,
                   extra={"extra_fields": {"stage": label,
                                           "duration_ms": round(elapsed * 1000, 1)}})
