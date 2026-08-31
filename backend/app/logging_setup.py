"""
One place to configure logging, plus a small timing helper.

Every pipeline stage logs how long it took. Without per-stage timings you can't tell
whether a slow answer came from retrieval or from the LLM, which is the first question
you'll ask when tuning anything.
"""
import logging
import time
from contextlib import contextmanager

from app.config import LOG_LEVEL

_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)


@contextmanager
def timed(logger: logging.Logger, label: str, level: int = logging.INFO):
    """`with timed(log, "embed 1781 chunks"):` -> logs "embed 1781 chunks took 12.4s"."""
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.log(level, "%s took %.2fs", label, time.perf_counter() - start)
