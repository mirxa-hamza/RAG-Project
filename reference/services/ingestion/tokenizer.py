"""Token counting/encoding wrapper.

Isolated behind a tiny interface so the token model is swappable and the rest
of the codebase never imports ``tiktoken`` directly. ``cl100k_base`` is used as
a stable proxy for chunk sizing.
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoder().encode(text))


def encode(text: str) -> list[int]:
    return _encoder().encode(text)


def decode(tokens: list[int]) -> str:
    return _encoder().decode(tokens)
