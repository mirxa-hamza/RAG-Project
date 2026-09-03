"""LLM client abstraction and Gemini implementation.

An ``LLMClient`` protocol decouples callers from the provider. The Gemini
implementation uses the google-genai SDK's ``models.generate_content`` and
returns the response text. The underlying client is injectable so tests need
neither an API key nor the SDK.

The request config is built as a plain dict rather than ``google.genai.types``
objects on purpose: it keeps this module import-free of the SDK, which is what
lets the whole test suite run without google-genai installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from app.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, *, system: str, prompt: str, max_tokens: int) -> str: ...


@runtime_checkable
class StreamingLLMClient(LLMClient, Protocol):
    def stream(self, *, system: str, prompt: str, max_tokens: int) -> Iterator[str]: ...


def _text_of(response: Any) -> str:
    """Best-effort text extraction from a Gemini response or stream chunk.

    ``.text`` is None when a chunk carries only thinking output or the response
    was blocked, so callers get "" instead of a TypeError.
    """
    return getattr(response, "text", None) or ""


class GeminiLLM:
    """Gemini-backed LLM client.

    ``thinking_level`` is Gemini 3's reasoning-depth control
    ("minimal" | "low" | "medium" | "high"). "low" keeps latency and cost down
    for the short, structured calls this app makes (tool planning, cited
    answering, grounding verdicts); raise it if answer quality needs it.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        thinking_level: str = "low",
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._thinking_level = thinking_level
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai  # lazy

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _config(self, *, system: str, max_tokens: int) -> dict[str, Any]:
        return {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "thinking_config": {"thinking_level": self._thinking_level},
        }

    def generate(self, *, system: str, prompt: str, max_tokens: int) -> str:
        response = self._get_client().models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config(system=system, max_tokens=max_tokens),
        )
        text = _text_of(response)
        logger.info("llm_generate", model=self._model, chars=len(text))
        return text

    def stream(self, *, system: str, prompt: str, max_tokens: int) -> Iterator[str]:
        chunks = self._get_client().models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=self._config(system=system, max_tokens=max_tokens),
        )
        for chunk in chunks:
            text = _text_of(chunk)
            if text:
                yield text
