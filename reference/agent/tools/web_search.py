"""Web search tool backed by the Tavily API.

Degrades gracefully to an empty result set when no API key is configured, so
the agent still functions with vector search alone. The Tavily client is
injectable so tests need neither the SDK nor a key.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import Evidence
from app.core.logging import get_logger

logger = get_logger(__name__)


class TavilySearchTool:
    name = "web_search"
    description = (
        "Live web search for current events, recent information, or facts not "
        "present in the ingested documents."
    )

    def __init__(self, *, api_key: str, max_results: int = 5, client: Any | None = None) -> None:
        self._api_key = api_key
        self._max_results = max_results
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient  # lazy

            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    def run(self, query: str, *, document_id: str | None = None) -> list[Evidence]:
        if not self._api_key and self._client is None:
            logger.warning("web_search_skipped", reason="no_tavily_api_key")
            return []

        response = self._get_client().search(query, max_results=self._max_results)
        results = response.get("results", []) if isinstance(response, dict) else []
        evidence = [
            Evidence(
                text=r.get("content", ""),
                origin="web",
                score=float(r.get("score", 0.0)),
                source_id=r.get("url", ""),
                title=r.get("title", r.get("url", "web result")),
                url=r.get("url"),
            )
            for r in results
            if r.get("content")
        ]
        logger.info("web_search", query_len=len(query), hits=len(evidence))
        return evidence
