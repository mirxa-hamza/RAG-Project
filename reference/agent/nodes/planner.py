"""Planner: choose which tools to run for a question.

Uses the LLM to produce a JSON tool plan, with defensive parsing and a safe
default so the plan is always valid and non-empty when tools exist.
"""

from __future__ import annotations

import json
import re

from app.agent.tools.base import Tool
from app.core.logging import get_logger
from app.services.llm import LLMClient

logger = get_logger(__name__)

PLANNER_SYSTEM = (
    "You are an agent orchestrator. Given a user question and a set of available "
    "tools, decide which tools to run to answer it. Respond ONLY with a JSON object "
    'of the form {"tools": ["tool_name", ...], "reasoning": "..."}. '
    "Choose the minimal set of tools that will answer the question."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_DEFAULT_TOOL = "vector_search"


def build_planner_prompt(
    question: str,
    tools: list[tuple[str, str]],
    *,
    prior_insufficient: bool,
) -> str:
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in tools)
    lines = [f"Question: {question}", "", "Available tools:", catalog]
    if prior_insufficient:
        lines += [
            "",
            "NOTE: A previous attempt did not retrieve enough evidence to answer. "
            "Consider adding web_search or broadening the query.",
        ]
    lines += ["", 'Respond with JSON: {"tools": [...], "reasoning": "..."}']
    return "\n".join(lines)


def parse_plan(raw: str, valid: set[str]) -> list[str]:
    """Extract a deduped list of valid tool names from an LLM response."""
    names: list[str] = []

    match = _JSON_RE.search(raw)
    if match:
        try:
            data = json.loads(match.group(0))
            candidates = data.get("tools", []) if isinstance(data, dict) else []
            names = [t for t in candidates if isinstance(t, str)]
        except json.JSONDecodeError:
            names = []

    if not names:
        # Fallback: scan the raw text for known tool names.
        names = [name for name in valid if name in raw]

    seen: dict[str, None] = {}
    for n in names:
        if n in valid:
            seen.setdefault(n, None)
    return list(seen)


class Planner:
    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    def plan(
        self,
        *,
        question: str,
        tools: list[Tool],
        iteration: int = 0,
        prior_insufficient: bool = False,
    ) -> list[str]:
        if not tools:
            return []
        catalog = [(t.name, t.description) for t in tools]
        valid = {name for name, _ in catalog}

        prompt = build_planner_prompt(question, catalog, prior_insufficient=prior_insufficient)
        raw = self._llm.generate(system=PLANNER_SYSTEM, prompt=prompt, max_tokens=300)
        names = parse_plan(raw, valid)

        if not names:
            names = [_DEFAULT_TOOL] if _DEFAULT_TOOL in valid else [catalog[0][0]]

        logger.info("planned_tools", tools=names, iteration=iteration)
        return names
