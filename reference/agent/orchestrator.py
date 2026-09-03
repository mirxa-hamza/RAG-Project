"""Agent orchestrator: builds the LangGraph agent from real nodes and runs it."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from app.agent.graph import DEFAULT_MAX_ITERATIONS, build_agent_graph
from app.agent.merge import merge_evidence
from app.agent.state import AgentState
from app.agent.tools.base import Tool
from app.core.config import Settings
from app.core.logging import get_logger
from app.services.retrieval.answerer import INSUFFICIENT_EVIDENCE, Answerer, Citation
from app.services.retrieval.context import AssembledContext, ContextPassage

logger = get_logger(__name__)

DOCUMENT_TOOL = "vector_search"
WEB_TOOL = "web_search"


class PlannerProtocol(Protocol):
    def plan(
        self,
        *,
        question: str,
        tools: list[Tool],
        iteration: int = 0,
        prior_insufficient: bool = False,
    ) -> list[str]: ...


class AgentResult(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[ContextPassage]
    tools_run: list[str]
    iterations: int


class AgentOrchestrator:
    def __init__(
        self,
        *,
        planner: PlannerProtocol,
        tools: list[Tool],
        answerer: Answerer,
        settings: Settings,
    ) -> None:
        self._planner = planner
        self._tools: dict[str, Tool] = {t.name: t for t in tools}
        self._answerer = answerer
        self._settings = settings
        self._graph = build_agent_graph(
            plan_node=self._plan_node,
            tools_node=self._tools_node,
            assemble_node=self._assemble_node,
            generate_node=self._generate_node,
        )

    # --- tool policy ---
    def _apply_tool_policy(self, names: list[str], *, iteration: int) -> list[str]:
        """Constrain the planner's choice so the ingested corpus comes first.

        The planner is a small-model JSON call with no idea whether documents
        were ever uploaded, so left alone it will happily answer a question
        about the user's own PDF from the open web. Policy:

        - ``vector_search`` always runs, on every iteration.
        - ``web_search`` is a fallback: barred on the first pass (unless the
          corpus tool is unavailable), so the web is consulted only after the
          documents have actually failed to answer.
        """
        allowed = [n for n in names if n in self._tools]

        if DOCUMENT_TOOL in self._tools and DOCUMENT_TOOL not in allowed:
            allowed.insert(0, DOCUMENT_TOOL)

        first_pass_with_corpus = iteration == 0 and DOCUMENT_TOOL in self._tools
        web_barred = not self._settings.agent_allow_web_search or (
            self._settings.agent_web_search_is_fallback and first_pass_with_corpus
        )
        if web_barred:
            allowed = [n for n in allowed if n != WEB_TOOL]

        return allowed

    # --- nodes ---
    def _plan_node(self, state: AgentState) -> dict[str, Any]:
        iteration = state.get("iteration", 0)
        proposed = self._planner.plan(
            question=state["question"],
            tools=list(self._tools.values()),
            iteration=iteration,
            prior_insufficient=iteration > 0,
        )
        names = self._apply_tool_policy(proposed, iteration=iteration)
        if names != proposed:
            logger.info("tool_policy_applied", proposed=proposed, planned=names)
        return {"plan": names, "iteration": iteration + 1}

    def _tools_node(self, state: AgentState) -> dict[str, Any]:
        evidence = []
        ran: list[str] = []
        for name in state.get("plan", []):
            tool = self._tools.get(name)
            if tool is None:
                logger.warning("unknown_tool_skipped", tool=name)
                continue
            evidence.extend(tool.run(state["question"], document_id=state.get("document_id")))
            ran.append(name)
        return {"evidence": evidence, "tools_run": ran}

    def _assemble_node(self, state: AgentState) -> dict[str, Any]:
        context = merge_evidence(
            state.get("evidence", []),
            token_budget=self._settings.context_token_budget,
        )
        return {"context": context}

    def _generate_node(self, state: AgentState) -> dict[str, Any]:
        context = state.get("context") or AssembledContext(passages=[])
        answer = self._answerer.answer(question=state["question"], context=context)
        # Re-query only when retrieval genuinely came up short: no passages, or
        # the model declined for lack of evidence. Keying this on citations
        # instead would discard a correct, document-grounded answer whenever a
        # small model forgets the [n] markers -- and the re-plan that follows
        # explicitly nudges the planner toward the web, so a formatting slip
        # would turn a RAG answer into a web answer.
        declined = answer.answer.strip() == INSUFFICIENT_EVIDENCE
        done = not context.is_empty and not declined
        return {"answer": answer.answer, "citations": answer.citations, "done": done}

    # --- entry point ---
    def run(
        self,
        *,
        question: str,
        document_id: str | None = None,
        max_iterations: int | None = None,
    ) -> AgentResult:
        initial: AgentState = {
            "question": question,
            "document_id": document_id,
            "iteration": 0,
            "max_iterations": max_iterations or DEFAULT_MAX_ITERATIONS,
        }
        final = self._graph.invoke(initial)
        context = final.get("context") or AssembledContext(passages=[])
        result = AgentResult(
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            sources=context.passages,
            tools_run=final.get("tools_run", []),
            iterations=final.get("iteration", 0),
        )
        logger.info(
            "agent_run_complete",
            iterations=result.iterations,
            tools_run=result.tools_run,
            citations=len(result.citations),
        )
        return result
