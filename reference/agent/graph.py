"""LangGraph orchestration skeleton.

Wires the agent control flow ``plan -> tools -> assemble -> generate`` with a
bounded re-query loop. Node implementations are injected so they can be built
and tested independently.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from app.agent.state import AgentState

NodeFn = Callable[[AgentState], dict[str, Any]]

DEFAULT_MAX_ITERATIONS = 2


def should_continue(state: AgentState) -> str:
    """Decide whether to re-query (loop) or finish."""
    if state.get("done"):
        return "end"
    if state.get("iteration", 0) >= state.get("max_iterations", DEFAULT_MAX_ITERATIONS):
        return "end"
    return "continue"


def build_agent_graph(
    *,
    plan_node: NodeFn,
    tools_node: NodeFn,
    assemble_node: NodeFn,
    generate_node: NodeFn,
) -> Any:
    """Build and compile the agent graph from injected node functions."""
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("tools", tools_node)
    graph.add_node("assemble", assemble_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "tools")
    graph.add_edge("tools", "assemble")
    graph.add_edge("assemble", "generate")
    graph.add_conditional_edges(
        "generate", should_continue, {"continue": "plan", "end": END}
    )
    return graph.compile()
