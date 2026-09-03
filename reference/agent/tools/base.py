"""Agent tool interface.

Every tool exposes a name + description (used by the planner to select it) and
a ``run`` method returning normalized ``Evidence`` so results from any tool can
be merged uniformly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.agent.state import Evidence


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    def run(self, query: str, *, document_id: str | None = None) -> list[Evidence]: ...
