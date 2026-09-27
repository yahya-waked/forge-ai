"""
Tool Registry — where every capability the Agent can use is declared,
schema-described, and gated by a risk level.

The LLM never executes anything itself: it asks to call a tool by name
with arguments. The Orchestrator looks the tool up here, checks
permissions (Phase 7), and only then runs the real Python function.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

from app.llm.base import ToolSchema


class RiskLevel(IntEnum):
    LOW = 1       # read_file, search_code, git_status, list_files
    MEDIUM = 2    # write_file, apply_patch, run_tests
    HIGH = 3      # git_commit, git_push, create_pull_request


class ToolError(Exception):
    """Raised on a handled tool failure (bad arguments, not found, execution error)."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema describing the arguments
    executor: Callable[..., Any]
    risk: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False

    def to_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name, description=self.description, parameters=self.parameters
        )


class ToolRegistry:
    """Holds every Tool available to the agent(s) in a run."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        risk: RiskLevel = RiskLevel.LOW,
        requires_approval: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form: @registry.register("read_file", "...", {...schema...})"""

        def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = Tool(
                name=name,
                description=description,
                parameters=parameters,
                executor=fn,
                risk=risk,
                requires_approval=requires_approval,
            )
            return fn

        return _wrap

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool '{name}'") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def schemas(self, *, max_risk: RiskLevel = RiskLevel.HIGH) -> list[ToolSchema]:
        """Schemas for tools at or below max_risk — what gets handed to the LLM."""
        return [t.to_schema() for t in self._tools.values() if t.risk <= max_risk]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self.get(name)
        try:
            return tool.executor(**arguments)
        except TypeError as exc:
            raise ToolError(f"Bad arguments for '{name}': {exc}") from exc


# Shared registry instance; individual tool modules attach to it via
# @registry.register(...) so importing a tool module is enough to enable it.
registry = ToolRegistry()