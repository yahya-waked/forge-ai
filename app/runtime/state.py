"""
Agent State — the operational memory of a single ForgeAI run.

One AgentState instance is created per task and threaded through the
whole Agent Loop: every LLM call reads it, every tool call updates it.
Nothing about a run's progress should live anywhere else — no globals,
no hidden state on the agent object.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.llm.base import Message


class RunStatus(str, Enum):
    PENDING = "pending"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    CODING = "coding"
    TESTING = "testing"
    RECOVERING = "recovering"
    REVIEWING = "reviewing"
    AWAITING_APPROVAL = "awaiting_approval"
    COMMITTING = "committing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ToolCallRecord:
    """One executed tool call — kept for the live trace and episodic memory."""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


@dataclass
class TestResult:
    passed: bool
    total: int = 0
    failed: int = 0
    output: str = ""


@dataclass
class AgentState:
    """The full operational memory of one ForgeAI task run."""

    task: str
    repository: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: RunStatus = RunStatus.PENDING

    # Conversation the LLM sees (system + user + assistant + tool messages)
    messages: list[Message] = field(default_factory=list)

    # Planning
    current_plan: list[str] = field(default_factory=list)

    # Repository understanding
    files_examined: set[str] = field(default_factory=set)
    retrieved_context: list[str] = field(default_factory=list)

    # Execution trace
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    modifications: dict[str, str] = field(default_factory=dict)  # file path -> diff/patch

    # Testing / recovery
    test_results: list[TestResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recovery_attempts: int = 0

    # Review / safety
    review_results: dict[str, Any] = field(default_factory=dict)
    permissions_granted: set[str] = field(default_factory=set)

    # Loop control (see 21. Infinite Loop Protection in the project spec)
    iteration_count: int = 0
    max_iterations: int = 15
    max_tool_calls: int = 30
    max_recovery_attempts: int = 5  # separate cap on the fix -> test -> fail cycle

    final_result: str | None = None

    def record_tool_call(self, call: ToolCallRecord) -> None:
        self.tool_calls.append(call)

    def add_observation(self, text: str) -> None:
        self.observations.append(text)

    def mark_file_examined(self, path: str) -> None:
        self.files_examined.add(path)

    def has_budget(self) -> bool:
        """False once the loop must stop regardless of what the LLM wants next."""
        return (
            self.iteration_count < self.max_iterations
            and len(self.tool_calls) < self.max_tool_calls
        )

    def summary(self) -> dict[str, Any]:
        """Compact snapshot for logging / the live trace UI."""
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "iteration": self.iteration_count,
            "tool_calls": len(self.tool_calls),
            "files_examined": len(self.files_examined),
            "errors": len(self.errors),
            "recovery_attempts": self.recovery_attempts,
        }