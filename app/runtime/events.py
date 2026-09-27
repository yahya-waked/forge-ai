"""
Agent Events — the stream that powers the Live Agent Trace UI (section 29
of the project spec).

Every meaningful step of the loop (task received, tool call started,
observation, test result, ...) is emitted as an AgentEvent. The runtime
doesn't care who's listening — a terminal logger, a websocket to the
frontend, or nothing at all in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    TASK_RECEIVED = "task_received"
    STATUS_CHANGED = "status_changed"
    LLM_DECISION = "llm_decision"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"
    TOOL_CALL_FAILED = "tool_call_failed"
    OBSERVATION = "observation"
    PLAN_CREATED = "plan_created"
    TEST_RESULT = "test_result"
    RECOVERY_STARTED = "recovery_started"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_RESOLVED = "approval_resolved"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"


@dataclass
class AgentEvent:
    run_id: str
    type: EventType
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


EventListener = Callable[[AgentEvent], None]


class EventBus:
    """Minimal pub/sub so the runtime never imports the API/UI layer directly."""

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []

    def subscribe(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def emit(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            listener(event)

    def emit_new(self, run_id: str, type: EventType, message: str, **data: Any) -> AgentEvent:
        event = AgentEvent(run_id=run_id, type=type, message=message, data=data)
        self.emit(event)
        return event