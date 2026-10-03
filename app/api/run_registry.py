"""
Run Registry — Phase 9 of the project spec.

Holds one RunRecord per orchestrator run: the live event trace (for the
dashboard's Live Trace panel) and a blocking approval handler that
pauses the orchestrator's background thread until a human clicks
Approve/Reject in the browser, instead of terminal_approval_handler's
blocking input().

This is intentionally in-memory and single-process — fine for a local
dashboard; Phase 10's Observability work would swap this for something
durable if ForgeAI ever runs as a real multi-user service.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.agents.orchestrator import Orchestrator, OrchestratorResult
from app.runtime.events import AgentEvent, EventBus
from app.safety.approval import ApprovalDecision, ApprovalRequest


def _json_safe(value: Any) -> Any:
    """Best-effort conversion of arbitrary event data to something JSON can encode."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class RunRecord:
    run_id: str
    task: str
    repository: str
    status: str = "starting"  # starting | running | awaiting_approval | done | error
    events: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    _approval_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _approval_decision: ApprovalDecision | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def create(self, task: str, repository: str) -> RunRecord:
        record = RunRecord(run_id=str(uuid.uuid4()), task=task, repository=repository)
        self._runs[record.run_id] = record
        return record

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def list_summaries(self) -> list[dict[str, Any]]:
        return [
            {"run_id": r.run_id, "task": r.task, "status": r.status}
            for r in reversed(list(self._runs.values()))
        ]

    # ---- called from the orchestrator's background thread ----

    def event_listener(self, run_id: str):
        def _listener(event: AgentEvent) -> None:
            record = self.get(run_id)
            if record is None:
                return
            with record._lock:
                record.events.append(
                    {
                        "type": event.type.value,
                        "message": event.message,
                        "data": _json_safe(event.data),
                        "at": event.at.isoformat(),
                    }
                )

        return _listener

    def approval_handler(self, run_id: str):
        def _handler(request: ApprovalRequest) -> ApprovalDecision:
            record = self.get(run_id)
            if record is None:
                return ApprovalDecision(approved=False, note="run no longer exists")

            with record._lock:
                record.pending_approval = {
                    "tool_name": request.tool_name,
                    "arguments": _json_safe(request.arguments),
                    "reason": request.reason,
                }
                record.status = "awaiting_approval"
                record._approval_event.clear()

            record._approval_event.wait()  # blocks this run's background thread only

            with record._lock:
                decision = record._approval_decision or ApprovalDecision(approved=False)
                record.pending_approval = None
                record.status = "running"
            return decision

        return _handler

    def submit_approval(self, run_id: str, approved: bool, note: str = "") -> bool:
        record = self.get(run_id)
        if record is None or record.pending_approval is None:
            return False
        with record._lock:
            record._approval_decision = ApprovalDecision(approved=approved, note=note)
        record._approval_event.set()
        return True

    # ---- run lifecycle ----

    def run_in_background(self, run_id: str) -> None:
        record = self.get(run_id)
        if record is None:
            return
        record.status = "running"

        try:
            event_bus = EventBus()
            event_bus.subscribe(self.event_listener(run_id))
            orchestrator = Orchestrator(
                events=event_bus,
                approval_handler=self.approval_handler(run_id),
            )
            result: OrchestratorResult = orchestrator.run(record.task, record.repository)
            record.result = {
                "status": result.status,
                "summary": result.summary,
                "review": result.review,
                "security": result.security,
                "commit": result.commit,
                "pr_url": result.pr_url,
                "rounds": result.rounds,
            }
            record.status = "done"
        except Exception as exc:  # a crashed run must be visible on the dashboard, not silent
            record.error = str(exc)
            record.status = "error"

    def start(self, task: str, repository: str) -> RunRecord:
        record = self.create(task, repository)
        thread = threading.Thread(target=self.run_in_background, args=(record.run_id,), daemon=True)
        thread.start()
        return record


registry = RunRegistry()