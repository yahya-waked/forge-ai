"""
Human Approval Gate — Phase 7 groundwork, wired in early (Phase 3) because
the first risky tools (write_file, apply_patch) need it from day one.

For now approval is a blocking terminal prompt. The interface
(ApprovalRequest / ApprovalDecision / ApprovalHandler) is what stays
stable — Phase 7/9 swaps `terminal_approval_handler` for one backed by
the real UI's "Approve / Reject" buttons, and nothing in the Agent Loop
has to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ApprovalRequest:
    run_id: str
    tool_name: str
    arguments: dict[str, Any]
    reason: str = ""


@dataclass
class ApprovalDecision:
    approved: bool
    note: str = ""


ApprovalHandler = Callable[[ApprovalRequest], ApprovalDecision]


def terminal_approval_handler(request: ApprovalRequest) -> ApprovalDecision:
    """Blocks on stdin. Fine for local dev; not what ships in the real product."""
    print("\n⚠️  Approval required")
    print(f"   Tool:      {request.tool_name}")
    print(f"   Arguments: {request.arguments}")
    answer = input("   Approve this action? [y/N]: ").strip().lower()
    return ApprovalDecision(approved=answer == "y")


def auto_approve_handler(request: ApprovalRequest) -> ApprovalDecision:
    """Approves everything automatically. Only ever used in automated tests —
    never as a default in agent-facing code."""
    return ApprovalDecision(approved=True, note="auto-approved (test mode)")