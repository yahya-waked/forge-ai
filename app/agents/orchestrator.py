"""
Orchestrator — Phase 6 of the project spec (sections 5 and 6).

Pipeline:

    Repository Agent  ->  Coding Agent  ->  Review Agent + Security Agent
                              ^                        |
                              |__ feedback (bounded) __|
                                                       |
                                          Human Approval Gate -> git_commit

The Orchestrator is plain Python, not an LLM: control flow (who runs
next, when to stop, when a human must approve) is deterministic and
auditable. Only the individual stages use LLMs. Notably, the commit is
performed by the Orchestrator itself — no agent can commit on its own.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

import app.tools  # noqa: F401  (registers all tools with the shared registry)
from app.agents.coding_agent import build_coding_agent
from app.agents.repository_agent import build_repository_agent
from app.agents.review_agent import build_review_agent
from app.agents.security_agent import build_security_agent
from app.github.client import GitHubError, resolve_owner_repo
from app.llm.base import LLMProvider, Message, Role
from app.llm.factory import get_llm_provider
from app.runtime.events import EventBus, EventType
from app.runtime.loop import LoopStoppedError
from app.runtime.state import AgentState, RunStatus
from app.safety.approval import ApprovalHandler, ApprovalRequest, terminal_approval_handler
from app.tools.registry import ToolError
from app.tools.registry import registry as default_registry

NO_CHANGES_MARKER = "No uncommitted changes."


@dataclass
class OrchestratorResult:
    status: str  # committed | commit_rejected | changes_not_approved | no_changes | stage_failed
    summary: str
    repo_context: str = ""
    coding_summary: str = ""
    review: str = ""
    security: str = ""
    commit: str | None = None
    pr_url: str | None = None
    rounds: int = 0
    runs: list[tuple[str, AgentState]] = field(default_factory=list)


def _verdict_is(text: str, positive: str) -> bool:
    """True if the first 'VERDICT:' line in `text` starts with `positive`."""
    for line in text.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            return line.split(":", 1)[1].strip().upper().startswith(positive)
    return False


class Orchestrator:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        events: EventBus | None = None,
        approval_handler: ApprovalHandler = terminal_approval_handler,
        max_review_rounds: int = 2,
        allow_dirty: bool = False,
        publish_pr: bool = True,
    ) -> None:
        self.llm = llm or get_llm_provider()
        self.events = events or EventBus()
        self.approval_handler = approval_handler
        self.max_review_rounds = max_review_rounds
        self.allow_dirty = allow_dirty
        self.publish_pr = publish_pr
        self._agent_kwargs = {"events": self.events, "approval_handler": approval_handler}

    # ------------------------------------------------------------------ helpers

    def _stage(
        self, name: str, agent, task: str, repository: str, result: OrchestratorResult
    ) -> AgentState | None:
        """Runs one agent; returns its final state, or None if it failed/stopped."""
        try:
            state = agent.run(task=task, repository=repository)
        except LoopStoppedError:
            return None
        result.runs.append((name, state))
        return state if state.status == RunStatus.DONE else None

    def _tool(self, name: str, **arguments) -> str:
        """Deterministic tool call made by the Orchestrator itself (no LLM involved)."""
        return str(default_registry.get(name).executor(**arguments))

    def _commit_message(self, task: str, coding_summary: str) -> str:
        try:
            response = self.llm.generate(
                messages=[
                    Message(
                        role=Role.SYSTEM,
                        content=(
                            "Write a single-line conventional commit message (max 72 chars, "
                            "e.g. 'fix: handle expired JWT tokens'). Reply with only the message."
                        ),
                    ),
                    Message(
                        role=Role.USER,
                        content=f"Task: {task}\n\nWhat changed:\n{coding_summary}",
                    ),
                ]
            )
            lines = response.content.strip().splitlines()
            line = lines[0].strip().strip("`\"'") if lines else ""
            if line:
                return line[:72]
        except Exception:  # a commit message must never block the pipeline
            pass
        return f"forgeai: {task[:60]}"

    def _fail(self, result: OrchestratorResult, status: str, summary: str) -> OrchestratorResult:
        result.status = status
        result.summary = summary
        return result

    # --------------------------------------------------------------------- main

    def run(self, task: str, repository: str) -> OrchestratorResult:
        result = OrchestratorResult(status="running", summary="")

        # 0) Clean-baseline check. git_commit stages everything, so if the working
        # tree already has uncommitted work, the agent's commit would silently
        # swallow it — and the reviewers couldn't tell agent changes from yours.
        if not self.allow_dirty:
            if self._tool("git_diff", repo_root=repository).strip() != NO_CHANGES_MARKER:
                return self._fail(
                    result,
                    "dirty_worktree",
                    "The repository has uncommitted changes that predate this task, and the "
                    "agent's commit would include them. Commit them first, or stash them with "
                    "`git stash -u`, then run again (or pass allow_dirty=True to override).\n\n"
                    + self._tool("git_status", repo_root=repository),
                )

        # 1) Repository Agent: understand the codebase (read-only).
        repo_state = self._stage(
            "repository",
            build_repository_agent(self.llm, **self._agent_kwargs),
            f"Analyze this repository to prepare for the following task.\n\nTask: {task}",
            repository,
            result,
        )
        if repo_state is None:
            return self._fail(result, "stage_failed", "Repository analysis did not complete.")
        result.repo_context = repo_state.final_result or ""

        feedback = ""
        for round_number in range(1, self.max_review_rounds + 1):
            result.rounds = round_number

            # 2) Coding Agent: make the change and run the recovery loop.
            if round_number == 1:
                coding_task = (
                    f"{task}\n\nContext from the Repository Agent's analysis:\n{result.repo_context}"
                )
            else:
                coding_task = (
                    f"{task}\n\nThis is revision round {round_number}. You are already on the "
                    "working branch and your earlier changes are still uncommitted — do NOT "
                    "create another branch. Fix the problems raised below, re-run the tests, "
                    f"then summarize what you changed.\n\n{feedback}"
                )
            coding_state = self._stage(
                "coding",
                build_coding_agent(self.llm, **self._agent_kwargs),
                coding_task,
                repository,
                result,
            )
            if coding_state is None:
                return self._fail(result, "stage_failed", "The Coding Agent did not complete.")
            result.coding_summary = coding_state.final_result or ""

            diff = self._tool("git_diff", repo_root=repository)
            if diff.strip() == NO_CHANGES_MARKER:
                return self._fail(result, "no_changes", "The Coding Agent made no changes.")

            # 3) Review Agent + Security Agent (both read-only).
            review_state = self._stage(
                "review",
                build_review_agent(self.llm, **self._agent_kwargs),
                (
                    "Review the uncommitted changes in this repository.\n\n"
                    f"Original task: {task}\n\nThe Coding Agent reports:\n{result.coding_summary}"
                ),
                repository,
                result,
            )
            security_state = self._stage(
                "security",
                build_security_agent(self.llm, **self._agent_kwargs),
                (
                    "Security-review the uncommitted changes in this repository.\n\n"
                    f"Original task: {task}"
                ),
                repository,
                result,
            )
            if review_state is None or security_state is None:
                return self._fail(result, "stage_failed", "Review or security stage did not complete.")

            result.review = review_state.final_result or ""
            result.security = security_state.final_result or ""

            # Deterministic guardrail: a [SECRET] finding blocks regardless of LLM opinion.
            scan = self._tool("scan_diff_security", repo_root=repository)
            secret_found = "[SECRET]" in scan

            review_ok = _verdict_is(result.review, "APPROVE")
            security_ok = _verdict_is(result.security, "SAFE") and not secret_found

            if review_ok and security_ok:
                break

            feedback = ""
            if not review_ok:
                feedback += f"Review Agent feedback:\n{result.review}\n\n"
            if not security_ok:
                feedback += f"Security findings:\n{result.security}\n\nAutomated scan:\n{scan}\n"
        else:
            return self._fail(
                result,
                "changes_not_approved",
                f"Review/security did not approve after {self.max_review_rounds} rounds. "
                "Changes are left uncommitted for a human to inspect.",
            )

        # 4) Human Approval Gate -> commit. Only the Orchestrator can commit.
        message = self._commit_message(task, result.coding_summary)
        status_text = self._tool("git_status", repo_root=repository)
        run_id = str(uuid.uuid4())

        self.events.emit_new(run_id, EventType.APPROVAL_REQUIRED, "Approval required for git_commit")
        decision = self.approval_handler(
            ApprovalRequest(
                run_id=run_id,
                tool_name="git_commit",
                arguments={"repo_root": repository, "message": message},
                reason=(
                    f"Review: APPROVED, Security: SAFE (after {result.rounds} round(s)).\n"
                    f"   {status_text.replace(chr(10), chr(10) + '   ')}"
                ),
            )
        )
        self.events.emit_new(
            run_id,
            EventType.APPROVAL_RESOLVED,
            f"git_commit {'approved' if decision.approved else 'rejected'}",
        )

        if not decision.approved:
            return self._fail(
                result, "commit_rejected", "Commit rejected by the human reviewer; changes left uncommitted."
            )

        try:
            result.commit = self._tool("git_commit", repo_root=repository, message=message)
        except ToolError as exc:
            return self._fail(result, "stage_failed", f"Commit failed: {exc}")

        result.status = "committed"
        result.summary = f"{result.commit}\n\nWhat changed:\n{result.coding_summary}"

        if self.publish_pr:
            result.summary += f"\n\n{self._publish_pr(repository, task, result)}"

        return result

    def _publish_pr(self, repository: str, task: str, result: OrchestratorResult) -> str:
        """Pushes the branch and opens a PR. Any missing prerequisite (no GitHub
        remote, no token) is reported as a skip, not a failure — a successful
        local commit should never be undone or hidden by this being optional."""
        try:
            owner, name = resolve_owner_repo(repository)
        except GitHubError as exc:
            return f"(GitHub publishing skipped: {exc})"
        if not os.environ.get("GITHUB_TOKEN"):
            return "(GitHub publishing skipped: GITHUB_TOKEN is not set)"

        run_id = str(uuid.uuid4())
        status_text = self._tool("git_status", repo_root=repository)

        self.events.emit_new(run_id, EventType.APPROVAL_REQUIRED, "Approval required for git_push")
        push_decision = self.approval_handler(
            ApprovalRequest(
                run_id=run_id,
                tool_name="git_push",
                arguments={"repo_root": repository},
                reason=f"Pushing to {owner}/{name}.\n   {status_text.replace(chr(10), chr(10) + '   ')}",
            )
        )
        if not push_decision.approved:
            return "(Push rejected by the human reviewer; commit stays local.)"

        try:
            push_result = self._tool("git_push", repo_root=repository)
        except ToolError as exc:
            return f"(Push failed: {exc})"

        pr_title = (result.commit or task)[:72]
        if pr_title.lower().startswith("committed "):
            pr_title = pr_title.split(": ", 1)[-1]
        pr_body = (
            f"## Task\n{task}\n\n## Changes\n{result.coding_summary}\n\n"
            f"## Review\n{result.review}\n\n## Security\n{result.security}\n\n"
            "---\n*Opened by ForgeAI after automated review and security checks, "
            "and human approval.*"
        )

        self.events.emit_new(
            run_id, EventType.APPROVAL_REQUIRED, "Approval required for create_pull_request"
        )
        pr_decision = self.approval_handler(
            ApprovalRequest(
                run_id=run_id,
                tool_name="create_pull_request",
                arguments={"repo_root": repository, "title": pr_title, "body": pr_body},
                reason=f"Target: {owner}/{name}",
            )
        )
        if not pr_decision.approved:
            return f"{push_result}\n(PR creation rejected by the human reviewer.)"

        try:
            pr_result = self._tool(
                "create_pull_request", repo_root=repository, title=pr_title, body=pr_body
            )
        except ToolError as exc:
            return f"{push_result}\n(PR creation failed: {exc})"

        result.pr_url = pr_result.rsplit(": ", 1)[-1]
        return f"{push_result}\n{pr_result}"