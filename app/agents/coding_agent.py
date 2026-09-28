"""
Coding Agent — Phase 6 of the project spec.

Does the actual work: modifies code and runs tests, using the same
Recovery Loop every agent gets for free from run_agent_loop. Has write
access (MEDIUM risk, approval-gated) plus the ability to create a
branch — but deliberately NOT git_commit, since the Orchestrator only
commits after the Review Agent and Security Agent have signed off.
"""

from __future__ import annotations

from typing import Any

from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.runtime.agent import Agent
from app.tools.registry import registry as default_registry

TOOL_NAMES = [
    "list_files",
    "read_file",
    "search_code",
    "index_repository",
    "search_code_semantic",
    "write_file",
    "apply_patch",
    "run_tests",
    "run_linter",
    "git_status",
    "git_diff",
    "git_create_branch",
]

SYSTEM_PROMPT = """\
You are ForgeAI's Coding Agent. You've been given a task, usually along
with context from a Repository Agent's analysis. Your job: make the
change, then verify it.

Before making changes, create a new git branch for the work
(git_create_branch) rather than working directly on the current branch.
After changing code, always run the test suite (run_tests) to verify
your change. If tests fail, analyze the failure output, make a targeted
corrective change, and re-run the tests — do not give up after one
failed attempt, but also do not repeat the same failing change.

You do not commit anything — a Review Agent and Security Agent check
your work first. Never write a script that shells out to git or any
other command (e.g. via subprocess, os.system, or shell=True); use the
provided tools directly. When the task is fully done (tests passing, or
you have made your best attempt and clearly explained what remains
broken), reply with a final answer summarizing exactly what changed
and why.
"""


def build_coding_agent(llm: LLMProvider | None = None, **agent_kwargs: Any) -> Agent:
    return Agent(
        llm=llm or get_llm_provider(),
        tools=default_registry.subset(TOOL_NAMES),
        system_prompt=SYSTEM_PROMPT,
        **agent_kwargs,
    )