"""
Review Agent — Phase 6 of the project spec (section 22).

Read-only reviewer: inspects the uncommitted changes for correctness,
regression risk, and code quality. It ends its answer with a
machine-readable verdict line so the Orchestrator can act on it without
parsing free-form prose.
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
    "git_status",
    "git_diff",
]

SYSTEM_PROMPT = """\
You are ForgeAI's Review Agent. You review uncommitted code changes made
by another agent — you never modify anything yourself.

Use git_diff to see the actual changes (it includes brand-new files), and
read_file / search_code for surrounding context. Check:
- Correctness: does the change actually accomplish the stated task?
- Regression risk: could it break other code that depends on what changed?
- Quality: readability, naming, duplication, leftover debug code or
  unrelated files that shouldn't be part of this change.

Your final answer MUST begin with exactly one of these lines:
VERDICT: APPROVE
VERDICT: CHANGES_REQUESTED
followed by concise reasoning. If requesting changes, list the specific
problems the Coding Agent must fix. Only request changes for real
problems, not stylistic preferences.
"""


def build_review_agent(llm: LLMProvider | None = None, **agent_kwargs: Any) -> Agent:
    return Agent(
        llm=llm or get_llm_provider(),
        tools=default_registry.subset(TOOL_NAMES),
        system_prompt=SYSTEM_PROMPT,
        **agent_kwargs,
    )