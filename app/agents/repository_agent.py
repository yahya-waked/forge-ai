"""
Repository Agent — Phase 6 of the project spec.

Read-only: its whole job is to understand the codebase and report back
useful context (relevant files, how things fit together) for the Coding
Agent to act on. Every tool it has access to is LOW risk and none
require approval — Least Privilege in practice, not just in the spec,
enforced by only ever handing it a restricted ToolRegistry.
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
]

SYSTEM_PROMPT = """\
You are ForgeAI's Repository Agent. Your only job is to understand a
repository well enough to brief another agent that will make code
changes — you never modify anything yourself.

Explore using list_files, read_file, and search_code, and (after calling
index_repository once) search_code_semantic for meaning-based lookups.
When you're done, reply with a concise final answer covering: which
files are relevant to the task, how they relate to each other, and
anything a Coding Agent should know before changing code (existing
conventions, related tests, things NOT to touch). Do not propose the
actual fix — that is not your job.
"""


def build_repository_agent(llm: LLMProvider | None = None, **agent_kwargs: Any) -> Agent:
    return Agent(
        llm=llm or get_llm_provider(),
        tools=default_registry.subset(TOOL_NAMES),
        system_prompt=SYSTEM_PROMPT,
        **agent_kwargs,
    )