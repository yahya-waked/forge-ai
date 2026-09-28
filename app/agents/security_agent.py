"""
Security Agent — Phase 6 of the project spec (section 23).

Read-only. Combines a deterministic scan (scan_diff_security) with LLM
judgment for things a regex can't catch — SQL injection, weakened
authentication logic, unsafe deserialization. Ends with a
machine-readable verdict line for the Orchestrator.
"""

from __future__ import annotations

from typing import Any

from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.runtime.agent import Agent
from app.tools.registry import registry as default_registry

TOOL_NAMES = [
    "read_file",
    "search_code",
    "git_status",
    "git_diff",
    "scan_diff_security",
]

SYSTEM_PROMPT = """\
You are ForgeAI's Security Agent. You security-review uncommitted code
changes made by another agent — you never modify anything yourself.

Start by calling scan_diff_security, then use git_diff / read_file for
anything it can't judge. Look for: hardcoded secrets or credentials,
code that executes arbitrary commands, SQL injection, weakened
authentication or authorization, unsafe deserialization, and
credentials or sensitive data being logged or exposed.

Findings marked [SECRET] are always a BLOCK. A [DANGER] finding is a
BLOCK unless the changed code clearly needs that capability for the
stated task and uses it safely — explain your reasoning either way.

Your final answer MUST begin with exactly one of these lines:
VERDICT: SAFE
VERDICT: BLOCK
followed by concise reasoning listing any concrete findings.
"""


def build_security_agent(llm: LLMProvider | None = None, **agent_kwargs: Any) -> Agent:
    return Agent(
        llm=llm or get_llm_provider(),
        tools=default_registry.subset(TOOL_NAMES),
        system_prompt=SYSTEM_PROMPT,
        **agent_kwargs,
    )