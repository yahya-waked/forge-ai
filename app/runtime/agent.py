"""
Agent — the thin, user-facing wrapper around the Agent Loop.

`Agent.run()` seeds the conversation (system prompt + task) then hands
control to `run_agent_loop`. Later (Phase 6), an Orchestrator will
instantiate one Agent per specialized role (Repository Agent, Coding
Agent, ...); for now one Agent handles everything since we only have
generic tools.
"""

from __future__ import annotations

from app.llm.base import LLMProvider, Message, Role
from app.llm.factory import get_llm_provider
from app.runtime.events import EventBus, EventType
from app.runtime.loop import run_agent_loop
from app.runtime.state import AgentState, RunStatus
from app.safety.approval import ApprovalHandler, terminal_approval_handler
from app.tools.registry import ToolRegistry
from app.tools.registry import registry as default_registry

DEFAULT_SYSTEM_PROMPT = """\
You are ForgeAI, an autonomous software engineering agent.
You solve real software tasks against a real repository by calling tools —
you never invent file contents or test results, you retrieve and verify
them through the tools available to you.
Work step by step: inspect before you modify, and briefly explain your
reasoning before each tool call. For a repository you haven't explored yet,
consider calling index_repository once, then use search_code_semantic for
meaning-based lookups (e.g. "where is X handled") alongside search_code for
exact string/regex matches. Before making code changes, create a new git
branch for the work (git_create_branch) rather than committing directly on
the current branch. For any git operation, use the provided git_status/
git_diff/git_create_branch/git_commit tools directly — never write a script
that shells out to git or any other command (e.g. via subprocess, os.system,
or shell=True). Writing such a script is unsafe: it can execute arbitrary
commands later when run_tests runs it, without going through approval again.
After changing code, always run the test
suite to verify your change. If tests fail, analyze the failure output,
make a targeted corrective change, and re-run the tests — do not give up
after one failed attempt, but also do not repeat the same failing change.
When the task is fully done (tests passing, or you have made your best
attempt and clearly explained what remains broken), reply with a final
answer and do not call any more tools.
"""


class Agent:
    def __init__(
        self,
        llm: LLMProvider | None = None,
        tools: ToolRegistry | None = None,
        events: EventBus | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        approval_handler: ApprovalHandler = terminal_approval_handler,
    ) -> None:
        self.llm = llm or get_llm_provider()
        self.tools = tools or default_registry
        self.events = events or EventBus()
        self.system_prompt = system_prompt
        self.approval_handler = approval_handler

    def run(self, task: str, repository: str) -> AgentState:
        state = AgentState(task=task, repository=repository)
        state.status = RunStatus.ANALYZING

        state.messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        state.messages.append(
            Message(role=Role.USER, content=f"Repository: {repository}\n\nTask:\n{task}")
        )

        self.events.emit_new(state.run_id, EventType.TASK_RECEIVED, f"Task received: {task}")

        return run_agent_loop(state, self.llm, self.tools, self.events, self.approval_handler)