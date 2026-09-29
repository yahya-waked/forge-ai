"""
The Agent Loop — the concrete implementation of:

    Agent State -> LLM Decision -> Tool Call -> Validation
    -> Tool Execution -> Observation -> Update State -> LLM Decision Again -> ...

This is the ReAct loop (section 7 of the project spec): the LLM never
executes a tool itself, it only ever says "I want to call X with these
arguments". This module is the only place that turns that request into
a real Python function call.
"""

from __future__ import annotations

import re

from app.llm.base import LLMProvider, Message, Role
from app.runtime.events import EventBus, EventType
from app.runtime.state import AgentState, RunStatus, TestResult, ToolCallRecord
from app.safety.approval import ApprovalHandler, ApprovalRequest, terminal_approval_handler
from app.safety.content_scan import scan_for_danger
from app.tools.registry import ToolError, ToolRegistry


class LoopStoppedError(Exception):
    """Raised when the loop stops without a final answer (budget exhausted).
    The caller decides what happens next — escalate to a human, or hand off
    to the Recovery Loop once Phase 4 exists."""


def _parse_test_result(observation: str) -> TestResult:
    """
    Best-effort parse of our run_tests tool's output (which starts with
    'PASSED'/'FAILED (exit code N)' followed by raw pytest output). Good
    enough to drive the recovery loop now; Phase 11's Evaluation system
    can replace this with structured pytest-json-report parsing later.
    """
    passed = observation.startswith("PASSED")
    failed_match = re.search(r"(\d+)\s+failed", observation)
    passed_match = re.search(r"(\d+)\s+passed", observation)
    failed = int(failed_match.group(1)) if failed_match else (0 if passed else 1)
    passed_count = int(passed_match.group(1)) if passed_match else 0
    return TestResult(passed=passed, total=failed + passed_count, failed=failed, output=observation)


def run_agent_loop(
    state: AgentState,
    llm: LLMProvider,
    tools: ToolRegistry,
    events: EventBus | None = None,
    approval_handler: ApprovalHandler = terminal_approval_handler,
) -> AgentState:
    """
    Drives `state` forward until the LLM returns a final answer with no
    further tool calls, or the iteration/tool-call budget runs out.
    Mutates and returns `state`.
    """
    events = events or EventBus()

    while True:
        if not state.has_budget():
            state.status = RunStatus.FAILED
            events.emit_new(
                state.run_id,
                EventType.RUN_FAILED,
                "Stopped: iteration or tool-call budget exhausted",
            )
            raise LoopStoppedError(
                f"Budget exhausted after {state.iteration_count} iterations "
                f"and {len(state.tool_calls)} tool calls"
            )

        state.iteration_count += 1

        response = llm.generate(messages=state.messages, tools=tools.schemas())

        # Record the assistant turn (text and/or tool calls) in the conversation.
        state.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
                raw=response.provider_message,
            )
        )

        if not response.wants_tool_call:
            # No more tools requested -> the LLM considers the task done.
            state.final_result = response.content
            state.status = RunStatus.DONE
            events.emit_new(state.run_id, EventType.RUN_FINISHED, "Agent produced a final answer")
            return state

        events.emit_new(
            state.run_id,
            EventType.LLM_DECISION,
            f"Requested {len(response.tool_calls)} tool call(s)",
            tool_calls=[c.name for c in response.tool_calls],
        )

        for call in response.tool_calls:
            record = ToolCallRecord(id=call.id, tool_name=call.name, arguments=call.arguments)

            try:
                tool = tools.get(call.name)
            except ToolError as exc:
                observation = f"ERROR: {exc}"
                record.error = str(exc)
                state.errors.append(observation)
                events.emit_new(state.run_id, EventType.TOOL_CALL_FAILED, observation)
                state.record_tool_call(record)
                state.add_observation(observation)
                state.messages.append(
                    Message(role=Role.TOOL, content=observation, tool_call_id=call.id, name=call.name)
                )
                continue

            if tool.requires_approval:
                events.emit_new(
                    state.run_id, EventType.APPROVAL_REQUIRED, f"Approval required for {call.name}"
                )
                danger_reasons = scan_for_danger(
                    " ".join(v for v in call.arguments.values() if isinstance(v, str))
                )
                reason = (
                    "DANGER: this content " + "; ".join(danger_reasons) + ". It can execute "
                    "arbitrary commands, and if it's a test file, running it later via "
                    "run_tests will NOT ask for approval again."
                    if danger_reasons
                    else ""
                )
                decision = approval_handler(
                    ApprovalRequest(
                        run_id=state.run_id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        reason=reason,
                    )
                )
                events.emit_new(
                    state.run_id,
                    EventType.APPROVAL_RESOLVED,
                    f"{call.name} {'approved' if decision.approved else 'rejected'}",
                )
                if not decision.approved:
                    observation = f"REJECTED by human reviewer: {call.name} was not approved."
                    record.error = "rejected_by_human"
                    state.record_tool_call(record)
                    state.add_observation(observation)
                    state.messages.append(
                        Message(role=Role.TOOL, content=observation, tool_call_id=call.id, name=call.name)
                    )
                    continue

            events.emit_new(
                state.run_id, EventType.TOOL_CALL_STARTED, f"Calling {call.name}({call.arguments})"
            )

            try:
                result = tool.executor(**call.arguments)
                record.result = result
                events.emit_new(
                    state.run_id, EventType.TOOL_CALL_FINISHED, f"{call.name} finished", result=result
                )
                observation = str(result)
            except ToolError as exc:
                record.error = str(exc)
                events.emit_new(state.run_id, EventType.TOOL_CALL_FAILED, f"{call.name} failed: {exc}")
                observation = f"ERROR: {exc}"
                state.errors.append(observation)
            except TypeError as exc:
                record.error = str(exc)
                observation = f"ERROR: bad arguments for '{call.name}': {exc}"
                state.errors.append(observation)
            except Exception as exc:  # noqa: BLE001 — a buggy/unexpected tool failure
                # must become feedback for the agent, never crash the whole run.
                record.error = str(exc)
                observation = f"ERROR: unexpected failure in '{call.name}': {exc}"
                state.errors.append(observation)
                events.emit_new(state.run_id, EventType.TOOL_CALL_FAILED, observation)

            state.record_tool_call(record)
            state.add_observation(observation)

            if call.name == "read_file" and record.error is None:
                path = call.arguments.get("path")
                if path:
                    state.mark_file_examined(path)

            # Feed the observation back to the LLM as a tool message.
            state.messages.append(
                Message(role=Role.TOOL, content=observation, tool_call_id=call.id, name=call.name)
            )

            if call.name == "run_tests" and record.error is None:
                test_result = _parse_test_result(observation)
                state.test_results.append(test_result)
                events.emit_new(
                    state.run_id,
                    EventType.TEST_RESULT,
                    "Test run finished",
                    passed=test_result.passed,
                    failed=test_result.failed,
                    total=test_result.total,
                )

                if test_result.passed:
                    state.status = RunStatus.TESTING
                else:
                    state.recovery_attempts += 1
                    state.status = RunStatus.RECOVERING
                    events.emit_new(
                        state.run_id,
                        EventType.RECOVERY_STARTED,
                        f"Tests failed (recovery attempt {state.recovery_attempts}/"
                        f"{state.max_recovery_attempts}) — agent will attempt a fix",
                    )
                    if state.recovery_attempts > state.max_recovery_attempts:
                        state.status = RunStatus.FAILED
                        state.final_result = (
                            f"Stopped after {state.recovery_attempts} failed recovery "
                            "attempts — tests are still failing and this needs human review."
                        )
                        events.emit_new(
                            state.run_id, EventType.RUN_FAILED, "Recovery attempts exhausted"
                        )
                        return state