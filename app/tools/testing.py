"""
Testing & linting tools — Phase 4 of the project spec.

Both tools shell out to a fixed allowlist of commands (section 11 of the
spec) so the agent can never run an arbitrary command — only the specific
test/lint runners explicitly enabled here. Every invocation runs with a
hard timeout, captured output, and shell=False (no shell metacharacter
interpretation, no injection via extra_args).
"""

from __future__ import annotations

import os
import shlex
import subprocess

from app.safety.sandbox import SandboxError, run_in_sandbox
from app.tools.registry import RiskLevel, ToolError, registry

# Command name -> the fixed argv it expands to. The agent may add extra
# arguments (e.g. a specific test path) but can never change the base
# command itself or step outside this list.
_ALLOWED_COMMANDS: dict[str, list[str]] = {
    "pytest": ["pytest"],
    "ruff": ["ruff", "check"],
    "mypy": ["mypy"],
}

_DEFAULT_TIMEOUT_SECONDS = 120
_MAX_OUTPUT_CHARS = 8000
_SANDBOX_ENV_VAR = "FORGEAI_SANDBOX"  # set to "off" to run on the host instead (not recommended)


def _run_on_host(argv: list[str], repo_root: str, timeout: int) -> tuple[int, str]:
    try:
        result = subprocess.run(
            argv,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise ToolError(f"'{argv[0]}' is not installed or not on PATH in this environment") from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"'{argv[0]}' timed out after {timeout}s") from exc
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _run_allowlisted(repo_root: str, command: str, extra_args: str, timeout: int) -> str:
    if command not in _ALLOWED_COMMANDS:
        raise ToolError(
            f"Command '{command}' is not on the allowlist. Allowed: {sorted(_ALLOWED_COMMANDS)}"
        )

    argv = list(_ALLOWED_COMMANDS[command])
    if extra_args:
        argv.extend(shlex.split(extra_args))

    if os.environ.get(_SANDBOX_ENV_VAR, "on").lower() == "off":
        returncode, output = _run_on_host(argv, repo_root, timeout)
    else:
        try:
            returncode, output = run_in_sandbox(argv, repo_root, timeout)
        except SandboxError as exc:
            raise ToolError(
                f"Sandboxed execution unavailable: {exc}\n"
                f"Set {_SANDBOX_ENV_VAR}=off to run on the host instead (not recommended — "
                "this removes the isolation that protects against a test file running "
                "arbitrary commands)."
            ) from exc

    status = "PASSED" if returncode == 0 else "FAILED"
    if len(output) > _MAX_OUTPUT_CHARS:
        cut = len(output) - _MAX_OUTPUT_CHARS
        output = output[:_MAX_OUTPUT_CHARS] + f"\n... [truncated, {cut} more chars]"

    return f"{status} (exit code {returncode})\n{output}"


@registry.register(
    name="run_tests",
    description=(
        "Run the project's test suite (pytest) and return PASSED/FAILED plus the "
        "output, including any failure tracebacks. Optionally target specific tests "
        "via extra_args, e.g. 'tests/test_auth.py -k expired'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "extra_args": {
                "type": "string",
                "description": "Extra pytest arguments, e.g. a path or '-k <expr>'.",
                "default": "",
            },
        },
        "required": ["repo_root"],
    },
    risk=RiskLevel.MEDIUM,
    requires_approval=False,  # read-only w.r.t. the codebase — safe to run unattended
)
def run_tests(repo_root: str, extra_args: str = "") -> str:
    return _run_allowlisted(repo_root, "pytest", extra_args, _DEFAULT_TIMEOUT_SECONDS)


@registry.register(
    name="run_linter",
    description=(
        "Run a static analysis / lint check and return PASSED/FAILED plus the output. "
        "tool selects which linter: 'ruff' (default) or 'mypy'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "tool": {
                "type": "string",
                "description": "Which linter to run.",
                "enum": ["ruff", "mypy"],
                "default": "ruff",
            },
            "extra_args": {
                "type": "string",
                "description": "Extra arguments to the linter, e.g. a specific path.",
                "default": "",
            },
        },
        "required": ["repo_root"],
    },
    risk=RiskLevel.MEDIUM,
    requires_approval=False,
)
def run_linter(repo_root: str, tool: str = "ruff", extra_args: str = "") -> str:
    if tool not in ("ruff", "mypy"):
        raise ToolError(f"Unknown linter '{tool}' — choose 'ruff' or 'mypy'")
    return _run_allowlisted(repo_root, tool, extra_args, _DEFAULT_TIMEOUT_SECONDS)