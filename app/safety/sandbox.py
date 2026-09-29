"""
Docker Sandbox — Phase 7 of the project spec (section 12).

Isolates test/lint command execution from the host: no network, no
credentials, no filesystem access beyond the mounted repo, capped
CPU/memory/process count, and a non-root user. This is the real fix for
the class of vulnerability found during Phase 6 testing — a test file
that spawns arbitrary processes now runs inside a disposable container
with nothing to exfiltrate and nothing to damage, instead of directly on
the developer's machine.

Honest limitation: the sandbox image only has what run_tests/run_linter
need pre-installed (pytest, ruff, mypy) — see docker/Dockerfile for how
to extend it for a repository with its own extra dependencies. Deps are
fixed at image *build* time, when network access is deliberately
allowed; the actual test/lint run never gets network at all.
"""

from __future__ import annotations

import os
import subprocess

_IMAGE_NAME = "forgeai-sandbox:latest"
_DOCKER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "docker")
)
_MEMORY_LIMIT = "512m"
_CPU_LIMIT = "1"
_PIDS_LIMIT = "128"


class SandboxError(Exception):
    """Docker is missing/unreachable, the image failed to build, or the run timed out."""


def _docker_available() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_image_built() -> None:
    """Builds the sandbox image once; a no-op on every later call."""
    if not _docker_available():
        raise SandboxError(
            "Docker is not running or not installed. Sandboxed execution requires "
            "Docker Desktop to be running — start it and try again."
        )

    check = subprocess.run(
        ["docker", "image", "inspect", _IMAGE_NAME], capture_output=True, text=True
    )
    if check.returncode == 0:
        return  # already built

    build = subprocess.run(
        ["docker", "build", "-t", _IMAGE_NAME, _DOCKER_DIR],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise SandboxError(f"Failed to build the sandbox image:\n{build.stdout}\n{build.stderr}")


def run_in_sandbox(argv: list[str], repo_root: str, timeout: int) -> tuple[int, str]:
    """Runs argv inside the sandbox container, working directory = the
    mounted repo. Returns (returncode, combined stdout+stderr)."""
    ensure_image_built()

    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--memory", _MEMORY_LIMIT,
        "--cpus", _CPU_LIMIT,
        "--pids-limit", _PIDS_LIMIT,
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-v", f"{os.path.abspath(repo_root)}:/workspace",
        "-w", "/workspace",
        _IMAGE_NAME,
        *argv,
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"Sandboxed command timed out after {timeout}s") from exc
    except FileNotFoundError as exc:
        raise SandboxError("The 'docker' CLI is not on PATH") from exc

    output = (result.stdout or "") + (result.stderr or "")
    # /workspace only exists inside the container — rewrite it back to a plain
    # relative path so it matches what every other tool reports, and so the
    # agent can never mistake it for a real path on the host in a later call.
    output = output.replace("/workspace/", "").replace("/workspace", ".")

    return result.returncode, output