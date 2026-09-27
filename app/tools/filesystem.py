"""
Filesystem tools — the agent's first real capability: seeing the repository.

Phase 2 of the project spec. Starts with `list_files`; `read_file` and
`search_code` are added the same way in the next batch.
"""

from __future__ import annotations

import os

from app.tools.registry import RiskLevel, ToolError, registry

_IGNORED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".ruff_cache",
}


@registry.register(
    name="list_files",
    description=(
        "List files and directories inside the repository, starting from a given "
        "relative path. Use this first to understand the project layout before "
        "reading or searching specific files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "path": {
                "type": "string",
                "description": "Path relative to repo_root to list. Use '.' for the root.",
                "default": ".",
            },
            "max_depth": {
                "type": "integer",
                "description": "How many directory levels deep to descend.",
                "default": 2,
            },
        },
        "required": ["repo_root"],
    },
    risk=RiskLevel.LOW,
)
def list_files(repo_root: str, path: str = ".", max_depth: int = 2) -> list[str]:
    root = os.path.abspath(repo_root)
    base = os.path.abspath(os.path.join(root, path))

    if not base.startswith(root):
        raise ToolError(f"Path '{path}' escapes the repository root")
    if not os.path.isdir(base):
        raise ToolError(f"'{path}' is not a directory in this repository")

    results: list[str] = []

    for current_root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in _IGNORED_DIRS)

        depth = 0 if current_root == base else os.path.relpath(current_root, base).count(os.sep) + 1
        if depth >= max_depth:
            dirs[:] = []

        rel_root = os.path.relpath(current_root, root)
        for name in sorted(files):
            rel_path = os.path.normpath(os.path.join(rel_root, name))
            results.append(rel_path)

    return sorted(results)


@registry.register(
    name="read_file",
    description=(
        "Read the contents of a single file in the repository. Pass start_line/"
        "end_line (1-indexed) to read a slice of a large file instead of the whole thing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "path": {
                "type": "string",
                "description": "File path relative to repo_root.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-indexed first line to include (optional).",
            },
            "end_line": {
                "type": "integer",
                "description": "1-indexed last line to include (optional).",
            },
        },
        "required": ["repo_root", "path"],
    },
    risk=RiskLevel.LOW,
)
def read_file(
    repo_root: str,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    root = os.path.abspath(repo_root)
    full = os.path.abspath(os.path.join(root, path))

    if not full.startswith(root):
        raise ToolError(f"Path '{path}' escapes the repository root")
    if not os.path.isfile(full):
        raise ToolError(f"'{path}' is not a file in this repository")

    max_bytes = 200_000
    size = os.path.getsize(full)
    if size > max_bytes and start_line is None:
        raise ToolError(
            f"'{path}' is {size} bytes, too large to read in full ({max_bytes} byte limit) "
            f"— pass start_line/end_line to read a slice instead"
        )

    try:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        raise ToolError(f"Could not read '{path}': {exc}") from exc

    if start_line is None and end_line is None:
        return "".join(lines)

    start = max((start_line or 1) - 1, 0)
    end = end_line if end_line is not None else len(lines)
    return "".join(lines[start:end])