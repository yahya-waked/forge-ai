"""
Code search — Tool 3 from the project spec.

Phase 2 implementation: a plain keyword/regex scan across the repository's
text files. This intentionally does not try to be smart about relevance —
Phase 5 adds the Hybrid Retrieval system (BM25 + vector search over
chunked code) on top of/instead of this. For now the agent just needs a
reliable way to find where something is mentioned.
"""

from __future__ import annotations

import os
import re

from app.tools.registry import RiskLevel, ToolError, registry

_IGNORED_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".ruff_cache",
}
_MAX_MATCHES = 50
_MAX_FILE_BYTES = 500_000  # skip huge files (usually generated/binary-ish)


@registry.register(
    name="search_code",
    description=(
        "Search the repository's text files for a keyword or regular expression. "
        "Returns matching file paths, line numbers, and the matching line. Use this "
        "to locate where a function, class, string, or error message appears before "
        "reading the full file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "query": {
                "type": "string",
                "description": "Text or regular expression to search for.",
            },
            "regex": {
                "type": "boolean",
                "description": "Treat 'query' as a regular expression instead of plain text.",
                "default": False,
            },
            "file_extension": {
                "type": "string",
                "description": "Optional filter, e.g. '.py' to search only Python files.",
            },
        },
        "required": ["repo_root", "query"],
    },
    risk=RiskLevel.LOW,
)
def search_code(
    repo_root: str,
    query: str,
    regex: bool = False,
    file_extension: str | None = None,
) -> list[str]:
    root = os.path.abspath(repo_root)
    if not os.path.isdir(root):
        raise ToolError(f"'{repo_root}' is not a valid repository root")

    try:
        pattern = re.compile(query) if regex else re.compile(re.escape(query))
    except re.error as exc:
        raise ToolError(f"Invalid regex '{query}': {exc}") from exc

    matches: list[str] = []

    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _IGNORED_DIRS]

        for name in sorted(files):
            if file_extension and not name.endswith(file_extension):
                continue

            full_path = os.path.join(current_root, name)
            try:
                if os.path.getsize(full_path) > _MAX_FILE_BYTES:
                    continue
                with open(full_path, "r", encoding="utf-8", errors="strict") as f:
                    for line_no, line in enumerate(f, start=1):
                        if pattern.search(line):
                            rel_path = os.path.relpath(full_path, root)
                            matches.append(f"{rel_path}:{line_no}: {line.strip()}")
                            if len(matches) >= _MAX_MATCHES:
                                return matches
            except (UnicodeDecodeError, OSError):
                continue  # skip binary or unreadable files

    return matches