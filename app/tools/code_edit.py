"""
Code editing tools — Phase 3 of the project spec.

`apply_patch` is deliberately NOT a unified-diff applier. Asking an LLM to
produce a perfectly-formatted unified diff is fragile; asking it to find an
exact block of existing text and give the replacement is far more reliable
(the same approach used by most production coding agents). Both tools are
MEDIUM risk and require human approval before they touch disk.
"""

from __future__ import annotations

import difflib
import os

from app.tools.registry import RiskLevel, ToolError, registry


def _resolve_in_repo(repo_root: str, path: str) -> str:
    root = os.path.abspath(repo_root)
    full = os.path.abspath(os.path.join(root, path))
    if not full.startswith(root):
        raise ToolError(f"Path '{path}' escapes the repository root")
    return full


def _unified_diff(path: str, old: str, new: str) -> str:
    diff_lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    diff_text = "".join(diff_lines)
    return diff_text or f"No changes applied to '{path}'"


@registry.register(
    name="write_file",
    description=(
        "Create a new file, or fully overwrite an existing file's contents. "
        "Prefer apply_patch for editing an existing file — use write_file mainly "
        "for brand-new files, since it replaces the entire file content."
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
                "description": "File path relative to repo_root to create or overwrite.",
            },
            "content": {
                "type": "string",
                "description": "The full new content of the file.",
            },
        },
        "required": ["repo_root", "path", "content"],
    },
    risk=RiskLevel.MEDIUM,
    requires_approval=True,
)
def write_file(repo_root: str, path: str, content: str) -> str:
    full = _resolve_in_repo(repo_root, path)
    existed = os.path.isfile(full)
    old_content = ""
    if existed:
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            old_content = f.read()

    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)

    if existed:
        return _unified_diff(path, old_content, content)
    return f"Created '{path}' ({len(content)} chars)"


@registry.register(
    name="apply_patch",
    description=(
        "Apply a precise, minimal edit to an existing file by replacing one exact "
        "block of existing text with new text. 'old_text' must match the file's "
        "current content exactly and must appear exactly once — include enough "
        "surrounding lines to make it unique. Prefer this over write_file when "
        "editing an existing file."
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
                "description": "File path relative to repo_root to edit.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact existing text to replace. Must match exactly once in the file.",
            },
            "new_text": {
                "type": "string",
                "description": "Text to replace old_text with.",
            },
        },
        "required": ["repo_root", "path", "old_text", "new_text"],
    },
    risk=RiskLevel.MEDIUM,
    requires_approval=True,
)
def apply_patch(repo_root: str, path: str, old_text: str, new_text: str) -> str:
    full = _resolve_in_repo(repo_root, path)
    if not os.path.isfile(full):
        raise ToolError(f"'{path}' is not a file in this repository")

    with open(full, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    occurrences = content.count(old_text)
    if occurrences == 0:
        raise ToolError(
            f"old_text not found in '{path}' — it must match the file's current content exactly"
        )
    if occurrences > 1:
        raise ToolError(
            f"old_text matches {occurrences} places in '{path}' — include more "
            "surrounding context to make it unique"
        )

    new_content = content.replace(old_text, new_text, 1)
    with open(full, "w", encoding="utf-8") as f:
        f.write(new_content)

    return _unified_diff(path, content, new_content)