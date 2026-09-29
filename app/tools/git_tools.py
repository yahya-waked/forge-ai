"""
Git tools — section 10 of the project spec.

Basic working-tree git operations, independent of the GitHub PR API
(that's Phase 8's job). git_commit is deliberately HIGH risk and
approval-gated: it rewrites repository history, unlike anything else the
agent can do so far.
"""

from __future__ import annotations

import difflib
import os

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo

from app.tools.registry import RiskLevel, ToolError, registry

_MAX_UNTRACKED_BYTES = 100_000


def _get_repo(repo_root: str) -> Repo:
    try:
        return Repo(repo_root)
    except InvalidGitRepositoryError as exc:
        raise ToolError(f"'{repo_root}' is not a git repository (run 'git init' first)") from exc
    except NoSuchPathError as exc:
        raise ToolError(f"'{repo_root}' does not exist on disk") from exc


def collect_working_diff(repo: Repo, path: str | None = None) -> str:
    """Everything that would go into a commit, as one diff: changes to tracked
    files (vs HEAD) PLUS brand-new untracked files shown as full additions.
    Plain `git diff` silently omits new files, which would blind the Review
    and Security agents to exactly the files an agent just created."""
    norm_path = path.replace("\\", "/") if path else None
    parts: list[str] = []

    if repo.head.is_valid():
        tracked = repo.git.diff("HEAD", "--", norm_path) if norm_path else repo.git.diff("HEAD")
    else:
        tracked = repo.git.diff("--", norm_path) if norm_path else repo.git.diff()
    if tracked:
        parts.append(tracked)

    for rel in repo.untracked_files:
        if norm_path and rel != norm_path:
            continue
        full = os.path.join(repo.working_tree_dir, rel)
        try:
            if os.path.getsize(full) > _MAX_UNTRACKED_BYTES:
                parts.append(f"[new file {rel}: too large to display]")
                continue
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, OSError):
            parts.append(f"[new file {rel}: binary or unreadable]")
            continue
        added = "".join(
            difflib.unified_diff(
                [], content.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{rel}"
            )
        )
        parts.append(added or f"[new empty file {rel}]")

    return "\n".join(parts)


@registry.register(
    name="git_status",
    description="Show the working tree status: which tracked files are staged, modified, or untracked.",
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
        },
        "required": ["repo_root"],
    },
    risk=RiskLevel.LOW,
)
def git_status(repo_root: str) -> str:
    repo = _get_repo(repo_root)

    branch = "(detached HEAD)" if repo.head.is_detached else repo.active_branch.name
    modified = [item.a_path for item in repo.index.diff(None)]
    staged = [item.a_path for item in repo.index.diff("HEAD")] if repo.head.is_valid() else []
    untracked = repo.untracked_files

    return (
        f"Branch: {branch}\n"
        f"Staged ({len(staged)}): {', '.join(staged) or '(none)'}\n"
        f"Modified, not staged ({len(modified)}): {', '.join(modified) or '(none)'}\n"
        f"Untracked ({len(untracked)}): {', '.join(untracked) or '(none)'}"
    )


@registry.register(
    name="git_diff",
    description="Show the diff of uncommitted changes in the working tree, optionally limited to one file.",
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "path": {
                "type": "string",
                "description": "Limit the diff to one file, relative to repo_root (optional).",
            },
        },
        "required": ["repo_root"],
    },
    risk=RiskLevel.LOW,
)
def git_diff(repo_root: str, path: str | None = None) -> str:
    repo = _get_repo(repo_root)
    return collect_working_diff(repo, path) or "No uncommitted changes."


@registry.register(
    name="git_create_branch",
    description=(
        "Create a new git branch from the current HEAD and switch to it. Use this before "
        "making code changes, so work happens on an isolated branch instead of directly on "
        "the current branch (e.g. main)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "branch_name": {
                "type": "string",
                "description": "Name for the new branch, e.g. 'forgeai/fix-jwt-expiration'.",
            },
        },
        "required": ["repo_root", "branch_name"],
    },
    risk=RiskLevel.MEDIUM,
    requires_approval=False,
)
def git_create_branch(repo_root: str, branch_name: str) -> str:
    repo = _get_repo(repo_root)
    if branch_name in [h.name for h in repo.heads]:
        raise ToolError(f"Branch '{branch_name}' already exists")

    new_branch = repo.create_head(branch_name)
    new_branch.checkout()
    return f"Created and switched to branch '{branch_name}'"


@registry.register(
    name="git_commit",
    description=(
        "Stage all current changes and commit them with the given message. This rewrites "
        "repository history — a high-risk, approval-gated action. Use git_diff first so the "
        "human reviewer can see what's being committed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "message": {
                "type": "string",
                "description": "Commit message.",
            },
        },
        "required": ["repo_root", "message"],
    },
    risk=RiskLevel.HIGH,
    requires_approval=True,
)
def git_commit(repo_root: str, message: str) -> str:
    repo = _get_repo(repo_root)
    if not repo.is_dirty(untracked_files=True):
        raise ToolError("Nothing to commit — the working tree is clean")

    try:
        repo.git.add(A=True)
        commit = repo.index.commit(message)
    except GitCommandError as exc:
        raise ToolError(f"git commit failed: {exc}") from exc

    return f"Committed {commit.hexsha[:8]}: {message}"