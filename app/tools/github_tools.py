"""
GitHub tools — Phase 8 of the project spec.

git_push and create_pull_request are both HIGH risk and approval-gated,
the same class of action as git_commit: externally visible and hard to
undo. Neither is in any specialized agent's ToolRegistry subset — only
the Orchestrator calls them, after a commit has already been approved.
"""

from __future__ import annotations

from github import GithubException

from app.github.client import GitHubError, get_client, resolve_owner_repo
from app.tools.git_tools import _get_repo
from app.tools.registry import RiskLevel, ToolError, registry


@registry.register(
    name="git_push",
    description="Push the current branch to the 'origin' remote on GitHub.",
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
    risk=RiskLevel.HIGH,
    requires_approval=True,
)
def git_push(repo_root: str) -> str:
    repo = _get_repo(repo_root)
    if repo.head.is_detached:
        raise ToolError("Cannot push from a detached HEAD — check out a branch first")

    branch = repo.active_branch.name
    try:
        origin = repo.remotes["origin"]
    except IndexError as exc:
        raise ToolError("No 'origin' remote configured for this repository") from exc

    try:
        push_infos = origin.push(branch)
    except Exception as exc:  # GitCommandError plus auth failures from the credential helper
        raise ToolError(f"git push failed: {exc}") from exc

    summaries = [info.summary.strip() for info in push_infos if info.summary]
    return f"Pushed '{branch}' to origin.\n" + "\n".join(summaries)


@registry.register(
    name="create_pull_request",
    description=(
        "Create a Pull Request on GitHub from the current branch. The branch must "
        "already be pushed (git_push) before calling this. Omit 'base' to target the "
        "repository's default branch."
    ),
    parameters={
        "type": "object",
        "properties": {
            "repo_root": {
                "type": "string",
                "description": "Absolute path to the repository root on disk.",
            },
            "title": {
                "type": "string",
                "description": "Pull request title.",
            },
            "body": {
                "type": "string",
                "description": "Pull request description.",
            },
            "base": {
                "type": "string",
                "description": "Base branch to merge into. Omit for the repo's default branch.",
            },
        },
        "required": ["repo_root", "title", "body"],
    },
    risk=RiskLevel.HIGH,
    requires_approval=True,
)
def create_pull_request(repo_root: str, title: str, body: str, base: str | None = None) -> str:
    repo = _get_repo(repo_root)
    if repo.head.is_detached:
        raise ToolError("Cannot open a PR from a detached HEAD — check out a branch first")
    head_branch = repo.active_branch.name

    try:
        owner, name = resolve_owner_repo(repo_root)
        gh_repo = get_client().get_repo(f"{owner}/{name}")
        base = base or gh_repo.default_branch
        pr = gh_repo.create_pull(title=title, body=body, head=head_branch, base=base)
    except GitHubError as exc:
        raise ToolError(str(exc)) from exc
    except GithubException as exc:
        message = exc.data.get("message", str(exc)) if isinstance(exc.data, dict) else str(exc)
        raise ToolError(f"GitHub API error: {message}") from exc

    return f"Created PR #{pr.number}: {pr.html_url}"