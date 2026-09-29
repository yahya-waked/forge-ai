"""
GitHub client — Phase 8 of the project spec.

Thin wrapper around PyGithub: authentication, and resolving which
owner/repo a local checkout's 'origin' remote points to, so tools never
need "owner/repo" typed in manually — they derive it from the same repo
git_status/git_diff already operate on.
"""

from __future__ import annotations

import os
import re

from git import Repo
from github import Github

_REMOTE_URL_RE = re.compile(
    r"(?:https://github\.com/|git@github\.com:)(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$"
)


class GitHubError(Exception):
    """No GitHub remote, no token, or the GitHub API itself rejected the request."""


def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise GitHubError("GITHUB_TOKEN is not set (copy .env.example -> .env and fill it in)")
    return token


def resolve_owner_repo(repo_root: str, remote_name: str = "origin") -> tuple[str, str]:
    repo = Repo(repo_root)
    try:
        url = repo.remotes[remote_name].url
    except (IndexError, AttributeError) as exc:
        raise GitHubError(f"No git remote named '{remote_name}' in this repository") from exc

    match = _REMOTE_URL_RE.search(url)
    if not match:
        raise GitHubError(f"Remote '{remote_name}' ({url}) doesn't look like a GitHub URL")
    return match.group("owner"), match.group("repo")


def get_client() -> Github:
    return Github(get_token())