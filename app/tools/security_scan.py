"""
Security scan tool — backs the Security Agent (sections 22-23 of the spec).

A deterministic scan of everything that's about to be committed (added
lines only): hardcoded secrets and code that can execute arbitrary
commands. An LLM reviewer can miss or talk itself out of a finding; a
regex can't. The Orchestrator treats any [SECRET] finding as a hard
block regardless of what the LLM concludes.
"""

from __future__ import annotations

import re

from app.safety.content_scan import scan_for_danger
from app.tools.git_tools import _get_repo, collect_working_diff
from app.tools.registry import RiskLevel, registry

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key ID"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "Google API key"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), "private key block"),
    (
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
        ),
        "hardcoded credential assignment",
    ),
]


@registry.register(
    name="scan_diff_security",
    description=(
        "Deterministically scan every uncommitted change (including new files) for "
        "hardcoded secrets ([SECRET]) and code that can execute arbitrary commands "
        "([DANGER]). Returns one finding per line, or 'No findings.'"
    ),
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
def scan_diff_security(repo_root: str) -> str:
    diff = collect_working_diff(_get_repo(repo_root))

    findings: list[str] = []
    current_file = "?"
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current_file = line[4:].removeprefix("b/")
            continue
        if not line.startswith("+"):
            continue

        added = line[1:]
        for pattern, label in _SECRET_PATTERNS:
            if pattern.search(added):
                findings.append(f"[SECRET] {current_file}: {label}")
        for reason in scan_for_danger(added):
            findings.append(f"[DANGER] {current_file}: {reason}: {added.strip()[:100]}")

    unique = list(dict.fromkeys(findings))
    return "\n".join(unique) if unique else "No findings."