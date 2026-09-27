"""
Content danger scan — a stopgap safety net until the real Docker sandbox
(Phase 7) isolates test/command execution entirely.

This does not block anything by itself; it only makes sure that when a
write contains something that can execute arbitrary commands (subprocess,
shell=True, eval/exec...), the human reviewer sees exactly why it's
dangerous instead of just a raw arguments dict they might skim past.
"""

from __future__ import annotations

import re

_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsubprocess\b"), "spawns a subprocess"),
    (re.compile(r"\bos\.system\("), "calls os.system()"),
    (re.compile(r"\bos\.popen\("), "calls os.popen()"),
    (re.compile(r"shell\s*=\s*True"), "uses shell=True"),
    (re.compile(r"\beval\("), "calls eval()"),
    (re.compile(r"\bexec\("), "calls exec()"),
]


def scan_for_danger(text: str) -> list[str]:
    """Returns human-readable reasons this text looks dangerous; empty if none matched."""
    return [reason for pattern, reason in _DANGEROUS_PATTERNS if pattern.search(text)]