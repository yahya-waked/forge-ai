"""
Shared retry policy for Gemini API calls (chat + embeddings).

Retries a few times on transient 503 overload or 429 rate-limit
responses. When Gemini's own error message tells us exactly how long to
wait ("Please retry in 52.6s"), we trust that number instead of guessing
with a fixed exponential backoff — it directly reflects the real quota
window, which a guessed schedule can easily undershoot.
"""

from __future__ import annotations

import re

from google.genai import errors as genai_errors

_RETRY_DELAY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)
_FALLBACK_MAX_WAIT = 60.0


def is_retryable_genai_error(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)
    return False


def wait_for_genai_retry(retry_state) -> float:
    """Tenacity wait callable: prefer the exact delay Gemini's error message
    suggests over a guessed exponential backoff."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None:
        match = _RETRY_DELAY_RE.search(str(exc))
        if match:
            return float(match.group(1)) + 1.0  # small buffer past the server's estimate

    # Fallback for errors with no explicit delay (e.g. a bare 503 with no
    # RetryInfo): standard exponential backoff capped at _FALLBACK_MAX_WAIT.
    attempt = retry_state.attempt_number
    return min(_FALLBACK_MAX_WAIT, 3.0 * (2 ** (attempt - 1)))