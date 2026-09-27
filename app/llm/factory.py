"""
LLMProvider factory.

The rest of ForgeAI never imports GeminiProvider (or any other backend)
directly — it asks this factory for "the configured provider". This is
what keeps the Agent Runtime model-agnostic and lets us add OpenAI /
local-model support later without touching agent or tool code.
"""

from __future__ import annotations

import os

from app.llm.base import LLMProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {}


def register_provider(key: str):
    """Decorator to register a provider class under a short key ('gemini', 'openai', ...)."""

    def _wrap(cls: type[LLMProvider]) -> type[LLMProvider]:
        _PROVIDERS[key] = cls
        return cls

    return _wrap


def _load_builtin_providers() -> None:
    # Imported lazily so an unused backend's SDK doesn't have to be installed.
    if "gemini" not in _PROVIDERS:
        from app.llm.gemini import GeminiProvider

        _PROVIDERS["gemini"] = GeminiProvider


def get_llm_provider(name: str | None = None, **kwargs) -> LLMProvider:
    """
    Build the configured LLMProvider.

    name:   'gemini', 'openai', ... Defaults to env var LLM_PROVIDER, then 'gemini'.
    kwargs: forwarded to the provider's constructor (e.g. model="gemini-1.5-flash").
    """
    _load_builtin_providers()

    name = name or os.environ.get("LLM_PROVIDER", "gemini")
    try:
        provider_cls = _PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown LLM provider '{name}'. Available: {list(_PROVIDERS)}"
        ) from exc

    return provider_cls(**kwargs)