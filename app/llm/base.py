"""
Abstract LLM provider interface for ForgeAI.

Every model backend (Gemini, OpenAI, a local model, ...) implements this
interface. The Agent Runtime only ever talks to `LLMProvider` — never to a
specific SDK — so swapping models never touches agent/tool code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One turn in the conversation sent to / returned from the LLM."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set when role == Role.TOOL
    name: str | None = None  # tool name, set when role == Role.TOOL
    raw: Any = None  # opaque provider-specific payload for this turn (e.g. Gemini's
    # Content object, carrying thought_signature). Generic code never reads this —
    # only the provider that produced it knows how to replay it on the next call.


@dataclass
class ToolSchema:
    """JSON-Schema description of a tool the model is allowed to call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object (type: object, properties, ...)


@dataclass
class LLMResponse:
    """Normalized response returned by any provider implementation."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    provider_message: Any = None  # opaque payload to carry into the next Message.raw,
    # so a provider that needs to replay this exact turn (e.g. Gemini's thought_signature) can.
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def wants_tool_call(self) -> bool:
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Base class every model backend must implement."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a conversation (+ optional tool schemas) and get one response back."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, e.g. 'gemini:gemini-1.5-pro'."""
        raise NotImplementedError