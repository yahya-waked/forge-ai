"""
Gemini implementation of LLMProvider, built on the current `google-genai` SDK.

The older `google-generativeai` package is deprecated and its Schema proto
rejects plain JSON Schema keywords like "default" — this version avoids
both problems by using `parameters_json_schema`, which accepts our tool
schemas as-is.

Install:   pip install google-genai
Env var:   GEMINI_API_KEY
"""

from __future__ import annotations

import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt

from app.llm.base import LLMProvider, LLMResponse, Message, Role, ToolCall, ToolSchema
from app.llm.retry_utils import is_retryable_genai_error, wait_for_genai_retry

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # GA, higher free-tier RPM/RPD than 3.1-flash-lite


def _tool_schema_to_function_declaration(tool: ToolSchema) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=tool.parameters,
    )


def _messages_to_contents(messages: list[Message]) -> tuple[str | None, list[types.Content]]:
    """
    Splits our Message list into (system_instruction, contents) the way the
    genai SDK expects: system prompt kept separate, everything else as
    alternating user/model Content turns.
    """
    system_instruction: str | None = None
    contents: list[types.Content] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_instruction = (system_instruction or "") + msg.content + "\n"
            continue

        if msg.role == Role.USER:
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=msg.content)])
            )

        elif msg.role == Role.ASSISTANT:
            if isinstance(msg.raw, types.Content):
                # Replay Gemini's own Content object unchanged — this is what
                # carries thought_signature on the function_call parts, which
                # Gemini 3 requires back on the next turn.
                contents.append(msg.raw)
                continue

            parts: list[types.Part] = []
            if msg.content:
                parts.append(types.Part.from_text(text=msg.content))
            for call in msg.tool_calls:
                parts.append(types.Part.from_function_call(name=call.name, args=call.arguments))
            contents.append(types.Content(role="model", parts=parts))

        elif msg.role == Role.TOOL:
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=msg.name,
                            response={"result": msg.content},
                        )
                    ],
                )
            )

    return system_instruction, contents


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model

    @property
    def name(self) -> str:
        return f"gemini:{self._model_name}"

    @retry(
        retry=retry_if_exception(is_retryable_genai_error),
        wait=wait_for_genai_retry,
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_api(
        self, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        # Gemini's servers occasionally return 503 UNAVAILABLE under load — this
        # is transient and worth a few automatic retries with backoff before we
        # surface it as a real failure to the agent loop.
        return self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        system_instruction, contents = _messages_to_contents(messages)
        if not contents:
            raise ValueError("No messages to send to Gemini")

        gemini_tools = None
        if tools:
            gemini_tools = [
                types.Tool(
                    function_declarations=[
                        _tool_schema_to_function_declaration(t) for t in tools
                    ]
                )
            ]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=gemini_tools,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self._call_api(contents=contents, config=config)

        candidate = response.candidates[0] if response.candidates else None
        candidate_parts = candidate.content.parts if candidate else []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in candidate_parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc:
                tool_calls.append(
                    ToolCall(id=str(uuid.uuid4()), name=fc.name, arguments=dict(fc.args))
                )

        usage = getattr(response, "usage_metadata", None)

        return LLMResponse(
            content="".join(text_parts),
            tool_calls=tool_calls,
            raw=response,
            provider_message=candidate.content if candidate else None,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )