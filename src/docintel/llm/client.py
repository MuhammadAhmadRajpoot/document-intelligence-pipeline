"""A minimal JSON-returning chat client over any OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Literal, Protocol, TypedDict

from openai import APIError, OpenAI

logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMError(RuntimeError):
    """The LLM call failed or returned something that is not a JSON object."""


class JSONLLMClient(Protocol):
    """Anything that can turn chat messages into a parsed JSON object."""

    model: str

    def complete_json(
        self,
        messages: list[ChatMessage],
        *,
        schema_name: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating markdown fences or prose around it."""
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("response contains no JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON from model: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("model returned JSON that is not an object")
    return value


class OpenAICompatibleClient:
    """Groq (or any OpenAI-compatible server) client using JSON mode or JSON-schema output."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = "https://api.groq.com/openai/v1",
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_retries: int = 4,
        response_format: Literal["json_object", "json_schema"] = "json_object",
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._response_format = response_format
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=max_retries
        )
        self._usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _format(self, schema_name: str, json_schema: dict[str, Any] | None) -> dict[str, Any]:
        if self._response_format == "json_schema" and json_schema is not None:
            return {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": False},
            }
        return {"type": "json_object"}

    def complete_json(
        self,
        messages: list[ChatMessage],
        *,
        schema_name: str,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self._temperature,
                response_format=self._format(schema_name, json_schema),  # type: ignore[arg-type]
            )
        except APIError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        usage = response.usage
        if usage is not None:
            with self._usage_lock:
                self.prompt_tokens += usage.prompt_tokens
                self.completion_tokens += usage.completion_tokens
        logger.debug(
            "llm call",
            extra={
                "schema": schema_name,
                "model": self.model,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
            },
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMError("empty response from model")
        return parse_json_object(content)
