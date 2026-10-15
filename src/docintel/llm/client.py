"""A minimal JSON-returning chat client over any OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Literal, Protocol, TypedDict

from openai import APIError, OpenAI, RateLimitError

logger = logging.getLogger(__name__)

_RETRY_IN_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)")
MAX_BACKOFF_S = 60.0


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


def rate_limit_wait_s(exc: RateLimitError, attempt: int) -> float:
    """Seconds to wait after a 429: the server's hint if present, else exponential backoff.

    The hint is returned uncapped so callers can tell a per-minute limit (seconds) from
    an exhausted daily quota (minutes), which is not worth waiting for.
    """
    header = exc.response.headers.get("retry-after") if exc.response is not None else None
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass
    match = _RETRY_IN_RE.search(str(exc))
    if match:
        minutes, value, unit = match.groups()
        seconds = float(value) / (1000 if unit == "ms" else 1) + 60 * int(minutes or 0)
        return seconds + 0.5
    return min(2.0**attempt, MAX_BACKOFF_S)


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
        max_retries: int = 2,
        rate_limit_retries: int = 8,
        response_format: Literal["json_object", "json_schema"] = "json_object",
    ) -> None:
        self.model = model
        self._temperature = temperature
        self._response_format = response_format
        self._rate_limit_retries = rate_limit_retries
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
        for attempt in range(self._rate_limit_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self._temperature,
                    response_format=self._format(schema_name, json_schema),  # type: ignore[arg-type]
                )
                break
            except RateLimitError as exc:
                if attempt == self._rate_limit_retries:
                    raise LLMError(f"rate limited after {attempt + 1} tries: {exc}") from exc
                wait = rate_limit_wait_s(exc, attempt)
                if wait > MAX_BACKOFF_S:
                    raise LLMError(f"LLM quota exhausted (retry in {wait:.0f}s): {exc}") from exc
                logger.info("rate limited, backing off", extra={"wait_s": round(wait, 2)})
                time.sleep(wait)
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
