"""Anthropic Messages API provider — separate from OpenAI shape."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .base import CompletionRequest, CompletionResponse
from .openai_compat import ProviderError


ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider:
    name: str

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _build_body(self, req: CompletionRequest) -> dict[str, Any]:
        # Anthropic separates 'system' from 'messages'.
        system_parts = [m["content"] for m in req.messages if m["role"] == "system"]
        msgs = [m for m in req.messages if m["role"] != "system"]
        body: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": msgs,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if req.stop:
            body["stop_sequences"] = req.stop
        body.update(req.extra)
        return body

    def complete(self, req: CompletionRequest) -> CompletionResponse:
        url = f"{self.base_url}/messages"
        body = json.dumps(self._build_body(req)).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": self.api_key,
            "User-Agent": "elmo-harness/0.1",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                return _parse_anthropic_response(raw)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
                last_err = ProviderError(f"{self.name} {e.code}: {err_body}")
                if e.code != 429 and not (500 <= e.code < 600):
                    raise last_err from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = ProviderError(f"{self.name} transport: {e!r}")
            time.sleep(min(2**attempt, 8))
        assert last_err is not None
        raise last_err


def _parse_anthropic_response(raw: dict) -> CompletionResponse:
    try:
        blocks = raw.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        stop = raw.get("stop_reason", "end_turn")
    except (KeyError, TypeError) as e:
        raise ProviderError(f"malformed anthropic response: {raw}") from e
    usage = raw.get("usage") or {}
    return CompletionResponse(
        text=text,
        model=raw.get("model", "?"),
        prompt_tokens=int(usage.get("input_tokens", 0)),
        completion_tokens=int(usage.get("output_tokens", 0)),
        finish_reason=stop,
        raw=raw,
    )
