"""OpenAI-compatible provider — works for OpenAI, OpenRouter, Groq, DeepSeek,
Together, Cerebras, LM Studio, Ollama. Stdlib HTTP only — no extra deps."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .base import CompletionRequest, CompletionResponse


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    name: str

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "User-Agent": "elmo-harness/0.1"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter wants attribution headers — harmless elsewhere.
        if "openrouter" in self.base_url:
            h["HTTP-Referer"] = "https://github.com/sangwansahil/elmo-harness"
            h["X-Title"] = "elmo-harness"
        return h

    def _build_body(self, req: CompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.response_format is not None:
            body["response_format"] = req.response_format
        if req.stop:
            body["stop"] = req.stop
        body.update(req.extra)
        return body

    def complete(self, req: CompletionRequest) -> CompletionResponse:
        url = f"{self.base_url}/chat/completions"
        body = json.dumps(self._build_body(req)).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(
                    url, data=body, headers=self._headers(), method="POST"
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                return _parse_openai_response(raw)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
                last_err = ProviderError(f"{self.name} {e.code}: {err_body}")
                # 5xx and 429 are retried; 4xx other than 429 are not
                if e.code != 429 and not (500 <= e.code < 600):
                    raise last_err from e
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = ProviderError(f"{self.name} transport: {e!r}")
            time.sleep(min(2**attempt, 8))
        assert last_err is not None
        raise last_err


def _parse_openai_response(raw: dict) -> CompletionResponse:
    try:
        choice = raw["choices"][0]
        text = choice["message"]["content"] or ""
        finish = choice.get("finish_reason", "stop")
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderError(f"malformed response: {raw}") from e
    usage = raw.get("usage") or {}
    return CompletionResponse(
        text=text,
        model=raw.get("model", "?"),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        finish_reason=finish,
        raw=raw,
    )
