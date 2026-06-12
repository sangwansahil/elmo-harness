"""Tests for the provider HTTP layer — body shaping, response parsing.

We don't make real network calls. We patch urllib.request.urlopen to return
canned bytes and assert the request body / parsing behavior.
"""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from elmo.providers.anthropic import AnthropicProvider
from elmo.providers.base import CompletionRequest
from elmo.providers.openai_compat import OpenAICompatibleProvider


@contextmanager
def _mock_urlopen(payload: dict, recorder: dict[str, Any] | None = None):
    body = json.dumps(payload).encode("utf-8")

    class _Resp:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    def _fake_urlopen(request, timeout=None):  # noqa: ANN001
        if recorder is not None:
            recorder["url"] = request.full_url
            recorder["headers"] = dict(request.headers)
            recorder["body"] = json.loads(request.data.decode("utf-8"))
        return _Resp(body)

    with patch("urllib.request.urlopen", _fake_urlopen):
        yield


def test_openai_compat_body_and_parse() -> None:
    rec: dict[str, Any] = {}
    payload = {
        "model": "gpt-x",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    p = OpenAICompatibleProvider("openai", "https://api.openai.com/v1", "sk-test")
    with _mock_urlopen(payload, rec):
        resp = p.complete(CompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-x",
            temperature=0.3,
            max_tokens=100,
        ))
    assert rec["url"].endswith("/chat/completions")
    assert rec["headers"]["Authorization"] == "Bearer sk-test"
    assert rec["body"]["model"] == "gpt-x"
    assert rec["body"]["temperature"] == 0.3
    assert rec["body"]["messages"][0]["content"] == "hi"
    assert resp.text == "hello"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 5
    assert resp.finish_reason == "stop"


def test_openai_compat_no_key_omits_auth() -> None:
    rec: dict[str, Any] = {}
    payload = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }
    p = OpenAICompatibleProvider("lmstudio", "http://localhost:1234/v1", None)
    with _mock_urlopen(payload, rec):
        p.complete(CompletionRequest(messages=[{"role": "user", "content": "hi"}], model="x"))
    # urllib normalizes headers; check both casings just in case.
    headers = rec["headers"]
    assert "Authorization" not in headers and "authorization" not in headers


def test_openrouter_attribution_headers() -> None:
    rec: dict[str, Any] = {}
    payload = {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {},
    }
    p = OpenAICompatibleProvider(
        "openrouter", "https://openrouter.ai/api/v1", "or-test"
    )
    with _mock_urlopen(payload, rec):
        p.complete(CompletionRequest(messages=[{"role": "user", "content": "hi"}], model="x"))
    headers = rec["headers"]
    # urllib normalizes header keys to title-case
    has_referer = "Http-referer" in headers or "HTTP-Referer" in headers
    has_title = "X-title" in headers or "X-Title" in headers
    assert has_referer and has_title


def test_anthropic_body_splits_system_and_messages() -> None:
    rec: dict[str, Any] = {}
    payload = {
        "model": "claude-opus",
        "content": [{"type": "text", "text": "hi back"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    p = AnthropicProvider("anthropic", "https://api.anthropic.com/v1", "ant-test")
    with _mock_urlopen(payload, rec):
        resp = p.complete(CompletionRequest(
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            model="claude-opus",
            max_tokens=20,
        ))
    assert rec["url"].endswith("/messages")
    assert rec["body"]["system"] == "be brief"
    assert rec["body"]["messages"] == [{"role": "user", "content": "hi"}]
    headers = rec["headers"]
    # urllib title-cases header keys
    assert headers.get("X-api-key") == "ant-test" or headers.get("X-Api-Key") == "ant-test"
    assert resp.text == "hi back"
    assert resp.prompt_tokens == 7 and resp.completion_tokens == 3
