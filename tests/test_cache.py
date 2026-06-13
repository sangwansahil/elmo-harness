"""Tests for the completion cache and CachedProvider wrapper."""

from __future__ import annotations

from pathlib import Path

from elmo.cache import CompletionCache
from elmo.pricing import estimate_cost
from elmo.providers.base import CompletionRequest, CompletionResponse
from elmo.providers.cached import CachedProvider


class _StubProvider:
    name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, req: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        return CompletionResponse(
            text=f"answer for {req.messages[-1]['content']}",
            model=req.model,
            prompt_tokens=12, completion_tokens=8, finish_reason="stop",
        )


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path / "c.db")
    req = CompletionRequest(messages=[{"role": "user", "content": "hi"}], model="m1")
    key = cache.make_key("stub", req)
    assert cache.get(key) is None
    cache.put(key, "stub", text="hello", model="m1",
              prompt_tokens=10, completion_tokens=5, finish_reason="stop")
    hit = cache.get(key)
    assert hit is not None and hit["text"] == "hello"


def test_cache_hit_increments_counter(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path / "c.db")
    req = CompletionRequest(messages=[{"role": "user", "content": "x"}], model="m1")
    key = cache.make_key("stub", req)
    cache.put(key, "stub", text="a", model="m1",
              prompt_tokens=1, completion_tokens=1, finish_reason="stop")
    cache.get(key)
    cache.get(key)
    cache.get(key)
    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["total_hits"] == 3


def test_cache_key_is_deterministic_across_message_order(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path / "c.db")
    req_a = CompletionRequest(messages=[
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ], model="m1", temperature=0.0)
    req_b = CompletionRequest(messages=[
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ], model="m1", temperature=0.0)
    assert cache.make_key("p", req_a) == cache.make_key("p", req_b)


def test_cache_key_differs_for_temperature(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path / "c.db")
    req_a = CompletionRequest(messages=[{"role": "user", "content": "hi"}], model="m1", temperature=0.0)
    req_b = CompletionRequest(messages=[{"role": "user", "content": "hi"}], model="m1", temperature=0.5)
    assert cache.make_key("p", req_a) != cache.make_key("p", req_b)


def test_cached_provider_dedups_calls(tmp_path: Path) -> None:
    inner = _StubProvider()
    cached = CachedProvider(inner, CompletionCache(tmp_path / "c.db"))
    req = CompletionRequest(messages=[{"role": "user", "content": "ping"}], model="m1")
    r1 = cached.complete(req)
    r2 = cached.complete(req)
    assert r1.text == r2.text == "answer for ping"
    assert inner.calls == 1  # second call served from cache


def test_cache_clear(tmp_path: Path) -> None:
    cache = CompletionCache(tmp_path / "c.db")
    for i in range(3):
        req = CompletionRequest(messages=[{"role": "user", "content": f"q{i}"}], model="m1")
        cache.put(cache.make_key("p", req), "p", text=f"a{i}", model="m1",
                  prompt_tokens=1, completion_tokens=1, finish_reason="stop")
    n = cache.clear()
    assert n == 3
    assert cache.stats()["entries"] == 0


def test_pricing_known_model() -> None:
    cost = estimate_cost("openai", "gpt-4o", 1000, 2000)
    # 1000 input * 0.0025/1k + 2000 output * 0.010/1k = 0.0025 + 0.020 = 0.0225
    assert abs(cost - 0.0225) < 1e-9


def test_pricing_free_model_is_zero() -> None:
    cost = estimate_cost("openrouter", "deepseek/deepseek-chat:free", 100, 100)
    assert cost == 0.0


def test_pricing_unknown_falls_back_to_zero() -> None:
    cost = estimate_cost("imaginary", "no-such-model", 1000, 1000)
    assert cost == 0.0


def test_pricing_local_wildcard() -> None:
    cost = estimate_cost("lmstudio", "any-model-you-like", 5000, 5000)
    assert cost == 0.0
