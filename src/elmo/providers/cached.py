"""Caching provider wrapper — transparent passthrough with a cache check."""

from __future__ import annotations

from elmo.cache import CompletionCache
from elmo.providers.base import CompletionRequest, CompletionResponse, Provider


class CachedProvider:
    """Wraps any Provider; identical requests resolve from a SQLite cache."""

    def __init__(self, inner: Provider, cache: CompletionCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name

    def complete(self, req: CompletionRequest) -> CompletionResponse:
        key = CompletionCache.make_key(self.inner.name, req)
        hit = self.cache.get(key)
        if hit is not None:
            return CompletionResponse(
                text=hit["text"],
                model=hit["model"],
                prompt_tokens=hit["prompt_tokens"],
                completion_tokens=hit["completion_tokens"],
                finish_reason=hit["finish_reason"],
            )
        resp = self.inner.complete(req)
        self.cache.put(
            key, self.inner.name,
            text=resp.text, model=resp.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            finish_reason=resp.finish_reason,
        )
        return resp
