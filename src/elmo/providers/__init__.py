"""Provider registry — resolves a (provider, model) into a callable client.

Three tiers, opencode-style:
  local     — LM Studio / Ollama / llama.cpp at http://localhost:1234/v1
  free      — OpenRouter :free models, Groq free tier, Cerebras
  byok      — OpenAI, Anthropic, DeepSeek, Together, Mistral, etc.

Provider keys are read from env vars. See `KNOWN_PROVIDERS` for the registry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .anthropic import AnthropicProvider
from .base import ChatMessage, CompletionRequest, CompletionResponse, Provider
from .cached import CachedProvider
from .openai_compat import OpenAICompatibleProvider


__all__ = [
    "ChatMessage",
    "CompletionRequest",
    "CompletionResponse",
    "Provider",
    "CachedProvider",
    "get_provider",
    "list_available",
    "KNOWN_PROVIDERS",
]


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    env_key: str | None
    flavor: str = "openai"  # "openai" or "anthropic"


KNOWN_PROVIDERS: dict[str, ProviderConfig] = {
    "openai": ProviderConfig("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ProviderConfig(
        "anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY", flavor="anthropic"
    ),
    "openrouter": ProviderConfig(
        "openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"
    ),
    "groq": ProviderConfig("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "deepseek": ProviderConfig("deepseek", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "together": ProviderConfig("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "cerebras": ProviderConfig(
        "cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"
    ),
    "lmstudio": ProviderConfig("lmstudio", "http://localhost:1234/v1", None),
    "ollama": ProviderConfig("ollama", "http://localhost:11434/v1", None),
}


def get_provider(name: str, *, cache: bool = True) -> Provider:
    """Return a Provider instance for the given name.

    Reads the API key from env. If `cache` is True and the env var
    ELMO_NO_CACHE is unset, the provider is wrapped in a CachedProvider that
    deduplicates identical requests against `.elmo/cache/completions.db`.
    """
    if name not in KNOWN_PROVIDERS:
        raise ValueError(
            f"unknown provider '{name}'. known: {sorted(KNOWN_PROVIDERS)}. "
            "add a custom one by instantiating OpenAICompatibleProvider directly."
        )
    cfg = KNOWN_PROVIDERS[name]
    api_key = os.environ.get(cfg.env_key) if cfg.env_key else None
    if cfg.env_key and not api_key:
        raise RuntimeError(
            f"provider '{name}' needs env var {cfg.env_key}. "
            f"local providers (lmstudio, ollama) don't need a key."
        )
    if cfg.flavor == "anthropic":
        inner: Provider = AnthropicProvider(name=name, base_url=cfg.base_url, api_key=api_key or "")
    else:
        inner = OpenAICompatibleProvider(name=name, base_url=cfg.base_url, api_key=api_key)

    if cache:
        from elmo.cache import CompletionCache, cache_disabled
        if not cache_disabled():
            cache_path = Path.cwd() / ".elmo" / "cache" / "completions.db"
            return CachedProvider(inner, CompletionCache(cache_path))
    return inner


def list_available() -> list[str]:
    """Return the providers for which the env key is present (or which need no key)."""
    out: list[str] = []
    for name, cfg in KNOWN_PROVIDERS.items():
        if cfg.env_key is None or os.environ.get(cfg.env_key):
            out.append(name)
    return out
