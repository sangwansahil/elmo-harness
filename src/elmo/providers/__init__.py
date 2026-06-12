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

from .anthropic import AnthropicProvider
from .base import ChatMessage, CompletionRequest, CompletionResponse, Provider
from .openai_compat import OpenAICompatibleProvider


__all__ = [
    "ChatMessage",
    "CompletionRequest",
    "CompletionResponse",
    "Provider",
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


def get_provider(name: str) -> Provider:
    """Return a Provider instance for the given name, reading the API key from env."""
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
        return AnthropicProvider(name=name, base_url=cfg.base_url, api_key=api_key or "")
    return OpenAICompatibleProvider(name=name, base_url=cfg.base_url, api_key=api_key)


def list_available() -> list[str]:
    """Return the providers for which the env key is present (or which need no key)."""
    out: list[str] = []
    for name, cfg in KNOWN_PROVIDERS.items():
        if cfg.env_key is None or os.environ.get(cfg.env_key):
            out.append(name)
    return out
