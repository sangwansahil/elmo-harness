"""Static price table — $ per 1k input / output tokens, mid-2026 figures.

Estimates only. Producers update this without notice; check provider docs
before relying on the number for billing decisions. Zero for any unknown
model means cost is reported as $0.00 instead of crashing.
"""

from __future__ import annotations


# (input $/1k, output $/1k)
PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "openai/gpt-4o": (0.0025, 0.010),
    "openai/gpt-4o-mini": (0.00015, 0.0006),
    "openai/o1-mini": (0.003, 0.012),
    # Anthropic
    "anthropic/claude-opus-4-7": (0.015, 0.075),
    "anthropic/claude-sonnet-4-6": (0.003, 0.015),
    "anthropic/claude-haiku-4-5-20251001": (0.0008, 0.004),
    # DeepSeek
    "deepseek/deepseek-chat": (0.00027, 0.0011),
    "deepseek/deepseek-reasoner": (0.00055, 0.0022),
    # OpenRouter — most ":free" tags are zero-priced for trials
    "openrouter/deepseek/deepseek-chat:free": (0.0, 0.0),
    "openrouter/deepseek/deepseek-r1:free": (0.0, 0.0),
    "openrouter/meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "openrouter/anthropic/claude-3.5-sonnet": (0.003, 0.015),
    # Groq (free tier rate-limited)
    "groq/llama-3.3-70b-versatile": (0.0, 0.0),
    "groq/mixtral-8x7b-32768": (0.0, 0.0),
    # Cerebras free
    "cerebras/llama3.3-70b": (0.0, 0.0),
    # Together
    "together/deepseek-ai/DeepSeek-V3": (0.00125, 0.00125),
    # Local
    "lmstudio/*": (0.0, 0.0),
    "ollama/*": (0.0, 0.0),
}


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    key = f"{provider}/{model}"
    if key not in PRICES:
        key = f"{provider}/*"
    inp, out = PRICES.get(key, (0.0, 0.0))
    return (prompt_tokens / 1000.0) * inp + (completion_tokens / 1000.0) * out
