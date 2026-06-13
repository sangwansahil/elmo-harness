"""Named role-config presets. `elmo preset list` / `elmo preset apply <name>`."""

from __future__ import annotations

from elmo.roles import RoleConfig


PRESETS: dict[str, dict[str, RoleConfig]] = {
    "free-openrouter": {
        "planner": RoleConfig(
            provider="openrouter",
            model="deepseek/deepseek-r1:free",
        ),
        "generator": RoleConfig(
            provider="openrouter",
            model="deepseek/deepseek-chat:free",
        ),
        "judge": RoleConfig(
            provider="openrouter",
            model="meta-llama/llama-3.3-70b-instruct:free",
        ),
    },
    "groq-fast": {
        "planner": RoleConfig(provider="groq", model="llama-3.3-70b-versatile"),
        "generator": RoleConfig(provider="groq", model="llama-3.3-70b-versatile"),
        "judge": RoleConfig(provider="groq", model="llama-3.3-70b-versatile"),
    },
    "anthropic-byok": {
        "planner": RoleConfig(provider="anthropic", model="claude-opus-4-7"),
        "generator": RoleConfig(provider="anthropic", model="claude-haiku-4-5-20251001"),
        "judge": RoleConfig(provider="anthropic", model="claude-haiku-4-5-20251001"),
    },
    "local-only": {
        "planner": RoleConfig(provider="lmstudio", model="qwen3-4b-thinking"),
        "generator": RoleConfig(provider="lmstudio", model="qwen2.5-coder-7b-instruct"),
        "judge": RoleConfig(provider="lmstudio", model="qwen3-4b"),
    },
}


def list_presets() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str) -> dict[str, RoleConfig]:
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'. choose from: {list_presets()}")
    return PRESETS[name]
