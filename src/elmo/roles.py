"""Three roles: planner (strong), generator (cheap), judge (medium).

A role binds a logical name to a (provider, model). Resolution order:
  1. explicit spec.roles.<role> block
  2. env var ELMO_<ROLE>_PROVIDER + ELMO_<ROLE>_MODEL
  3. a sensible default that uses whatever provider keys are present
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel

from elmo.providers import Provider, get_provider, list_available


RoleName = Literal["planner", "generator", "judge"]


class RoleConfig(BaseModel):
    provider: str
    model: str


def _default_for(role: RoleName) -> RoleConfig | None:
    """Pick a role default using whatever providers are configured."""
    available = set(list_available())
    if role == "planner":
        # strong first
        for p, m in [
            ("anthropic", "claude-opus-4-7"),
            ("openai", "gpt-4o"),
            ("openrouter", "anthropic/claude-3.5-sonnet"),
            ("groq", "llama-3.3-70b-versatile"),
            ("lmstudio", "qwen3-4b-thinking"),
        ]:
            if p in available:
                return RoleConfig(provider=p, model=m)
    if role == "generator":
        for p, m in [
            ("openrouter", "deepseek/deepseek-chat:free"),
            ("deepseek", "deepseek-chat"),
            ("groq", "llama-3.3-70b-versatile"),
            ("together", "deepseek-ai/DeepSeek-V3"),
            ("lmstudio", "qwen2.5-coder-7b-instruct"),
        ]:
            if p in available:
                return RoleConfig(provider=p, model=m)
    if role == "judge":
        for p, m in [
            ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
            ("groq", "llama-3.3-70b-versatile"),
            ("anthropic", "claude-haiku-4-5-20251001"),
            ("lmstudio", "qwen3-4b"),
        ]:
            if p in available:
                return RoleConfig(provider=p, model=m)
    return None


def resolve_role(role: RoleName, override: RoleConfig | None = None) -> RoleConfig | None:
    if override is not None:
        return override
    env_p = os.environ.get(f"ELMO_{role.upper()}_PROVIDER")
    env_m = os.environ.get(f"ELMO_{role.upper()}_MODEL")
    if env_p and env_m:
        return RoleConfig(provider=env_p, model=env_m)
    return _default_for(role)


def get_client(cfg: RoleConfig) -> Provider:
    return get_provider(cfg.provider)


class Roles(BaseModel):
    planner: RoleConfig | None = None
    generator: RoleConfig | None = None
    judge: RoleConfig | None = None
