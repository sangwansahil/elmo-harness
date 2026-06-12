"""Provider protocol — every backend implements .complete()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class CompletionRequest:
    messages: list[ChatMessage]
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048
    response_format: dict | None = None
    stop: list[str] | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class CompletionResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw: dict | None = None


class Provider(Protocol):
    name: str

    def complete(self, req: CompletionRequest) -> CompletionResponse:
        """Make one chat completion call. Synchronous, raises on hard failure."""
        ...
