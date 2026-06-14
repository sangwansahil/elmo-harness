"""Wizard-side task discovery — turn a natural-language prompt into a typed
TaskSpec the run loop can execute.

For the UI-first flow we don't want a non-technical user to look at YAML or
pick a dataset. This module routes the prompt to one of four known task
shapes via keyword heuristics, then fills in sensible defaults for the
picked base model.

The routing is intentionally simple and offline. When a planner role is
configured it can refine the result later — the wizard surfaces the
inferred spec so a user can confirm before the run starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from elmo.spec import (
    Budget,
    Capability,
    DatasetRef,
    EvalConfig,
    FoundryConfig,
    RolesSpec,
    TaskSpec,
    TrainConfig,
)


TaskKind = Literal["function-calling", "math", "structured", "general"]


@dataclass
class TaskGuess:
    kind: TaskKind
    confidence: float
    matched_keywords: list[str]


_KEYWORDS: dict[TaskKind, tuple[str, ...]] = {
    "function-calling": ("function", "tool", "api", "call", "endpoint", "rest", "invoke"),
    "math": ("math", "solve", "calculate", "equation", "arithmetic", "word problem", "gsm8k", "compute"),
    "structured": ("json", "extract", "structured", "schema", "parse", "form", "field", "key"),
}


def classify_prompt(prompt: str) -> TaskGuess:
    """Score the prompt against known task kinds, return the winner."""
    p = prompt.lower()
    best: tuple[TaskKind, int, list[str]] = ("general", 0, [])
    for kind, keywords in _KEYWORDS.items():
        hits = [k for k in keywords if k in p]
        if len(hits) > best[1]:
            best = (kind, len(hits), hits)
    confidence = min(1.0, best[1] / 3.0) if best[1] else 0.0
    return TaskGuess(kind=best[0], confidence=confidence, matched_keywords=best[2])


def _slug(prompt: str) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9 ]+", "", prompt.lower()).strip()
    s = re.sub(r"\s+", "-", s)[:48]
    return s or "wizard-task"


def discover_task(prompt: str, base_model_hf_id: str) -> TaskSpec:
    """Build a complete TaskSpec from a one-sentence prompt and a base model."""
    guess = classify_prompt(prompt)
    name = _slug(prompt)

    if guess.kind == "function-calling":
        return TaskSpec(
            name=name,
            prompt=prompt,
            base_model=base_model_hf_id,
            backend="auto",
            capabilities=[
                Capability(name="tool_selection", description="pick the right tool name", verifier="function_call", weight=1.0),
                Capability(name="arguments", description="extract correct arguments", verifier="json_schema", weight=1.5),
                Capability(name="parallel_calls", description="emit multiple calls when warranted", verifier="function_call", weight=1.0),
            ],
            dataset=DatasetRef(source="hf:Salesforce/xlam-function-calling-60k", split="train", max_rows=2000),
            train=TrainConfig(method="lora", max_steps=200, lora_rank=16, learning_rate=2e-4, batch_size=4),
            eval=EvalConfig(benchmark="bfcl-simple", max_examples=100, target_score=0.75),
            budget=Budget(max_iterations=1, max_dollars=0.0, max_wallclock_hours=2.0),
            foundry=FoundryConfig(enabled=False),
            roles=RolesSpec(),
        )

    if guess.kind == "math":
        return TaskSpec(
            name=name,
            prompt=prompt,
            base_model=base_model_hf_id,
            backend="auto",
            capabilities=[
                Capability(name="correctness", description="final answer matches ground truth", verifier="exact_match", weight=1.0),
            ],
            dataset=DatasetRef(source="hf:openai/gsm8k", split="train", max_rows=2000),
            train=TrainConfig(method="lora", max_steps=300, lora_rank=16, learning_rate=2e-4, batch_size=2),
            eval=EvalConfig(benchmark="gsm8k", max_examples=100, target_score=0.65),
            budget=Budget(max_iterations=1, max_wallclock_hours=3.0),
            foundry=FoundryConfig(enabled=False),
            roles=RolesSpec(),
        )

    if guess.kind == "structured":
        return TaskSpec(
            name=name,
            prompt=prompt,
            base_model=base_model_hf_id,
            backend="auto",
            capabilities=[
                Capability(name="parseable", description="output is valid JSON", verifier="json_schema", weight=1.0),
                Capability(name="keys_present", description="all required keys present", verifier="json_schema", weight=1.0),
                Capability(name="types_correct", description="each value matches schema type", verifier="json_schema", weight=1.5),
            ],
            dataset=DatasetRef(source="synthetic:structured", split="train", max_rows=200),
            train=TrainConfig(method="lora", max_steps=150, lora_rank=16, learning_rate=2e-4, batch_size=4),
            eval=EvalConfig(benchmark="json-format", max_examples=60, target_score=0.80),
            budget=Budget(max_iterations=1, max_wallclock_hours=1.0),
            foundry=FoundryConfig(enabled=False),
            roles=RolesSpec(),
        )

    # general — no public dataset matches, so foundry-from-prompt
    return TaskSpec(
        name=name,
        prompt=prompt,
        base_model=base_model_hf_id,
        backend="auto",
        capabilities=[
            Capability(name="correctness", description="task is completed correctly", verifier="judge", weight=1.0),
        ],
        dataset=DatasetRef(source="synthetic:from-prompt", split="train", max_rows=200),
        train=TrainConfig(method="lora", max_steps=200, lora_rank=16, learning_rate=2e-4, batch_size=4),
        eval=EvalConfig(benchmark="bfcl-simple", max_examples=50, target_score=0.70),
        budget=Budget(max_iterations=1, max_wallclock_hours=1.5),
        foundry=FoundryConfig(enabled=True, scenarios_per_brief=80, rows_target=200),
        roles=RolesSpec(),
    )
