"""Task specification — the typed contract a run is bound to."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


VerifierKind = Literal["function_call", "json_schema", "exec_python", "exact_match", "judge"]


class Capability(BaseModel):
    name: str
    description: str = ""
    verifier: VerifierKind = "judge"
    weight: float = 1.0


class DatasetRef(BaseModel):
    source: str  # "hf:..." or "local:..."
    split: str = "train"
    max_rows: int | None = None


class TrainConfig(BaseModel):
    method: Literal["lora", "qlora", "full"] = "lora"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_seq_len: int = 2048
    epochs: int = 1
    max_steps: int | None = None
    warmup_steps: int = 50
    grad_accum_steps: int = 1
    seed: int = 42


class EvalConfig(BaseModel):
    benchmark: str  # "bfcl-simple", "humaneval", etc.
    max_examples: int = 100
    target_score: float | None = None


class Budget(BaseModel):
    max_iterations: int = 1
    max_dollars: float = 10.0
    max_wallclock_hours: float = 4.0


class TaskSpec(BaseModel):
    name: str = Field(..., description="Short slug for this task.")
    prompt: str = Field(..., description="The original natural-language ask.")
    base_model: str = Field(..., description="HF model id, e.g. 'mlx-community/Qwen2.5-1.5B-Instruct-4bit'")
    backend: Literal["mlx", "unsloth", "auto"] = "auto"
    capabilities: list[Capability] = Field(default_factory=list)
    dataset: DatasetRef
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig
    budget: Budget = Field(default_factory=Budget)


def load_spec(path: str | Path) -> TaskSpec:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"spec not found: {path}")
    raw = yaml.safe_load(path.read_text())
    try:
        return TaskSpec.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"invalid spec {path}:\n{e}") from e


def dump_spec(spec: TaskSpec, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(spec.model_dump(), sort_keys=False))
