"""Backend protocol: every trainer implements the same minimal surface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from elmo.spec import TaskSpec


@dataclass
class TrainResult:
    adapter_path: Path
    train_loss: float
    val_loss: float | None
    steps: int


class TrainBackend(Protocol):
    name: str

    def train(
        self,
        spec: TaskSpec,
        train_jsonl: Path,
        val_jsonl: Path | None,
        out_dir: Path,
    ) -> TrainResult:
        """Run a single fine-tune. Returns paths to artifacts and final losses."""
        ...

    def generate(
        self,
        model_id: str,
        adapter_path: Path | None,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]:
        """Run inference. Used for eval and diagnose."""
        ...
