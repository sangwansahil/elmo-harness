"""Rejection-sampling Fine-Tuning — RL-flavored loop on top of plain SFT.

For each training row:
  1. Sample N completions from the *current* model (the policy).
  2. Score each with the verifier (the reward function).
  3. Keep completions above the acceptance threshold.
  4. Write them to a jsonl and run one SFT epoch on the filtered set.
  5. Repeat for `rft_rounds` rounds.

Conceptually GRPO-with-greedy-policy: no value model, no KL term, no PPO.
Works on Apple Silicon via MLX because everything is forward passes + SFT,
which MLX supports today.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elmo.backends.base import TrainResult
from elmo.backends.mlx import MLXBackend
from elmo.reward import function_call_reward
from elmo.spec import TaskSpec


@dataclass
class RFTConfig:
    rounds: int = 2
    samples_per_prompt: int = 4
    accept_threshold: float = 0.65
    temperature: float = 0.7
    max_train_rows: int = 256


def _load_chat_rows(jsonl: Path) -> list[dict]:
    return [json.loads(ln) for ln in jsonl.read_text().splitlines() if ln.strip()]


def _eval_companion(train_jsonl: Path) -> Path:
    """Look up the foundry_eval.jsonl that sits next to a foundry_train.jsonl
    (or eval.jsonl next to train.raw.jsonl). Returns the path or train_jsonl
    if no companion is found."""
    candidates = [
        train_jsonl.with_name(train_jsonl.stem.replace("train", "eval") + ".jsonl"),
        train_jsonl.parent / "eval.jsonl",
        train_jsonl.parent / "foundry_eval.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return train_jsonl


class RFTBackend:
    """Wraps MLXBackend with a rejection-sampling outer loop."""

    name = "rft"

    def __init__(self, base: MLXBackend | None = None, config: RFTConfig | None = None):
        self.base = base or MLXBackend()
        self.config = config or RFTConfig()

    def _score_completions(self, completions: list[str], expected: list[dict]) -> list[float]:
        return [function_call_reward(c, expected) for c in completions]

    def train(
        self,
        spec: TaskSpec,
        train_jsonl: Path,
        val_jsonl: Path | None,
        out_dir: Path,
    ) -> TrainResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = RFTConfig(
            rounds=spec.train.rft_rounds,
            samples_per_prompt=spec.train.rft_samples_per_prompt,
            accept_threshold=spec.train.rft_accept_threshold,
            temperature=self.config.temperature,
            max_train_rows=self.config.max_train_rows,
        )

        eval_jsonl = _eval_companion(train_jsonl)
        eval_rows = _load_chat_rows(eval_jsonl)[: cfg.max_train_rows]
        if not eval_rows:
            # Nothing to sample against — fall back to plain SFT.
            return self.base.train(spec, train_jsonl, val_jsonl, out_dir)

        current_adapter: Path | None = None
        last_result: TrainResult | None = None
        for round_n in range(1, cfg.rounds + 1):
            round_dir = out_dir / f"rft_round_{round_n:02d}"
            round_dir.mkdir(exist_ok=True)

            # Build the prompts from the eval rows' system + user message.
            prompts = [self._build_prompt(r) for r in eval_rows]

            # Generate N samples per prompt by repeating the prompt list N times.
            replicated = [p for p in prompts for _ in range(cfg.samples_per_prompt)]
            sampled = self.base.generate(
                spec.base_model, current_adapter, replicated,
                max_tokens=512, temperature=cfg.temperature,
            )

            # Re-shape into (n_prompts, samples_per_prompt) and score per row.
            spp = cfg.samples_per_prompt
            accepted: list[dict] = []
            for i, row in enumerate(eval_rows):
                group = sampled[i * spp : (i + 1) * spp]
                rewards = self._score_completions(group, row["expected_calls"])
                for completion, r in zip(group, rewards):
                    if r >= cfg.accept_threshold:
                        accepted.append(self._row_for_sft(row, completion))

            if not accepted:
                # Couldn't surface any good samples — give up rather than train on noise.
                break

            filtered_jsonl = round_dir / "filtered.jsonl"
            with filtered_jsonl.open("w") as f:
                for r in accepted:
                    f.write(json.dumps(r) + "\n")

            last_result = self.base.train(spec, filtered_jsonl, val_jsonl, round_dir)
            current_adapter = last_result.adapter_path

        if last_result is None:
            # No round accepted anything — final fallback to plain SFT on the raw set.
            return self.base.train(spec, train_jsonl, val_jsonl, out_dir)
        return last_result

    def generate(
        self,
        model_id: str,
        adapter_path: Path | None,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]:
        return self.base.generate(model_id, adapter_path, prompts, max_tokens, temperature)

    def _build_prompt(self, eval_row: dict) -> str:
        return json.dumps({"messages": [
            {"role": "system", "content": eval_row["system"]},
            {"role": "user", "content": eval_row["query"]},
        ]})

    def _row_for_sft(self, eval_row: dict, completion: str) -> dict:
        return {"messages": [
            {"role": "system", "content": eval_row["system"]},
            {"role": "user", "content": eval_row["query"]},
            {"role": "assistant", "content": completion},
        ]}
