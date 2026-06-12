"""Per-row provenance log. One jsonl line per accepted training row."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class ProvenanceLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate if exists (one log per run)
        path.write_text("")
        self._n = 0

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    def write(
        self,
        *,
        row_id: str,
        scenario_id: str,
        brief_id: str,
        planner_model: str,
        generator_model: str,
        generator_tokens: tuple[int, int],
        verifier_passed: bool,
        verifier_reasons: list[str],
        judge_score: float | None = None,
        seed_prompt_hash: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "row_id": row_id,
            "scenario_id": scenario_id,
            "brief_id": brief_id,
            "planner_model": planner_model,
            "generator_model": generator_model,
            "generator_prompt_tokens": generator_tokens[0],
            "generator_completion_tokens": generator_tokens[1],
            "verifier_passed": verifier_passed,
            "verifier_reasons": verifier_reasons,
            "judge_score": judge_score,
            "seed_prompt_hash": seed_prompt_hash,
            "timestamp": time.time(),
        }
        if extra:
            entry["extra"] = extra
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self._n += 1

    @property
    def n(self) -> int:
        return self._n
