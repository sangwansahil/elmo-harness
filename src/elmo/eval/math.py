"""Math evaluator — verifiable rewards via final-answer equivalence.

Parses the model's "#### N" or "boxed{N}" final answer and compares to ground
truth. Numeric equivalence first (within tolerance); falls back to string
match. Capability vector: correctness (single).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from elmo.eval.function_calling import ScoreReport


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_HASH_RE = re.compile(r"####\s*([\-+]?\d[\d,.\s/]*)")
_NUM_RE = re.compile(r"[\-+]?\d[\d,]*\.?\d*")


def _extract_answer(text: str) -> str | None:
    m = _BOXED_RE.search(text) or _HASH_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".,")
    # Last number in the text
    nums = _NUM_RE.findall(text)
    if not nums:
        return None
    return nums[-1].rstrip(".,")


def _to_number(s: str) -> float | None:
    s = s.strip().replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def math_equivalent(predicted: str | None, expected: str) -> bool:
    if predicted is None:
        return False
    p, e = _to_number(predicted), _to_number(expected)
    if p is not None and e is not None:
        return abs(p - e) < 1e-6
    return predicted.strip() == expected.strip()


class MathEvaluator:
    def __init__(self, eval_jsonl: Path):
        self.rows = [json.loads(line) for line in eval_jsonl.read_text().splitlines() if line.strip()]

    def _build_prompt(self, row: dict) -> str:
        return json.dumps({"messages": [
            {"role": "system", "content": row.get("system", "Solve the math problem. Show your work, then give the final answer after '#### '.")},
            {"role": "user", "content": row["problem"]},
        ]})

    def evaluate(self, generate: Callable[[list[str]], list[str]], max_examples: int | None = None) -> ScoreReport:
        rows = self.rows if max_examples is None else self.rows[:max_examples]
        prompts = [self._build_prompt(r) for r in rows]
        outputs = generate(prompts)
        per_example: list[dict] = []
        hits = 0
        for row, out in zip(rows, outputs):
            predicted = _extract_answer(out)
            ok = math_equivalent(predicted, row["answer"])
            if ok:
                hits += 1
            per_example.append({
                "query": row["problem"][:120],
                "expected": row["answer"],
                "predicted": predicted,
                "c1": 1.0 if ok else 0.0,
                "c2": 1.0 if ok else 0.0,
                "raw_output": out[:280],
            })
        n = len(rows)
        score = hits / max(1, n)
        return ScoreReport(
            n=n,
            tool_selection=score,
            arguments=score,
            parallel_calls=0.0,
            overall=score,
            per_example=per_example,
        )
