"""BFCL-style function-calling evaluation.

A simplified, self-contained evaluator that scores:
- C1 tool selection: function name exact match
- C2 argument extraction: structural args equality
- C3 parallel calls: set equality across calls

For Phase 0. We will swap in the real BFCL harness in Phase 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from elmo.eval.verifier import compare_calls, parse_tool_calls


@dataclass
class ScoreReport:
    n: int
    tool_selection: float
    arguments: float
    parallel_calls: float
    overall: float
    per_example: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "tool_selection": round(self.tool_selection, 4),
            "arguments": round(self.arguments, 4),
            "parallel_calls": round(self.parallel_calls, 4),
            "overall": round(self.overall, 4),
        }


GenerateFn = Callable[[list[str]], list[str]]


class FunctionCallEvaluator:
    """Drives a `generate` callable across an eval jsonl and scores outputs."""

    def __init__(self, eval_jsonl: Path):
        self.rows = [json.loads(line) for line in eval_jsonl.read_text().splitlines() if line.strip()]

    def _build_prompt(self, row: dict) -> str:
        # Plain-text prompt; backends apply the chat template at generation time.
        # We rely on the tokenizer's apply_chat_template when available, so here
        # we just emit a chat-shaped string the backend can format.
        return json.dumps({
            "messages": [
                {"role": "system", "content": row["system"]},
                {"role": "user", "content": row["query"]},
            ]
        })

    def evaluate(self, generate: GenerateFn, max_examples: int | None = None) -> ScoreReport:
        rows = self.rows if max_examples is None else self.rows[:max_examples]
        prompts = [self._build_prompt(r) for r in rows]
        outputs = generate(prompts)

        per_example: list[dict] = []
        c1_hits = c2_hits = c3_hits = 0
        for row, out in zip(rows, outputs, strict=False):
            predicted = parse_tool_calls(out)
            expected = row["expected_calls"]
            verdicts = compare_calls(predicted, expected)

            expected_v = verdicts[: len(expected)]
            name_hits = sum(1 for v in expected_v if v.name_match)
            full_hits = sum(1 for v in expected_v if v.name_match and v.args_match)

            c1 = name_hits / max(1, len(expected))
            c2 = full_hits / max(1, len(expected))
            parallel_ok = (
                len(expected) > 1
                and full_hits == len(expected)
                and not any(v.extra for v in verdicts)
            )

            c1_hits += c1
            c2_hits += c2
            if len(expected) > 1:
                c3_hits += 1.0 if parallel_ok else 0.0

            per_example.append({
                "query": row["query"][:120],
                "expected": expected,
                "predicted": predicted,
                "c1": round(c1, 3),
                "c2": round(c2, 3),
                "raw_output": out[:280],
            })

        n = len(rows)
        n_parallel = sum(1 for r in rows if len(r["expected_calls"]) > 1) or 1
        return ScoreReport(
            n=n,
            tool_selection=c1_hits / max(1, n),
            arguments=c2_hits / max(1, n),
            parallel_calls=c3_hits / n_parallel,
            overall=c2_hits / max(1, n),
            per_example=per_example,
        )
