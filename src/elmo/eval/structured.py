"""Structured-output evaluator — JSON-schema validity + key completeness."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from elmo.eval.function_calling import ScoreReport


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | list | None:
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Find first '{' or '['
    start_obj = text.find("{")
    start_arr = text.find("[")
    candidates = [s for s in (start_obj, start_arr) if s != -1]
    if not candidates:
        return None
    start = min(candidates)
    depth = 0
    end = -1
    in_str = False
    esc = False
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _type_ok(value, type_name: str) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    return True


def _score_against_schema(parsed: dict | list | None, schema: dict) -> tuple[bool, float, float]:
    """Return (parseable, key_completeness, type_correctness) in [0,1]."""
    if parsed is None:
        return False, 0.0, 0.0
    if not isinstance(parsed, dict) or schema.get("type") != "object":
        # We only support object root schemas here for Phase 7.
        return True, 0.0, 0.0
    props = schema.get("properties", {})
    required = schema.get("required") or list(props)
    if not required:
        return True, 1.0, 1.0
    present = sum(1 for k in required if k in parsed)
    type_ok = 0
    for k in required:
        spec_t = props.get(k, {}).get("type")
        if k in parsed and spec_t and _type_ok(parsed[k], spec_t):
            type_ok += 1
    return True, present / len(required), type_ok / len(required)


class StructuredEvaluator:
    def __init__(self, eval_jsonl: Path):
        self.rows = [json.loads(line) for line in eval_jsonl.read_text().splitlines() if line.strip()]

    def _build_prompt(self, row: dict) -> str:
        return json.dumps({"messages": [
            {"role": "system", "content": row.get("system", "Respond with valid JSON only that satisfies the given schema. No prose.")},
            {"role": "user", "content": row["instruction"] + "\n\nSchema:\n" + json.dumps(row["schema"], indent=2)},
        ]})

    def evaluate(self, generate: Callable[[list[str]], list[str]], max_examples: int | None = None) -> ScoreReport:
        rows = self.rows if max_examples is None else self.rows[:max_examples]
        prompts = [self._build_prompt(r) for r in rows]
        outputs = generate(prompts)
        per_example: list[dict] = []
        parse_hits = key_sum = type_sum = 0.0
        for row, out in zip(rows, outputs):
            parsed = _extract_json(out)
            parseable, kc, tc = _score_against_schema(parsed, row["schema"])
            parse_hits += 1.0 if parseable else 0.0
            key_sum += kc
            type_sum += tc
            per_example.append({
                "query": row["instruction"][:120],
                "expected": row["schema"],
                "predicted": parsed,
                "c1": 1.0 if parseable else 0.0,
                "c2": tc,
                "raw_output": out[:280],
            })
        n = len(rows)
        return ScoreReport(
            n=n,
            tool_selection=parse_hits / max(1, n),  # "parseable"
            arguments=key_sum / max(1, n),           # "keys present"
            parallel_calls=type_sum / max(1, n),    # "types right"
            overall=type_sum / max(1, n),
            per_example=per_example,
        )
