"""Verifier-first scoring primitives. Cheap, deterministic, non-gameable."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class CallVerdict:
    name_match: bool
    args_match: bool
    extra: bool = False


def _iter_balanced_json_objects(text: str):
    """Yield candidate JSON-object substrings using a brace-depth scanner."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start : i + 1]
                start = -1


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from a model output. Tolerant of formatting drift."""
    calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    if calls:
        return calls
    # Fallback: any balanced JSON object containing a top-level "name" key.
    for candidate in _iter_balanced_json_objects(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(obj)
    return calls


def _normalize(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _normalize(v[k]) for k in sorted(v)}
    if isinstance(v, list):
        return [_normalize(x) for x in v]
    if isinstance(v, str):
        return v.strip()
    return v


def args_equivalent(a: dict, b: dict) -> bool:
    return _normalize(a) == _normalize(b)


def compare_calls(predicted: list[dict], expected: list[dict]) -> list[CallVerdict]:
    """Order-insensitive comparison. Returns one verdict per expected call."""
    verdicts: list[CallVerdict] = []
    used: set[int] = set()
    for exp in expected:
        best_idx: int | None = None
        best = CallVerdict(name_match=False, args_match=False)
        for i, pred in enumerate(predicted):
            if i in used:
                continue
            name_ok = pred.get("name") == exp.get("name")
            args_ok = args_equivalent(
                pred.get("arguments", {}) or {},
                exp.get("arguments", {}) or {},
            )
            if name_ok and args_ok:
                best, best_idx = CallVerdict(True, True), i
                break
            if name_ok and not best.name_match:
                best, best_idx = CallVerdict(True, False), i
        if best_idx is not None:
            used.add(best_idx)
        verdicts.append(best)
    # Mark extra predicted calls
    for i in range(len(predicted)):
        if i not in used:
            verdicts.append(CallVerdict(False, False, extra=True))
    return verdicts
