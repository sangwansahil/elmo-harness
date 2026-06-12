"""Verifier-first row filter — accept or reject a generated row deterministically.

For function calling, a row is valid iff:
  - 'tools' is a non-empty list of dicts with a 'name' string
  - 'query' is a non-empty string
  - 'answers' is a list (may be empty for refusal scenarios)
  - every answer has a 'name' that exists in tools
  - every answer's 'arguments' is a dict
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FilterReport:
    passed: bool
    reasons: list[str]


def filter_row(row: dict, *, allow_refusal: bool = True) -> FilterReport:
    reasons: list[str] = []

    tools = row.get("tools")
    if not isinstance(tools, list) or not tools:
        reasons.append("tools missing or empty")
    else:
        for i, t in enumerate(tools):
            if not isinstance(t, dict):
                reasons.append(f"tools[{i}] not a dict")
                continue
            name = t.get("name")
            if not isinstance(name, str) or not name:
                reasons.append(f"tools[{i}].name missing")

    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        reasons.append("query missing or empty")

    answers = row.get("answers")
    if not isinstance(answers, list):
        reasons.append("answers must be a list")
        return FilterReport(passed=False, reasons=reasons)

    if not answers and not allow_refusal:
        reasons.append("answers empty (refusal not allowed)")

    valid_names = {
        t.get("name") for t in (tools or []) if isinstance(t, dict) and isinstance(t.get("name"), str)
    }
    for i, a in enumerate(answers):
        if not isinstance(a, dict):
            reasons.append(f"answers[{i}] not a dict")
            continue
        name = a.get("name")
        if not isinstance(name, str) or not name:
            reasons.append(f"answers[{i}].name missing")
        elif valid_names and name not in valid_names:
            reasons.append(f"answers[{i}].name '{name}' not in tools")
        args = a.get("arguments")
        if not isinstance(args, dict):
            reasons.append(f"answers[{i}].arguments must be a dict")

    return FilterReport(passed=not reasons, reasons=reasons)
