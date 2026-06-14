"""xLAM-function-calling-60k loader, formatted as chat messages with tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a function-calling assistant. You have access to the following tools. "
    "Use them when the user's request requires it. Respond with one or more tool "
    "calls in the form: <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>. "
    "If no tool fits, answer in plain text."
)


def _safe_json(s: Any) -> Any:
    if isinstance(s, (list, dict)):
        return s
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _format_tools(tools: list[dict]) -> str:
    lines = ["Available tools:"]
    for t in tools:
        name = t.get("name", "?")
        desc = t.get("description", "")
        params = t.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        if params:
            lines.append(f"  parameters: {json.dumps(params)}")
    return "\n".join(lines)


def _format_answers(answers: list[dict]) -> str:
    parts: list[str] = []
    for a in answers:
        call = {"name": a.get("name"), "arguments": a.get("arguments", {})}
        parts.append(f"<tool_call>{json.dumps(call)}</tool_call>")
    return "\n".join(parts)


def _row_to_messages(row: dict) -> dict | None:
    tools = _safe_json(row.get("tools"))
    answers = _safe_json(row.get("answers"))
    query = row.get("query")
    if not (isinstance(tools, list) and isinstance(answers, list) and isinstance(query, str)):
        return None
    if not answers:
        return None
    system = f"{SYSTEM_PROMPT}\n\n{_format_tools(tools)}"
    assistant = _format_answers(answers)
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
            {"role": "assistant", "content": assistant},
        ],
        "_eval": {
            "query": query,
            "tools": tools,
            "expected_calls": [
                {"name": a.get("name"), "arguments": a.get("arguments", {})}
                for a in answers
            ],
            "system": system,
        },
    }


def load_xlam_function_calling(
    max_rows: int | None = None,
    split: str = "train",
    cache_dir: Path | None = None,
) -> list[dict]:
    """Load xLAM-function-calling-60k and format each row for chat-style SFT.

    Returns a list of {"messages": [...], "_eval": {...}} dicts. The "_eval" key
    is stripped before writing training jsonl; it is retained so the same rows
    can drive evaluation.

    xLAM is gated on Hugging Face. You need to: (1) request access on
    https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k,
    (2) set HF_TOKEN in your env. If you'd rather avoid that, pick the
    `synthetic:function-calling` source — the wizard uses that by default.
    """
    import os

    from datasets import load_dataset  # type: ignore

    token = os.environ.get("HF_TOKEN")
    try:
        ds = load_dataset(
            "Salesforce/xlam-function-calling-60k",
            split=split,
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )
    except Exception as e:
        msg = str(e)
        if "gated" in msg.lower() or "401" in msg or "403" in msg or "Access" in msg:
            raise RuntimeError(
                "xLAM is a gated dataset on Hugging Face. "
                "Either (a) accept the terms at "
                "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k "
                "and set HF_TOKEN in your env, "
                "or (b) edit the spec to use 'synthetic:function-calling' "
                "(the wizard default — works offline, no auth required)."
            ) from e
        raise
    rows: list[dict] = []
    for r in ds:
        formatted = _row_to_messages(dict(r))
        if formatted:
            rows.append(formatted)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def write_sft_jsonl(rows: list[dict], path: Path) -> int:
    """Write the SFT-ready jsonl (no _eval key). Returns row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
            n += 1
    return n


def write_eval_jsonl(rows: list[dict], path: Path) -> int:
    """Write eval rows (query, tools, expected_calls, system)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r["_eval"]) + "\n")
            n += 1
    return n
