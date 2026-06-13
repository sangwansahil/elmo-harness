"""GSM8K loader — grade-school math word problems with verifiable final answers."""

from __future__ import annotations

import json
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "Solve the math problem. Think step by step, then give the final numeric "
    "answer on a new line in the form: #### <number>."
)

_ANSWER_RE = re.compile(r"####\s*([\-+]?\d[\d,.\s/]*)")


def _extract_gold(answer_field: str) -> str:
    m = _ANSWER_RE.search(answer_field)
    if m:
        return m.group(1).strip().replace(",", "")
    # Fallback: last token
    tokens = answer_field.strip().split()
    return tokens[-1] if tokens else ""


def load_gsm8k(
    max_rows: int | None = None,
    split: str = "train",
    cache_dir: Path | None = None,
) -> list[dict]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(
        "openai/gsm8k", "main", split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    rows: list[dict] = []
    for r in ds:
        problem = r.get("question", "")
        answer_full = r.get("answer", "")
        gold = _extract_gold(answer_full)
        if not problem or not gold:
            continue
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": answer_full},
            ],
            "_eval": {
                "problem": problem,
                "answer": gold,
                "system": SYSTEM_PROMPT,
            },
        })
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def write_sft_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
            n += 1
    return n


def write_eval_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r["_eval"]) + "\n")
            n += 1
    return n
