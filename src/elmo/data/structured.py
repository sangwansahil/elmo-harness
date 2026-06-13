"""Tiny synthetic dataset for the structured-output task.

For phase 7 we don't want to require a network fetch just to demo the
structured-output evaluator end-to-end. This module generates a small fixed
set of (instruction, schema, gold) rows deterministically so a user can
run the full loop offline. Real users should swap in their own dataset for
production runs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "Respond with valid JSON only that satisfies the given schema. "
    "No markdown fences, no prose."
)


_SEED_TEMPLATES: list[dict] = [
    {
        "instruction": "Extract person details from: 'Maya Rodriguez, VP of Engineering at Acme Corp, age 38'",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "title": {"type": "string"},
                "company": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "title", "company", "age"],
        },
        "gold": {"name": "Maya Rodriguez", "title": "VP of Engineering", "company": "Acme Corp", "age": 38},
    },
    {
        "instruction": "Parse this restaurant entry: 'Sushi Hana, Japanese, 4.6/5, $$, 1421 Mission St, San Francisco'",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "cuisine": {"type": "string"},
                "rating": {"type": "number"},
                "price_tier": {"type": "string"},
                "address": {"type": "string"},
                "city": {"type": "string"},
            },
            "required": ["name", "cuisine", "rating", "city"],
        },
        "gold": {"name": "Sushi Hana", "cuisine": "Japanese", "rating": 4.6, "price_tier": "$$",
                 "address": "1421 Mission St", "city": "San Francisco"},
    },
    {
        "instruction": "Convert this booking: '2 nights at Hotel Tropico, check-in 2026-08-12, 1 king bed'",
        "schema": {
            "type": "object",
            "properties": {
                "hotel": {"type": "string"},
                "nights": {"type": "integer"},
                "check_in": {"type": "string"},
                "bed": {"type": "string"},
            },
            "required": ["hotel", "nights", "check_in", "bed"],
        },
        "gold": {"hotel": "Hotel Tropico", "nights": 2, "check_in": "2026-08-12", "bed": "1 king"},
    },
    {
        "instruction": "Order: 'I'd like a large oat milk latte, 2 shots, with vanilla syrup'",
        "schema": {
            "type": "object",
            "properties": {
                "drink": {"type": "string"},
                "size": {"type": "string"},
                "milk": {"type": "string"},
                "shots": {"type": "integer"},
                "syrups": {"type": "array"},
            },
            "required": ["drink", "size", "milk", "shots"],
        },
        "gold": {"drink": "latte", "size": "large", "milk": "oat", "shots": 2, "syrups": ["vanilla"]},
    },
    {
        "instruction": "Parse repo: 'sangwansahil/elmo-harness, 2,140 stars, MIT-compatible Apache-2.0, Python'",
        "schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "stars": {"type": "integer"},
                "license": {"type": "string"},
                "language": {"type": "string"},
            },
            "required": ["repo", "stars", "license", "language"],
        },
        "gold": {"repo": "sangwansahil/elmo-harness", "stars": 2140, "license": "Apache-2.0", "language": "Python"},
    },
    {
        "instruction": "Address book entry: 'Dr. Wei Chen, +1 (415) 555-0143, wei@chen.dev, Berkeley CA'",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "phone": {"type": "string"},
                "email": {"type": "string"},
                "location": {"type": "string"},
            },
            "required": ["name", "phone", "email", "location"],
        },
        "gold": {"name": "Dr. Wei Chen", "phone": "+1 (415) 555-0143", "email": "wei@chen.dev",
                 "location": "Berkeley CA"},
    },
]


def load_structured_seed(
    max_rows: int | None = None,
    split: str = "train",
    cache_dir: Path | None = None,
) -> list[dict]:
    """Return up to `max_rows` synthetic structured-output training rows.

    We replicate the seed templates with light deterministic variation so
    a model has enough examples to actually train on.
    """
    rng = random.Random(42)
    rows: list[dict] = []
    target = max_rows or 200
    while len(rows) < target:
        for tpl in _SEED_TEMPLATES:
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": tpl["instruction"] + "\n\nSchema:\n" + json.dumps(tpl["schema"], indent=2)},
                    {"role": "assistant", "content": json.dumps(tpl["gold"])},
                ],
                "_eval": {
                    "instruction": tpl["instruction"],
                    "schema": tpl["schema"],
                    "gold": tpl["gold"],
                    "system": SYSTEM_PROMPT,
                },
            })
            if len(rows) >= target:
                break
    rng.shuffle(rows)
    return rows[:target]


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
