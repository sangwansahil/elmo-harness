"""Tiny synthetic function-calling seed dataset.

Used as a deterministic offline fallback for the wizard's function-calling
task kind when the user doesn't have a planner provider key (which would
otherwise drive the foundry to generate rows from the prompt).

Lean and minimal — twelve diverse tool schemas, replicated and shuffled so a
LoRA actually has something to learn from on a first run. Real production
use should reach for xLAM or Glaive once provider keys are wired.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a function-calling assistant. You have access to the tools "
    "listed below. Respond with one or more tool calls in the form: "
    "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>. "
    "If no tool fits, answer in plain text."
)


_SEEDS: list[dict] = [
    {
        "tools": [
            {"name": "get_weather", "description": "Look up current weather for a city.",
             "parameters": {"type": "object", "properties": {
                 "city": {"type": "string"}, "units": {"type": "string", "enum": ["c", "f"]}
             }, "required": ["city"]}},
        ],
        "query": "what's the weather in Tokyo right now?",
        "answers": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
    },
    {
        "tools": [
            {"name": "get_weather", "description": "Look up current weather for a city.",
             "parameters": {"type": "object", "properties": {
                 "city": {"type": "string"}, "units": {"type": "string"}
             }, "required": ["city"]}},
        ],
        "query": "weather in Osaka and Sapporo please",
        "answers": [
            {"name": "get_weather", "arguments": {"city": "Osaka"}},
            {"name": "get_weather", "arguments": {"city": "Sapporo"}},
        ],
    },
    {
        "tools": [
            {"name": "get_stock_price", "description": "Get the latest stock price.",
             "parameters": {"type": "object", "properties": {
                 "ticker": {"type": "string"}
             }, "required": ["ticker"]}},
        ],
        "query": "how is NVDA doing today",
        "answers": [{"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}],
    },
    {
        "tools": [
            {"name": "send_email", "description": "Send an email message.",
             "parameters": {"type": "object", "properties": {
                 "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}
             }, "required": ["to", "subject", "body"]}},
        ],
        "query": "email maya@acme.com saying I'll be late, subject 'running 15 minutes behind'",
        "answers": [{"name": "send_email", "arguments": {
            "to": "maya@acme.com", "subject": "running 15 minutes behind",
            "body": "I'll be late."
        }}],
    },
    {
        "tools": [
            {"name": "search_news", "description": "Search the news for recent articles.",
             "parameters": {"type": "object", "properties": {
                 "query": {"type": "string"}, "limit": {"type": "integer"}
             }, "required": ["query"]}},
        ],
        "query": "find five recent articles about quantum computing",
        "answers": [{"name": "search_news", "arguments": {"query": "quantum computing", "limit": 5}}],
    },
    {
        "tools": [
            {"name": "create_meeting", "description": "Schedule a meeting.",
             "parameters": {"type": "object", "properties": {
                 "title": {"type": "string"}, "time": {"type": "string"},
                 "attendees": {"type": "array", "items": {"type": "string"}}
             }, "required": ["title", "time", "attendees"]}},
        ],
        "query": "set up a 'Q3 planning' meeting next monday at 10am with jane@acme.com and dev@acme.com",
        "answers": [{"name": "create_meeting", "arguments": {
            "title": "Q3 planning", "time": "next monday 10am",
            "attendees": ["jane@acme.com", "dev@acme.com"]
        }}],
    },
    {
        "tools": [
            {"name": "translate", "description": "Translate text into another language.",
             "parameters": {"type": "object", "properties": {
                 "text": {"type": "string"}, "target_lang": {"type": "string"}
             }, "required": ["text", "target_lang"]}},
        ],
        "query": "translate 'good morning' to japanese",
        "answers": [{"name": "translate", "arguments": {
            "text": "good morning", "target_lang": "japanese"
        }}],
    },
    {
        "tools": [
            {"name": "get_directions", "description": "Get driving directions.",
             "parameters": {"type": "object", "properties": {
                 "origin": {"type": "string"}, "destination": {"type": "string"},
                 "mode": {"type": "string", "enum": ["driving", "walking", "transit"]}
             }, "required": ["origin", "destination"]}},
        ],
        "query": "how do i get from Mission to SOMA by transit",
        "answers": [{"name": "get_directions", "arguments": {
            "origin": "Mission", "destination": "SOMA", "mode": "transit"
        }}],
    },
    {
        "tools": [
            {"name": "calculate_tip", "description": "Compute a tip amount.",
             "parameters": {"type": "object", "properties": {
                 "bill": {"type": "number"}, "percent": {"type": "number"}
             }, "required": ["bill", "percent"]}},
        ],
        "query": "twenty percent on a $87 bill",
        "answers": [{"name": "calculate_tip", "arguments": {"bill": 87, "percent": 20}}],
    },
    {
        "tools": [
            {"name": "get_user_profile", "description": "Fetch a user profile by id.",
             "parameters": {"type": "object", "properties": {
                 "user_id": {"type": "string"}
             }, "required": ["user_id"]}},
        ],
        "query": "what's the profile for user u_8821",
        "answers": [{"name": "get_user_profile", "arguments": {"user_id": "u_8821"}}],
    },
    {
        "tools": [
            {"name": "summarize_document", "description": "Summarize a stored document.",
             "parameters": {"type": "object", "properties": {
                 "doc_id": {"type": "string"}, "max_words": {"type": "integer"}
             }, "required": ["doc_id"]}},
        ],
        "query": "give me a 100-word summary of doc_42",
        "answers": [{"name": "summarize_document", "arguments": {
            "doc_id": "doc_42", "max_words": 100
        }}],
    },
    {
        "tools": [
            {"name": "get_weather", "description": "Look up current weather.",
             "parameters": {"type": "object", "properties": {
                 "city": {"type": "string"}
             }, "required": ["city"]}},
            {"name": "get_time", "description": "Look up local time for a city.",
             "parameters": {"type": "object", "properties": {
                 "city": {"type": "string"}
             }, "required": ["city"]}},
        ],
        "query": "what time is it in london and what's the weather there",
        "answers": [
            {"name": "get_time", "arguments": {"city": "London"}},
            {"name": "get_weather", "arguments": {"city": "London"}},
        ],
    },
]


def _format_tools_block(tools: list[dict]) -> str:
    lines = ["Available tools:"]
    for t in tools:
        lines.append(f"- {t['name']}: {t.get('description', '')}")
        if t.get("parameters"):
            lines.append(f"  parameters: {json.dumps(t['parameters'])}")
    return "\n".join(lines)


def _format_answers_block(answers: list[dict]) -> str:
    return "\n".join(
        f"<tool_call>{json.dumps({'name': a['name'], 'arguments': a['arguments']})}</tool_call>"
        for a in answers
    )


def load_function_calling_seed(
    max_rows: int | None = None,
    split: str = "train",
    cache_dir: Path | None = None,
) -> list[dict]:
    """Return up to `max_rows` synthetic function-calling rows.

    Deterministic (seeded). Each row carries both an SFT-ready `messages`
    chat envelope and an `_eval` block compatible with FunctionCallEvaluator.
    """
    rng = random.Random(42)
    rows: list[dict] = []
    target = max_rows or 240
    while len(rows) < target:
        for tpl in _SEEDS:
            tools = tpl["tools"]
            answers = tpl["answers"]
            query = tpl["query"]
            system = f"{SYSTEM_PROMPT}\n\n{_format_tools_block(tools)}"
            rows.append({
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": _format_answers_block(answers)},
                ],
                "_eval": {
                    "query": query,
                    "tools": tools,
                    "expected_calls": answers,
                    "system": system,
                },
            })
            if len(rows) >= target:
                break
    rng.shuffle(rows)
    return rows[:target]
