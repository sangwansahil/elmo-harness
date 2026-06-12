"""Tests for the foundry — filter, provenance, planner JSON extraction."""

from __future__ import annotations

import json
from pathlib import Path

from elmo.foundry.filter import filter_row
from elmo.foundry.provenance import ProvenanceLog


def test_filter_accepts_canonical_row() -> None:
    row = {
        "tools": [{"name": "get_weather", "description": "...", "parameters": {}}],
        "query": "What's the weather in Tokyo?",
        "answers": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
    }
    report = filter_row(row)
    assert report.passed, report.reasons


def test_filter_rejects_missing_tools() -> None:
    row = {"tools": [], "query": "hi", "answers": []}
    assert not filter_row(row).passed


def test_filter_rejects_answer_not_in_tools() -> None:
    row = {
        "tools": [{"name": "a", "parameters": {}}],
        "query": "do b",
        "answers": [{"name": "b", "arguments": {}}],
    }
    report = filter_row(row)
    assert not report.passed
    assert any("not in tools" in r for r in report.reasons)


def test_filter_rejects_args_not_dict() -> None:
    row = {
        "tools": [{"name": "a", "parameters": {}}],
        "query": "do a",
        "answers": [{"name": "a", "arguments": "x"}],
    }
    assert not filter_row(row).passed


def test_filter_allows_refusal_empty_answers() -> None:
    row = {
        "tools": [{"name": "a", "parameters": {}}],
        "query": "do something unrelated",
        "answers": [],
    }
    assert filter_row(row, allow_refusal=True).passed
    assert not filter_row(row, allow_refusal=False).passed


def test_provenance_log_appends(tmp_path: Path) -> None:
    log = ProvenanceLog(tmp_path / "prov.jsonl")
    log.write(
        row_id="r1",
        scenario_id="s_001",
        brief_id="b_abc",
        planner_model="claude-opus",
        generator_model="deepseek",
        generator_tokens=(120, 80),
        verifier_passed=True,
        verifier_reasons=[],
        seed_prompt_hash=ProvenanceLog.hash_prompt("weather in tokyo"),
        extra={"capability": "tool_selection"},
    )
    log.write(
        row_id="r2",
        scenario_id="s_002",
        brief_id="b_abc",
        planner_model="claude-opus",
        generator_model="deepseek",
        generator_tokens=(100, 70),
        verifier_passed=False,
        verifier_reasons=["tools missing or empty"],
    )
    lines = [json.loads(ln) for ln in (tmp_path / "prov.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["verifier_passed"] is True
    assert lines[1]["verifier_reasons"] == ["tools missing or empty"]
    assert log.n == 2


def test_planner_json_extraction_strips_fences() -> None:
    from elmo.foundry.planner import _extract_json_array

    text = (
        "Here are the scenarios:\n"
        "```json\n"
        '[{"id": "s_001", "capability": "tool_selection", "scenario": "..."}]\n'
        "```\n"
    )
    arr = _extract_json_array(text)
    assert len(arr) == 1
    assert arr[0]["id"] == "s_001"


def test_planner_json_extraction_balanced_brackets() -> None:
    from elmo.foundry.planner import _extract_json_array

    text = '[{"id":"a","tool_schema":{"x":[1,2,3]}}]'
    arr = _extract_json_array(text)
    assert arr[0]["tool_schema"]["x"] == [1, 2, 3]


def test_generator_json_extraction_strips_prose() -> None:
    from elmo.foundry.generator import _extract_json_object

    text = (
        "Here is the row:\n"
        '{"tools":[{"name":"a"}],"query":"q","answers":[{"name":"a","arguments":{}}]}'
    )
    obj = _extract_json_object(text)
    assert obj["query"] == "q"
    assert obj["answers"][0]["name"] == "a"
