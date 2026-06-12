"""Tests for the verifier — tool-call parsing and matching."""

from __future__ import annotations

from elmo.eval.verifier import args_equivalent, compare_calls, parse_tool_calls


def test_parse_canonical() -> None:
    text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


def test_parse_multiple() -> None:
    text = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    assert len(parse_tool_calls(text)) == 2


def test_parse_loose_fallback() -> None:
    text = 'I will call {"name": "get_weather", "arguments": {"city": "Tokyo"}}'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"


def test_args_equivalent_key_order() -> None:
    assert args_equivalent({"a": 1, "b": 2}, {"b": 2, "a": 1})


def test_args_equivalent_string_strip() -> None:
    assert args_equivalent({"name": "Tokyo  "}, {"name": "Tokyo"})


def test_compare_calls_full_match() -> None:
    pred = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    exp = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    verdicts = compare_calls(pred, exp)
    assert verdicts[0].name_match and verdicts[0].args_match


def test_compare_calls_name_only() -> None:
    pred = [{"name": "get_weather", "arguments": {"city": "Osaka"}}]
    exp = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]
    verdicts = compare_calls(pred, exp)
    assert verdicts[0].name_match
    assert not verdicts[0].args_match


def test_compare_calls_unordered_parallel() -> None:
    pred = [
        {"name": "b", "arguments": {"y": 2}},
        {"name": "a", "arguments": {"x": 1}},
    ]
    exp = [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    verdicts = compare_calls(pred, exp)
    assert all(v.name_match and v.args_match for v in verdicts[:2])
