"""Tests for the verifier-as-reward function."""

from __future__ import annotations

import pytest

from elmo.reward import batch_function_call_rewards, function_call_reward


def _tc(name: str, args: dict) -> str:
    import json
    return f'<tool_call>{json.dumps({"name": name, "arguments": args})}</tool_call>'


def test_full_match_is_one() -> None:
    completion = _tc("get_weather", {"city": "Tokyo"})
    r = function_call_reward(completion, [{"name": "get_weather", "arguments": {"city": "Tokyo"}}])
    assert r == 1.0


def test_correct_name_wrong_args_is_partial() -> None:
    completion = _tc("get_weather", {"city": "Osaka"})
    r = function_call_reward(completion, [{"name": "get_weather", "arguments": {"city": "Tokyo"}}])
    assert 0.5 < r < 0.8


def test_wrong_name_is_low() -> None:
    completion = _tc("get_time", {"city": "Tokyo"})
    r = function_call_reward(completion, [{"name": "get_weather", "arguments": {"city": "Tokyo"}}])
    assert 0.0 < r < 0.5


def test_unparseable_completion_is_zero() -> None:
    r = function_call_reward("I will look that up.", [{"name": "get_weather", "arguments": {}}])
    assert r == 0.0


def test_correct_refusal_is_one() -> None:
    r = function_call_reward("I cannot help with that.", [])
    assert r == 1.0


def test_extra_spurious_call_penalized() -> None:
    completion = _tc("get_weather", {"city": "Tokyo"}) + _tc("get_time", {"city": "Tokyo"})
    r_full = function_call_reward(
        _tc("get_weather", {"city": "Tokyo"}),
        [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
    )
    r_with_extra = function_call_reward(
        completion, [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
    )
    assert r_with_extra < r_full


def test_parallel_full_match() -> None:
    completion = _tc("a", {"x": 1}) + _tc("b", {"y": 2})
    expected = [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    r = function_call_reward(completion, expected)
    assert r == 1.0


def test_parallel_partial_credit() -> None:
    completion = _tc("a", {"x": 1})  # missing the second expected call
    expected = [
        {"name": "a", "arguments": {"x": 1}},
        {"name": "b", "arguments": {"y": 2}},
    ]
    r = function_call_reward(completion, expected)
    # one perfect match + one missing-with-predictions-present (0.30 floor)
    # averaged → ~0.65, strictly between a full miss and a full hit
    assert 0.5 < r < 0.8


def test_batch_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        batch_function_call_rewards(["a"], [[{"name": "x", "arguments": {}}], []])


def test_batch_shape() -> None:
    rs = batch_function_call_rewards(
        [_tc("a", {"x": 1}), "nope"],
        [
            [{"name": "a", "arguments": {"x": 1}}],
            [{"name": "b", "arguments": {}}],
        ],
    )
    assert rs[0] == 1.0
    assert rs[1] == 0.0
