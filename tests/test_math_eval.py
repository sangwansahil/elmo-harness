"""Tests for the math + structured evaluators (parsing + scoring)."""

from __future__ import annotations

from pathlib import Path

from elmo.eval.math import _extract_answer, math_equivalent
from elmo.eval.structured import _extract_json, _score_against_schema


def test_extract_hash_answer() -> None:
    assert _extract_answer("Some reasoning. #### 42") == "42"


def test_extract_boxed_answer() -> None:
    assert _extract_answer("The answer is \\boxed{17}.") == "17"


def test_extract_last_number_fallback() -> None:
    assert _extract_answer("After all that, I get 31.") == "31"


def test_extract_none_for_no_number() -> None:
    assert _extract_answer("I do not know.") is None


def test_math_equivalent_numeric() -> None:
    assert math_equivalent("42", "42")
    assert math_equivalent("42.0", "42")
    assert math_equivalent("1,234", "1234")
    assert not math_equivalent("42", "43")


def test_math_equivalent_string_fallback() -> None:
    assert math_equivalent("yes", "yes")
    assert not math_equivalent("yes", "no")


def test_math_equivalent_predicted_none() -> None:
    assert not math_equivalent(None, "42")


def test_extract_json_canonical() -> None:
    obj = _extract_json('{"a": 1, "b": "x"}')
    assert obj == {"a": 1, "b": "x"}


def test_extract_json_fenced() -> None:
    obj = _extract_json('```json\n{"a": [1,2,3]}\n```')
    assert obj == {"a": [1, 2, 3]}


def test_extract_json_with_prose() -> None:
    obj = _extract_json('Here is the result: {"a": "b"} thank you')
    assert obj == {"a": "b"}


def test_extract_json_unparseable_returns_none() -> None:
    assert _extract_json("definitely not json") is None


def test_score_schema_full_match() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    parseable, kc, tc = _score_against_schema({"name": "Maya", "age": 38}, schema)
    assert parseable and kc == 1.0 and tc == 1.0


def test_score_schema_missing_key() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    parseable, kc, tc = _score_against_schema({"name": "Maya"}, schema)
    assert parseable and kc == 0.5 and tc == 0.5


def test_score_schema_wrong_type() -> None:
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer"}},
        "required": ["age"],
    }
    parseable, kc, tc = _score_against_schema({"age": "thirty"}, schema)
    assert parseable and kc == 1.0 and tc == 0.0


def test_score_schema_unparseable_is_zero() -> None:
    parseable, kc, tc = _score_against_schema(None, {"type": "object", "required": ["x"]})
    assert not parseable and kc == 0.0 and tc == 0.0
