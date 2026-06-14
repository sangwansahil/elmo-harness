"""Tests for the wizard task-discovery layer."""

from __future__ import annotations

from elmo.wizard import classify_prompt, discover_task


def test_classify_routes_function_calling() -> None:
    g = classify_prompt("build a function-calling expert that calls REST APIs")
    assert g.kind == "function-calling"
    assert g.confidence > 0
    assert any(k in {"function", "api", "rest", "call"} for k in g.matched_keywords)


def test_classify_routes_math() -> None:
    g = classify_prompt("solve grade-school math word problems step by step")
    assert g.kind == "math"
    assert g.confidence > 0


def test_classify_routes_structured() -> None:
    g = classify_prompt("extract structured json from messy invoices")
    assert g.kind == "structured"
    assert g.confidence > 0


def test_classify_falls_back_to_general() -> None:
    g = classify_prompt("teach the model to write haiku about cats")
    assert g.kind == "general"
    assert g.confidence == 0.0


def test_discover_function_calling_spec_shape() -> None:
    spec = discover_task("build me a function-calling assistant", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    assert spec.dataset.source.startswith("hf:Salesforce/xlam")
    assert spec.eval.benchmark == "bfcl-simple"
    cap_names = {c.name for c in spec.capabilities}
    assert "tool_selection" in cap_names
    assert "arguments" in cap_names


def test_discover_math_spec_shape() -> None:
    spec = discover_task("solve math problems", "mlx-community/Qwen2.5-Math-1.5B-Instruct-4bit")
    assert spec.dataset.source.startswith("hf:openai/gsm8k")
    assert spec.eval.benchmark == "gsm8k"
    assert spec.capabilities[0].name == "correctness"


def test_discover_structured_spec_shape() -> None:
    spec = discover_task("extract structured json from invoices", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    assert spec.dataset.source == "synthetic:structured"
    assert spec.eval.benchmark == "json-format"
    cap_names = {c.name for c in spec.capabilities}
    assert {"parseable", "keys_present", "types_correct"}.issubset(cap_names)


def test_discover_general_enables_foundry_with_from_prompt_source() -> None:
    spec = discover_task("write a haiku about owls", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    assert spec.dataset.source == "synthetic:from-prompt"
    assert spec.foundry.enabled is True
    assert spec.foundry.scenarios_per_brief >= 50


def test_discover_preserves_base_model_id() -> None:
    base = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    spec = discover_task("solve math problems", base)
    assert spec.base_model == base


def test_discover_name_is_slug_of_prompt() -> None:
    spec = discover_task("solve grade-school math word problems!!!", "x")
    assert spec.name == "solve-grade-school-math-word-problems"
