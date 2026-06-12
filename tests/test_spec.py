"""Tests for the task spec parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from elmo.spec import TaskSpec, load_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_spec_loads() -> None:
    spec = load_spec(REPO_ROOT / "examples" / "function-calling.yaml")
    assert isinstance(spec, TaskSpec)
    assert spec.name == "function-calling"
    assert spec.base_model.startswith("mlx-community/")
    assert spec.dataset.source.startswith("hf:")
    assert len(spec.capabilities) >= 3
    assert any(c.verifier == "function_call" for c in spec.capabilities)


def test_invalid_spec_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")  # missing required fields
    with pytest.raises(ValueError):
        load_spec(bad)


def test_dump_roundtrip(tmp_path: Path) -> None:
    from elmo.spec import dump_spec

    spec = load_spec(REPO_ROOT / "examples" / "function-calling.yaml")
    out = tmp_path / "out.yaml"
    dump_spec(spec, out)
    reloaded = load_spec(out)
    assert reloaded.model_dump() == spec.model_dump()
