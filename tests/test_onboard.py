"""Tests for the onboarding-flow modules: system probe shape, catalog
recommendation, hub round-trip."""

from __future__ import annotations

from pathlib import Path

from elmo.catalog import CATALOG, estimate_tokps, fits, recommend
from elmo.hub import HubEntry, list_models, register_base, remove, save_finetune
from elmo.system import SystemProbe, probe


def _probe(**kw) -> SystemProbe:
    defaults = dict(
        os="darwin", arch="arm64", chip="Apple M5 Pro", chip_class="apple-silicon",
        chip_tier="pro", ram_gb=24.0, free_disk_gb=200.0, gpu_name="", gpu_vram_gb=0.0,
        suggested_backend="mlx",
    )
    defaults.update(kw)
    return SystemProbe(**defaults)


def test_system_probe_returns_shape() -> None:
    p = probe()
    assert isinstance(p, SystemProbe)
    assert p.os in ("darwin", "linux", "windows")
    assert p.ram_gb >= 0
    assert p.chip_class in ("apple-silicon", "intel", "amd", "nvidia", "other")


def test_fits_on_m_pro() -> None:
    m = next(m for m in CATALOG if m.id == "qwen-1_5b")
    assert fits(m, _probe(ram_gb=24.0))


def test_fits_blocks_oversize() -> None:
    m = next(m for m in CATALOG if m.id == "qwen-7b")
    # 4GB total RAM is way too small for a 7B model
    assert not fits(m, _probe(ram_gb=4.0))


def test_tokps_scales_with_chip_tier() -> None:
    m = next(m for m in CATALOG if m.id == "qwen-1_5b")
    base = estimate_tokps(m, _probe(chip_tier="base"))
    pro = estimate_tokps(m, _probe(chip_tier="pro"))
    max_ = estimate_tokps(m, _probe(chip_tier="max"))
    assert base < pro < max_


def test_tokps_inverse_with_params() -> None:
    p = _probe()
    small = estimate_tokps(next(m for m in CATALOG if m.id == "qwen-0_5b"), p)
    big = estimate_tokps(next(m for m in CATALOG if m.id == "qwen-7b"), p)
    assert small > big


def test_recommend_marks_one_elmo_choice() -> None:
    recs = recommend(_probe())
    choices = [r for r in recs if r.elmo_choice]
    assert len(choices) == 1


def test_recommend_sorts_choice_first() -> None:
    recs = recommend(_probe())
    assert recs[0].elmo_choice is True


def test_recommend_orders_oversized_last() -> None:
    recs = recommend(_probe(ram_gb=4.0))
    # The 7B model should be marked as not fitting, and appear after fitting models
    fitting_count = sum(1 for r in recs if r.fits)
    if fitting_count and fitting_count < len(recs):
        assert all(r.fits for r in recs[:fitting_count])
        assert all(not r.fits for r in recs[fitting_count:])


def test_specialty_filter() -> None:
    recs_math = recommend(_probe(), specialty="math")
    ids = {r.model.id for r in recs_math}
    # math-specialty model should be present; code-specialty should not
    assert "qwen-math-1_5b" in ids
    assert "qwen-coder-1_5b" not in ids


def test_hub_register_idempotent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ELMO_HUB_ROOT", str(tmp_path))
    a = register_base(hf_id="X/Y", display_name="Y", path=tmp_path / "weights")
    b = register_base(hf_id="X/Y", display_name="Y", path=tmp_path / "weights")
    assert a.id == b.id
    assert len(list_models()) == 1


def test_hub_save_finetune_links_base(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ELMO_HUB_ROOT", str(tmp_path))
    base = register_base(hf_id="org/base-model", display_name="Base", path=tmp_path / "b")
    ft = save_finetune(
        display_name="My FT",
        base_hf_id="org/base-model",
        adapter_path=tmp_path / "adapter",
        task_name="function-calling",
        run_id="run_abc",
        baseline_score=0.5,
        final_score=0.72,
    )
    assert ft.base_id == base.id
    assert ft.kind == "fine-tuned"
    assert ft.final_score == 0.72


def test_hub_list_by_kind(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ELMO_HUB_ROOT", str(tmp_path))
    register_base(hf_id="X/A", display_name="A", path=tmp_path / "a")
    register_base(hf_id="X/B", display_name="B", path=tmp_path / "b")
    save_finetune(
        display_name="FT1", base_hf_id="X/A",
        adapter_path=tmp_path / "ad", task_name="t", run_id="r1",
    )
    assert len(list_models(kind="base")) == 2
    assert len(list_models(kind="fine-tuned")) == 1


def test_hub_remove(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ELMO_HUB_ROOT", str(tmp_path))
    e = register_base(hf_id="X/Z", display_name="Z", path=tmp_path / "z")
    assert remove(e.id)
    assert not remove(e.id)  # already gone


def test_hub_entry_dataclass_shape() -> None:
    e = HubEntry(id="x", kind="base", display_name="d")
    # default fields populated
    assert e.hf_id == ""
    assert e.added_at > 0
