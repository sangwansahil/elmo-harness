"""Curated catalog of small fine-tunable base models with speed + intelligence
scores. The recommender ranks them against the system probe.

Speed and intelligence are integer scores in [1,5]. They are deliberately
coarse: this is a starter UX, not a benchmark scoreboard. Numbers are
calibrated against published model-card benchmarks and apple-silicon
tokens-per-second runs observed in the wild as of mid-2026.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from elmo.system import SystemProbe


Specialty = Literal["general", "code", "math", "function-calling"]


@dataclass
class CatalogModel:
    id: str                  # internal slug
    display_name: str
    hf_id_mlx: str           # mlx-community 4-bit GGUF/MLX
    hf_id_raw: str           # original full-precision repo (for non-MLX backends)
    params_b: float          # parameter count in billions
    disk_4bit_gb: float      # approx download size for the 4-bit MLX variant
    ram_min_gb: float        # rough RAM required for inference + LoRA
    intelligence: int        # 1..5
    speed: int               # 1..5  (relative on the lowest-end recommended chip)
    specialty: Specialty = "general"
    note: str = ""

    def asdict(self) -> dict:
        return asdict(self)


CATALOG: list[CatalogModel] = [
    CatalogModel(
        id="qwen-0_5b",
        display_name="Qwen 2.5 0.5B",
        hf_id_mlx="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        hf_id_raw="Qwen/Qwen2.5-0.5B-Instruct",
        params_b=0.5, disk_4bit_gb=0.4, ram_min_gb=2.0,
        intelligence=1, speed=5,
        note="tiny, fast, good for snappy iteration on toy tasks.",
    ),
    CatalogModel(
        id="qwen-1_5b",
        display_name="Qwen 2.5 1.5B",
        hf_id_mlx="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        hf_id_raw="Qwen/Qwen2.5-1.5B-Instruct",
        params_b=1.5, disk_4bit_gb=1.0, ram_min_gb=4.0,
        intelligence=3, speed=4,
        note="sweet spot for first-time fine-tuning. ~60 MMLU, ~62 GSM8K out of the box.",
    ),
    CatalogModel(
        id="llama-3_2-1b",
        display_name="Llama 3.2 1B",
        hf_id_mlx="mlx-community/Llama-3.2-1B-Instruct-4bit",
        hf_id_raw="meta-llama/Llama-3.2-1B-Instruct",
        params_b=1.0, disk_4bit_gb=0.8, ram_min_gb=3.0,
        intelligence=1, speed=5,
        note="meta's smallest instruct model. fast but limited reasoning.",
    ),
    CatalogModel(
        id="llama-3_2-3b",
        display_name="Llama 3.2 3B",
        hf_id_mlx="mlx-community/Llama-3.2-3B-Instruct-4bit",
        hf_id_raw="meta-llama/Llama-3.2-3B-Instruct",
        params_b=3.0, disk_4bit_gb=2.0, ram_min_gb=6.0,
        intelligence=3, speed=3,
        note="solid baseline that beats most 7B's from a year ago.",
    ),
    CatalogModel(
        id="phi-3_5-mini",
        display_name="Phi 3.5 Mini",
        hf_id_mlx="mlx-community/Phi-3.5-mini-instruct-4bit",
        hf_id_raw="microsoft/Phi-3.5-mini-instruct",
        params_b=3.8, disk_4bit_gb=2.4, ram_min_gb=7.0,
        intelligence=3, speed=3,
        note="microsoft's reasoning-tuned mini; strong on logic.",
    ),
    CatalogModel(
        id="qwen-3b",
        display_name="Qwen 2.5 3B",
        hf_id_mlx="mlx-community/Qwen2.5-3B-Instruct-4bit",
        hf_id_raw="Qwen/Qwen2.5-3B-Instruct",
        params_b=3.0, disk_4bit_gb=1.9, ram_min_gb=6.0,
        intelligence=3, speed=3,
        note="qwen at 3b — slightly smarter than llama-3.2-3b on most evals.",
    ),
    CatalogModel(
        id="qwen-7b",
        display_name="Qwen 2.5 7B",
        hf_id_mlx="mlx-community/Qwen2.5-7B-Instruct-4bit",
        hf_id_raw="Qwen/Qwen2.5-7B-Instruct",
        params_b=7.0, disk_4bit_gb=4.5, ram_min_gb=12.0,
        intelligence=4, speed=2,
        note="serious model; needs an m-pro or better to fine-tune comfortably.",
    ),
    CatalogModel(
        id="qwen-math-1_5b",
        display_name="Qwen 2.5 Math 1.5B",
        hf_id_mlx="mlx-community/Qwen2.5-Math-1.5B-Instruct-4bit",
        hf_id_raw="Qwen/Qwen2.5-Math-1.5B-Instruct",
        params_b=1.5, disk_4bit_gb=1.0, ram_min_gb=4.0,
        intelligence=3, speed=4,
        specialty="math",
        note="specialty: math reasoning. great pair with gsm8k / math benches.",
    ),
    CatalogModel(
        id="qwen-coder-1_5b",
        display_name="Qwen 2.5 Coder 1.5B",
        hf_id_mlx="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        hf_id_raw="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        params_b=1.5, disk_4bit_gb=1.0, ram_min_gb=4.0,
        intelligence=3, speed=4,
        specialty="code",
        note="specialty: code. pair with humaneval / mbpp.",
    ),
    CatalogModel(
        id="hermes-3-3b",
        display_name="Hermes 3 Llama 3.2 3B",
        hf_id_mlx="mlx-community/Hermes-3-Llama-3.2-3B-4bit",
        hf_id_raw="NousResearch/Hermes-3-Llama-3.2-3B",
        params_b=3.0, disk_4bit_gb=2.0, ram_min_gb=6.0,
        intelligence=3, speed=3,
        specialty="function-calling",
        note="already function-call tuned; a strong baseline to beat.",
    ),
]


# Coarse tok/s baselines for 4-bit inference, indexed by chip class+tier.
# Multiply by the per-model fudge factor (smaller = faster).
_BASE_TOKPS: dict[tuple[str, str], float] = {
    ("apple-silicon", "base"):  40.0,
    ("apple-silicon", "pro"):   80.0,
    ("apple-silicon", "max"):  120.0,
    ("apple-silicon", "ultra"):170.0,
    ("nvidia", "none"):        100.0,   # any consumer GPU; rough
    ("intel", "none"):          12.0,   # cpu-only
    ("amd", "none"):            10.0,
    ("other", "none"):           8.0,
}


def estimate_tokps(model: CatalogModel, probe: SystemProbe) -> float:
    """Return a back-of-envelope tokens/sec figure for inference of this 4-bit
    model on this hardware. Used purely to drive the speed bar."""
    key = (probe.chip_class, probe.chip_tier if probe.chip_class == "apple-silicon" else "none")
    base = _BASE_TOKPS.get(key, _BASE_TOKPS[("other", "none")])
    # Empirically tok/s scales close to 1/params for memory-bound 4-bit inference.
    factor = 1.5 / max(0.5, model.params_b)
    return round(base * factor, 1)


def fits(model: CatalogModel, probe: SystemProbe, headroom_gb: float = 2.0) -> bool:
    """A model 'fits' if RAM and disk are both adequate with a headroom margin."""
    if probe.ram_gb < model.ram_min_gb + headroom_gb:
        return False
    if probe.free_disk_gb < model.disk_4bit_gb + 1.0:
        return False
    return True


@dataclass
class Recommendation:
    model: CatalogModel
    tokps_estimate: float
    fits: bool
    elmo_choice: bool
    reason: str = ""

    def asdict(self) -> dict:
        return {
            "model": self.model.asdict(),
            "tokps_estimate": self.tokps_estimate,
            "fits": self.fits,
            "elmo_choice": self.elmo_choice,
            "reason": self.reason,
        }


def recommend(probe: SystemProbe, specialty: Specialty | None = None) -> list[Recommendation]:
    """Rank the catalog for this system. The 'elmo's choice' label goes to the
    fitting model with the best (intelligence * 0.6 + speed * 0.4) score.

    When `specialty` is None, only general-purpose models are included —
    specialty models (math, code, function-calling) are opt-in to avoid them
    winning 'elmo's choice' on intelligence inflated by domain weighting.
    """
    if specialty is None:
        pool = [m for m in CATALOG if m.specialty == "general"]
    else:
        pool = [m for m in CATALOG if m.specialty == specialty or m.specialty == "general"]
    out: list[Recommendation] = []
    best_idx = -1
    best_score = -1.0
    for i, m in enumerate(pool):
        f = fits(m, probe)
        if f:
            score = m.intelligence * 0.6 + m.speed * 0.4
            if score > best_score:
                best_score, best_idx = score, i
        out.append(Recommendation(
            model=m,
            tokps_estimate=estimate_tokps(m, probe),
            fits=f,
            elmo_choice=False,
            reason="" if f else f"needs ~{m.ram_min_gb:.0f}GB RAM, you have {probe.ram_gb:.0f}GB",
        ))
    if best_idx >= 0:
        out[best_idx].elmo_choice = True
        out[best_idx].reason = "best balance of intelligence and speed for your machine"
    # Sort: elmo's choice first, then fitting models by score desc, non-fitting last
    out.sort(key=lambda r: (
        not r.elmo_choice,
        not r.fits,
        -(r.model.intelligence * 0.6 + r.model.speed * 0.4),
    ))
    return out
