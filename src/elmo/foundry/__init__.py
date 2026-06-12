"""The foundry — verifier-first synthetic data generation.

Pipeline: TaskSpec → planner → DataBrief → generator(per scenario) → row →
verifier filter → judge (only where no verifier) → SFT-ready jsonl + provenance.
"""

from .filter import FilterReport, filter_row
from .generator import generate_row
from .planner import DataBrief, Scenario, build_brief
from .provenance import ProvenanceLog
from .pipeline import run_foundry, FoundryResult

__all__ = [
    "DataBrief",
    "Scenario",
    "FilterReport",
    "FoundryResult",
    "ProvenanceLog",
    "build_brief",
    "generate_row",
    "filter_row",
    "run_foundry",
]
