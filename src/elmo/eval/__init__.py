from pathlib import Path

from .function_calling import FunctionCallEvaluator, ScoreReport
from .math import MathEvaluator
from .structured import StructuredEvaluator

__all__ = [
    "FunctionCallEvaluator",
    "MathEvaluator",
    "StructuredEvaluator",
    "ScoreReport",
    "make_evaluator",
]


def make_evaluator(benchmark: str, eval_jsonl: Path):
    if benchmark in ("bfcl-simple", "bfcl", "function-calling"):
        return FunctionCallEvaluator(eval_jsonl)
    if benchmark in ("gsm8k", "math"):
        return MathEvaluator(eval_jsonl)
    if benchmark in ("json-format", "json-mode", "structured"):
        return StructuredEvaluator(eval_jsonl)
    raise ValueError(
        f"unknown benchmark '{benchmark}'. supported: bfcl-simple, gsm8k, json-format"
    )
