from pathlib import Path

from .gsm8k import load_gsm8k
from .structured import load_structured_seed
from .xlam import load_xlam_function_calling

__all__ = [
    "load_xlam_function_calling",
    "load_gsm8k",
    "load_structured_seed",
    "load_dataset",
    "write_sft_jsonl",
    "write_eval_jsonl",
]


def load_dataset(source: str, max_rows: int | None, split: str, cache_dir: Path | None) -> list[dict]:
    """Dispatch to the right loader by source string."""
    if source.startswith("hf:Salesforce/xlam"):
        return load_xlam_function_calling(max_rows=max_rows, split=split, cache_dir=cache_dir)
    if source.startswith("hf:openai/gsm8k") or source == "gsm8k":
        return load_gsm8k(max_rows=max_rows, split=split, cache_dir=cache_dir)
    if source == "synthetic:structured":
        return load_structured_seed(max_rows=max_rows, split=split, cache_dir=cache_dir)
    raise NotImplementedError(f"no loader registered for source: {source}")


def write_sft_jsonl(rows: list[dict], path: Path) -> int:
    """Write the SFT-ready jsonl regardless of source — they all share the
    {"messages": [...]} shape per row."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
            n += 1
    return n


def write_eval_jsonl(rows: list[dict], path: Path) -> int:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r["_eval"]) + "\n")
            n += 1
    return n
