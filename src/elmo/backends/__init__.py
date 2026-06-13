from .base import TrainBackend, TrainResult
from .mlx import MLXBackend

__all__ = ["TrainBackend", "TrainResult", "MLXBackend", "get_backend"]


def get_backend(name: str, objective: str = "sft") -> TrainBackend:
    """Return a backend instance.

    `name`: 'auto' | 'mlx' | 'unsloth' | 'none' picks hardware.
    `objective`: 'sft' | 'rft' | 'grpo' picks the learning signal:
        sft  — supervised fine-tune (default, all backends)
        rft  — rejection-sampling fine-tune (mlx); rl-flavored sft
        grpo — verifiable-reward grpo (cuda only, via trl)
    """
    if name == "auto":
        from elmo.config import detect_backend
        name = detect_backend()
    if name == "none":
        raise RuntimeError(
            "no local training backend available on this machine. "
            "elmo needs apple silicon (mlx) or nvidia (unsloth)."
        )
    if objective == "grpo":
        if name not in ("unsloth",):
            raise RuntimeError(
                "grpo requires cuda. on apple silicon use objective='rft' instead."
            )
        from .grpo import GRPOBackend
        return GRPOBackend()
    if objective == "rft":
        if name != "mlx":
            raise RuntimeError("rft is implemented for the mlx backend; use grpo on cuda.")
        from .rft import RFTBackend
        return RFTBackend()
    # default: sft
    if name == "mlx":
        return MLXBackend()
    if name == "unsloth":
        raise NotImplementedError(
            "unsloth sft backend not wired yet — write one or use cuda+grpo via "
            "elmo.backends.grpo.GRPOBackend"
        )
    raise ValueError(f"unknown backend: {name}")
