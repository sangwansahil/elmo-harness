from .base import TrainBackend, TrainResult
from .mlx import MLXBackend

__all__ = ["TrainBackend", "TrainResult", "MLXBackend", "get_backend"]


def get_backend(name: str) -> TrainBackend:
    """Return a backend instance by name. 'auto' picks one based on hardware."""
    if name == "auto":
        from elmo.config import detect_backend
        name = detect_backend()
    if name == "mlx":
        return MLXBackend()
    if name == "unsloth":
        raise NotImplementedError(
            "unsloth backend not implemented yet — phase 4. install with pip install elmo-harness[cuda]"
        )
    if name == "none":
        raise RuntimeError(
            "no local training backend available on this machine. "
            "elmo needs apple silicon (mlx) or nvidia (unsloth)."
        )
    raise ValueError(f"unknown backend: {name}")
