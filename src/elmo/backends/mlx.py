"""MLX backend — apple silicon training and inference via mlx-lm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from elmo.backends.base import TrainResult
from elmo.spec import TaskSpec


class MLXBackend:
    name = "mlx"

    def _require_mlx(self) -> Any:
        try:
            import mlx_lm  # noqa: F401
            return mlx_lm
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. on apple silicon, run: pip install 'elmo-harness[mlx]'"
            ) from e

    def train(
        self,
        spec: TaskSpec,
        train_jsonl: Path,
        val_jsonl: Path | None,
        out_dir: Path,
    ) -> TrainResult:
        self._require_mlx()
        from mlx_lm.lora import train_model  # type: ignore
        from mlx_lm import load  # type: ignore

        out_dir.mkdir(parents=True, exist_ok=True)

        # mlx-lm reads data from a directory containing train.jsonl + valid.jsonl
        data_dir = out_dir / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "train.jsonl").write_text(train_jsonl.read_text())
        if val_jsonl and val_jsonl.exists():
            (data_dir / "valid.jsonl").write_text(val_jsonl.read_text())
        else:
            # mlx-lm requires a valid.jsonl; reuse a slice of train
            lines = train_jsonl.read_text().splitlines()
            (data_dir / "valid.jsonl").write_text(
                "\n".join(lines[: max(1, len(lines) // 20)])
            )

        model, tokenizer = load(spec.base_model)

        # mlx-lm exposes a train function; signatures shift across versions, so we
        # build the args dict and let users pin versions in their env.
        tc = spec.train
        args: dict[str, Any] = {
            "model": model,
            "tokenizer": tokenizer,
            "args": {
                "data": str(data_dir),
                "adapter_path": str(out_dir / "adapter"),
                "iters": tc.max_steps or 200,
                "batch_size": tc.batch_size,
                "learning_rate": tc.learning_rate,
                "lora_layers": tc.lora_rank,
                "max_seq_length": tc.max_seq_len,
                "seed": tc.seed,
                "grad_checkpoint": True,
                "save_every": tc.max_steps or 200,
            },
        }

        # train_model signatures differ between mlx-lm releases; try the modern one,
        # fall back to legacy positional args.
        try:
            train_info = train_model(**args)
        except TypeError:
            from types import SimpleNamespace

            train_info = train_model(
                model, tokenizer, SimpleNamespace(**args["args"])
            )

        train_loss = float(getattr(train_info, "train_loss", 0.0) or 0.0)
        val_loss = getattr(train_info, "val_loss", None)
        return TrainResult(
            adapter_path=out_dir / "adapter",
            train_loss=train_loss,
            val_loss=float(val_loss) if val_loss is not None else None,
            steps=tc.max_steps or 200,
        )

    def generate(
        self,
        model_id: str,
        adapter_path: Path | None,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]:
        self._require_mlx()
        import json as _json

        from mlx_lm import generate, load  # type: ignore

        if adapter_path and adapter_path.exists():
            model, tokenizer = load(model_id, adapter_path=str(adapter_path))
        else:
            model, tokenizer = load(model_id)

        # mlx-lm 0.31 replaced the `temp=` kwarg with a sampler callable.
        # The legacy path (older releases) is kept as a fallback.
        sampler = None
        try:
            from mlx_lm.sample_utils import make_sampler  # type: ignore
            sampler = make_sampler(temp=temperature)
        except ImportError:
            pass

        outputs: list[str] = []
        for raw_prompt in prompts:
            prompt = _format_for_chat(raw_prompt, tokenizer)
            kwargs: dict[str, Any] = {"max_tokens": max_tokens, "verbose": False}
            if sampler is not None:
                kwargs["sampler"] = sampler
            else:
                kwargs["temp"] = temperature  # type: ignore[assignment]
            text = generate(model, tokenizer, prompt=prompt, **kwargs)
            outputs.append(text)
        return outputs


def _format_for_chat(raw_prompt: str, tokenizer: Any) -> str:
    """Detect a json-encoded chat envelope and apply the tokenizer's chat
    template; otherwise wrap a plain string as a single user turn.

    The evaluators emit prompts shaped like
        json.dumps({"messages": [...]})
    so the model sees real chat-template tokens. A bare user message
    (used by the wizard probe step) is also wrapped here.
    """
    import json as _json

    if not hasattr(tokenizer, "apply_chat_template"):
        return raw_prompt

    messages = None
    s = raw_prompt.strip()
    if s.startswith("{") and '"messages"' in s:
        try:
            obj = _json.loads(s)
            if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
                messages = obj["messages"]
        except _json.JSONDecodeError:
            messages = None

    if messages is None:
        messages = [{"role": "user", "content": raw_prompt}]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
