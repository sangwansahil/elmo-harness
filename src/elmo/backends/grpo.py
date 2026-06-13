"""GRPO trainer — verifier-rewarded policy optimization via TRL.

CUDA-only (TRL + transformers + accelerate). On Apple Silicon use RFTBackend
instead.

Wires the existing function-calling verifier into TRL's GRPOTrainer as a
reward function. No reward model, no KL penalty by default — let the
verifier do the talking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elmo.backends.base import TrainResult
from elmo.reward import function_call_reward
from elmo.spec import TaskSpec


@dataclass
class GRPOConfig:
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    temperature: float = 0.7
    beta: float = 0.04  # KL coefficient


class GRPOBackend:
    """Thin wrapper around TRL's GRPOTrainer."""

    name = "grpo"

    def __init__(self, config: GRPOConfig | None = None):
        self.config = config or GRPOConfig()

    def _require_trl(self) -> Any:
        try:
            import trl  # noqa: F401
            import transformers  # noqa: F401
            return trl
        except ImportError as e:
            raise RuntimeError(
                "trl + transformers not installed. on cuda, run: pip install 'elmo-harness[cuda]'"
            ) from e

    def train(
        self,
        spec: TaskSpec,
        train_jsonl: Path,
        val_jsonl: Path | None,
        out_dir: Path,
    ) -> TrainResult:
        self._require_trl()
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig as TRLGRPOConfig
        from trl import GRPOTrainer

        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config

        # The eval-format jsonl carries (query, tools, expected_calls, system).
        # GRPO consumes a Dataset of prompt strings + a reward fn that closes
        # over the same row's expected_calls — TRL hands us `completions` and
        # the original column values via kwargs.
        eval_jsonl = self._find_eval_companion(train_jsonl)
        rows = [json.loads(ln) for ln in eval_jsonl.read_text().splitlines() if ln.strip()]

        tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = Dataset.from_list([
            {
                "prompt": self._build_prompt(r, tokenizer),
                "expected_calls": json.dumps(r["expected_calls"]),
            }
            for r in rows
        ])

        def reward_fn(completions, expected_calls, **kw):
            out: list[float] = []
            for completion, expected in zip(completions, expected_calls):
                exp = json.loads(expected) if isinstance(expected, str) else expected
                out.append(function_call_reward(completion, exp))
            return out

        model = AutoModelForCausalLM.from_pretrained(spec.base_model)
        peft = LoraConfig(
            r=spec.train.lora_rank,
            lora_alpha=spec.train.lora_alpha,
            lora_dropout=spec.train.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        args = TRLGRPOConfig(
            output_dir=str(out_dir),
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_train_epochs,
            num_generations=cfg.num_generations,
            max_prompt_length=cfg.max_prompt_length,
            max_completion_length=cfg.max_completion_length,
            temperature=cfg.temperature,
            beta=cfg.beta,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )
        trainer = GRPOTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            reward_funcs=[reward_fn],
            peft_config=peft,
            tokenizer=tokenizer,
        )
        result = trainer.train()
        trainer.save_model(str(out_dir / "adapter"))
        return TrainResult(
            adapter_path=out_dir / "adapter",
            train_loss=float(getattr(result, "training_loss", 0.0) or 0.0),
            val_loss=None,
            steps=int(getattr(result, "global_step", 0) or 0),
        )

    def generate(
        self,
        model_id: str,
        adapter_path: Path | None,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> list[str]:
        self._require_trl()
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if adapter_path and adapter_path.exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))
        outputs: list[str] = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature or 1e-6,
                do_sample=bool(temperature), pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            outputs.append(text)
        return outputs

    def _build_prompt(self, row: dict, tokenizer) -> str:
        # Use the tokenizer's chat template if it has one.
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["query"]},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{row['system']}\n\nUser: {row['query']}\nAssistant:"

    def _find_eval_companion(self, train_jsonl: Path) -> Path:
        for c in (
            train_jsonl.with_name(train_jsonl.stem.replace("train", "eval") + ".jsonl"),
            train_jsonl.parent / "eval.jsonl",
            train_jsonl.parent / "foundry_eval.jsonl",
        ):
            if c.exists():
                return c
        raise FileNotFoundError(
            "GRPO needs an eval-shaped jsonl (query + expected_calls) next to the "
            "training jsonl. could not find one near " + str(train_jsonl)
        )
