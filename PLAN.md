# elmo — prompt → model

> Type a prompt. Get an expert small language model, with receipts.

**Status:** planning · 2026-06-11
**License:** Apache-2.0 · **Primary dev machine:** Apple M5 Pro (MLX path)

---

## 1. What elmo is

elmo is an open-source harness that turns a natural-language prompt — *"build me a function-calling expert under 2B params"* — into a fine-tuned small language model. It plans the training pipeline, finds or generates data, trains locally, evaluates against a public benchmark, diagnoses failures, and iterates until the model hits the target or the budget runs out.

Inspired by Pioneer ([blog](https://pioneer.ai/blog/behind-pioneer), [arXiv:2604.09791](https://arxiv.org/pdf/2604.09791)), built to run on anyone's laptop, and designed so every automated decision is explainable.

**The product is one motion: prompt → model.** Everything else is a receipt the user can inspect.

## 2. The loop

```
 prompt
   │
   ▼
 ┌──────┐   ┌──────┐   ┌──────┐   ┌───────┐   ┌──────┐   ┌──────────┐   ┌────────┐
 │ spec │ → │ plan │ → │ data │ → │ train │ → │ eval │ → │ diagnose │ → │ export │
 └──────┘   └──────┘   └──────┘   └───────┘   └──────┘   └────┬─────┘   └────────┘
                ▲                                             │
                └────────────── next iteration ───────────────┘
```

1. **spec** — parse the prompt into a typed task contract: capability list, verifier bindings, base-model constraints, target metric, budget (tokens + dollars + wall-clock).
2. **plan** — a strong model (frontier via BYOK, free-tier, or local) emits a declarative `TrainPlan`: data recipe, base model, training strategy, hyperparameters. Before planning, it retrieves nearest-neighbor trajectories from the public prior (see bet 3).
3. **data** — search HF Hub for existing datasets first; a cheap generator model produces synthetic rows where coverage is missing; a **verifier-first filter** (schema/exec/exact-match) accepts or rejects each row, LLM-judge only where no verifier exists; hard negatives mined near the decision boundary. Every row gets a provenance record (generator, prompt hash, verifier verdict).
4. **train** — LoRA/QLoRA SFT via MLX (Apple Silicon) or Unsloth (CUDA); GRPO via TRL for verifier-backed capabilities (CUDA); rejection-sampling fine-tuning (RFT) as the Apple-Silicon RL fallback.
5. **eval** — public benchmark (BFCL for the anchor task) + the regression suite. Output is a **per-capability score vector**, never just an aggregate.
6. **diagnose** — cluster failures by embedding + capability tag; strong model writes a one-paragraph failure summary per cluster; every failure is promoted into the permanent regression suite; emits the next data brief.
7. **gate** — export/deploy only if no capability regressed. The gate is a vector comparison, not an average.
8. **export** — GGUF + MLX + LoRA adapter + eval report + regression suite + (opt-in) anonymized trajectory published to the public prior.

Loop terminates on: target hit, budget exhausted, or delta-per-dollar below threshold.

## 3. Three bets beyond Pioneer

1. **Verifiable rewards first.** Pioneer's loop is mostly SFT on judge-filtered synthetic data — gameable and perpetually teacher-dependent. Where a checker exists (function-name match, JSON-schema validation, AST equivalence, execution), elmo uses it as a free, non-gameable reward for GRPO/RFT and as the data filter. The judge is a fallback, not the default.
2. **The regression suite is a shipped artifact.** Every failure ever found becomes a permanent, capability-tagged test case. Deployment gates on per-capability monotonicity. Users get a model *plus* a proof of what it cannot regress on. This is also the catastrophic-forgetting alarm.
3. **An open prior over pipeline trajectories.** Every run can publish an anonymized `(task_spec, base_model, data_recipe, train_config, eval_deltas)` record to a public HF dataset. The planner retrieves from it before planning. Closed companies can't compound across customers; an open harness can — this is the structural moat.

## 4. Anchor task: function calling

| Capability | Verifier |
|---|---|
| C1 tool selection | exact match on function name |
| C2 argument extraction | JSON-schema validation + AST equivalence |
| C3 parallel calls | set equality on (name, normalized args) |
| C4 refusal / clarification on ambiguous input | label match (+ judge for tone) |
| C5 multi-turn chaining | state-machine replay of expected trace |

- **Datasets:** xLAM-function-calling-60k, Glaive-function-calling-v2, APIGen-MT, ToolACE.
- **Benchmark gate:** BFCL (latest version; report API-Bank as the second opinion).
- **Base models:** Qwen2.5-1.5B-Instruct first; Llama-3.2-3B and Phi-3.5-mini as follow-ups.
- **Headline target:** +15 BFCL overall on Qwen2.5-1.5B within 5 loop iterations, entirely on one M-series laptop (plus free/BYOK API for planner & generator).

## 5. Architecture

```
 elmod  — Python daemon (FastAPI + WebSocket), SQLite run state, pluggable backends
   ├─ backends: mlx (Apple Silicon) · unsloth/trl (CUDA) · api-only (no local training)
   ├─ providers: OpenAI-compatible client layer, config = "provider/model_id" (opencode-style)
   └─ roles: planner (best available) · generator (cheapest adequate) · judge (only when no verifier)
 web    — Vite + React, served by the daemon at localhost:7777
 cli    — elmo init · run · watch · export · publish
```

**Model-access tiers** (the honest version of opencode's Zen):

| Tier | What | Privacy |
|---|---|---|
| local | MLX / llama.cpp / LM Studio / Ollama | everything stays on-device |
| free | OpenRouter `:free`, Groq, Cerebras free tiers | retention tradeoffs documented per provider, up front |
| byok | Anthropic, OpenAI, Google, DeepSeek, Mistral, … | per provider policy |

Hardware is detected at first run; the UI states plainly what this machine can do (train locally vs orchestrate-only).

## 6. UI

See [DESIGN.md](DESIGN.md). One screen carries the product: the **run screen** — prompt at top, the seven stages as instrument rows, live progress per stage, and a "why" receipt under every automated decision. Max explainability rules:

- Every agent decision renders as a receipt: which model, tokens spent, cost, rationale.
- Every dataset row is traceable to its generator prompt and verifier verdict.
- Every score change is attributed to a diff between iterations.

## 7. Build phases

| Phase | Status | Scope | Acceptance |
|---|---|---|---|
| 0 — rails | ✅ done | MLX LoRA SFT of Qwen2.5-1.5B on an xLAM slice; BFCL-simple before/after; SQLite run log | one command, one before/after number, reproducible |
| 1 — foundry | ✅ done | planner→generator split, verifier-first filter, hard negatives, provenance log | synthetic data measurably beats raw-dataset baseline |
| 2 — loop | ✅ done | diagnose, regression suite, vector gate, unattended n-iteration runs | overnight run improves BFCL with zero capability regressions |
| 3 — UI | ✅ done | daemon API + live web run screen per DESIGN.md | a stranger can read a run without docs |
| 4 — GRPO | ✅ done | TRL GRPO on CUDA; RFT fallback on MLX | RL beats SFT-only on ≥1 capability |
| 5 — gateway | ✅ done | provider tiers, BYOK config, free-tier wiring, caching | full run completes on free tier only |
| 6 — prior + site | ✅ done | trajectory publish/retrieve; landing page, `curl \| sh` install | planner provably uses retrieved trajectories |
| 7 — showcase | ✅ done | write-up; release 3 expert models (function calling + 2 more tasks) | published numbers, reproducible from the repo |

## 8. Locked defaults

- **Daemon language:** Python (the entire training/eval ecosystem is Python; no FFI tax). Web UI in TypeScript.
- **Name:** elmo.
- **First base model:** Qwen2.5-1.5B-Instruct (fastest iteration loop).
- **License:** Apache-2.0.
- **Flagship gate:** BFCL.

## 9. Risks

- **MLX RL immaturity** → RFT (rejection-sampling + SFT) captures much of the verifiable-reward gain using SFT machinery only.
- **BFCL contamination** → hold out eval splits rigorously; always report API-Bank as a second benchmark.
- **Free-tier rate limits** → aggressive caching, resumable jobs; data-gen is embarrassingly parallel and tolerant of slow providers.
- **Judge gaming / distribution drift** → verifier-first design exists precisely for this; judge usage is logged and capped.

## 10. References

- Pioneer: https://pioneer.ai/blog/behind-pioneer · https://arxiv.org/pdf/2604.09791
- opencode (free-tier + config UX reference): https://opencode.ai/docs/ · https://opencode.ai/docs/zen/
- Training: MLX-LM, Unsloth, TRL (GRPO), Axolotl
- Eval: lm-evaluation-harness, BFCL, API-Bank
- Data: xLAM, Glaive-v2, APIGen-MT, ToolACE, Distilabel
