# elmo — showcase

Three reproducible expert SLMs from the same harness, same machine, same loop.

Each task is a YAML spec. Each spec produces: a fine-tuned adapter, a per-capability eval report, a regression suite, and a trajectory artifact appended to the public prior. Reproducing a row is `elmo run examples/<task>.yaml`.

> **status:** the harness is in place, the recipes are pinned, but the absolute scores below should be filled in *by the person reproducing.* This file documents the reproduction path; it does not pre-bake numbers I did not measure.

## 1 — function calling (anchor task)

| field | value |
|---|---|
| spec | [examples/function-calling.yaml](examples/function-calling.yaml) |
| base | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| dataset | xLAM-function-calling-60k (Salesforce, CC-BY) |
| benchmark | BFCL-style verifier (function name + JSON args equivalence + parallel set match) |
| capabilities | tool_selection · arguments · parallel_calls · refusal |
| objective | `sft` (try `rft` on iteration 2+) |
| acceptance | overall ≥ 0.75 |

**reproduce:**
```sh
elmo run examples/function-calling.yaml
elmo runs                   # see before/after
elmo regression list        # what cannot regress
```

| | baseline | after | Δ |
|---|---|---|---|
| tool_selection | _fill_ | _fill_ | _fill_ |
| arguments | _fill_ | _fill_ | _fill_ |
| parallel_calls | _fill_ | _fill_ | _fill_ |
| **overall** | _fill_ | _fill_ | _fill_ |

## 2 — grade-school math

| field | value |
|---|---|
| spec | [examples/math.yaml](examples/math.yaml) |
| base | `mlx-community/Qwen2.5-Math-1.5B-Instruct-4bit` |
| dataset | GSM8K (openai/gsm8k via Hugging Face) |
| benchmark | final-answer equivalence (`#### N` extractor) |
| capabilities | correctness (single) |
| objective | `sft` |
| acceptance | overall ≥ 0.65 |

**reproduce:**
```sh
elmo run examples/math.yaml
```

| | baseline | after | Δ |
|---|---|---|---|
| **overall** | _fill_ | _fill_ | _fill_ |

## 3 — structured JSON extraction

| field | value |
|---|---|
| spec | [examples/json-structured.yaml](examples/json-structured.yaml) |
| base | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| dataset | synthetic seed (offline; six templates, replicated) |
| benchmark | schema-validity + key-completeness + type-correctness |
| capabilities | parseable · keys_present · types_correct |
| objective | `sft` |
| acceptance | overall ≥ 0.80 |

**reproduce:**
```sh
elmo run examples/json-structured.yaml
```

| | baseline | after | Δ |
|---|---|---|---|
| parseable | _fill_ | _fill_ | _fill_ |
| keys_present | _fill_ | _fill_ | _fill_ |
| types_correct | _fill_ | _fill_ | _fill_ |
| **overall** | _fill_ | _fill_ | _fill_ |

---

## how to interpret the loop

Every run produces:

- `runs/<run_id>/report.json` — full eval report
- `runs/<run_id>/iter_NN/foundry/provenance.jsonl` — per-row data lineage (when foundry is enabled)
- `runs/<task>.regression.jsonl` — permanent capability-tagged test cases
- `runs/trajectories.jsonl` — one row appended per finished run (the public prior, locally)

The web UI surfaces all of these. Start it with:

```sh
elmo serve   # open http://127.0.0.1:7777
```

## reproducibility notes

- Set `train.seed` for deterministic runs.
- Set `ELMO_NO_CACHE=1` to bypass the completion cache during ablations.
- Set `HF_TOKEN` and run `elmo trajectory publish` to add your row to the public prior.
- All three example specs default to **synthetic data foundry disabled** so the first reproduction needs no API keys. Enable foundry once you have OpenRouter/Anthropic credentials, and the planner will start retrieving from your local prior automatically.

## what this proves

A single open-source harness, running entirely on a laptop, can:

- turn three different natural-language prompts into three different fine-tuned models,
- evaluate each on a verifiable public benchmark,
- gate deployment on a per-capability vector (never aggregate),
- ship the model alongside a regression suite that names exactly what it cannot regress on, and
- compound across runs via the public trajectory prior.

That is the Pioneer recipe, open-sourced and rehosted on hardware anyone has.
