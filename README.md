```
        _
   ___ | | _ __ ___    ___
  / _ \| || '_ ` _ \  / _ \
 |  __/| || | | | | || (_) |
  \___||_||_| |_| |_| \___/

  prompt -> expert slm.
```

elmo is an open-source harness that turns a natural-language prompt — *"build me a function-calling expert under 2B params"* — into a fine-tuned small language model. It plans the training pipeline, finds or generates data, trains locally, evaluates against a public benchmark, diagnoses failures, and iterates until the model hits the target or the budget runs out.

Inspired by [Pioneer](https://pioneer.ai/blog/behind-pioneer) and built to run on anyone's laptop.

> status: alpha. phases 0–1 (rails + data foundry). see [PLAN.md](PLAN.md) for the roadmap.

## install

```sh
# apple silicon (mlx)
pip install "elmo-harness[mlx] @ git+https://github.com/sangwansahil/elmo-harness"

# nvidia (unsloth + trl)
pip install "elmo-harness[cuda] @ git+https://github.com/sangwansahil/elmo-harness"

# from source
git clone https://github.com/sangwansahil/elmo-harness && cd elmo-harness
pip install -e ".[mlx]"   # or .[cuda]
```

## run

```sh
elmo init
elmo run examples/function-calling.yaml
elmo runs                 # list past runs
elmo eval --run <run_id>  # re-evaluate
```

a single run prints a before/after number, writes artifacts to `./runs/<id>/`, and logs the run to a local sqlite database (`./.elmo/elmo.db`).

## what works today

**phase 0 — rails**

- task spec parser (yaml + pydantic contract)
- mlx-lm lora fine-tune driver for apple silicon
- xlam function-calling dataset loader
- minimal bfcl-style verifier (function name + json args)
- sqlite run state, jsonl artifact log

**phase 1 — data foundry**

- provider layer: openai-compatible (openai, openrouter, groq, deepseek,
  together, cerebras, lmstudio, ollama) + anthropic native; stdlib http, no deps
- planner / generator / judge roles, configured via spec, env vars, or defaults
- verifier-first synthetic data: strong planner emits scenarios, cheap generator
  produces rows, deterministic filter rejects malformed ones
- per-row provenance log (planner model, generator model, token counts,
  verifier verdict, seed-prompt hash)
- `elmo plan <spec>` dry-runs the planner; `elmo providers` and `elmo doctor`
  show what is configured

## what is coming (see [PLAN.md](PLAN.md))

- closed-loop diagnose + monotonic regression suite
- grpo training on verifiable rewards
- web ui per [DESIGN.md](DESIGN.md)
- public trajectory prior on hugging face

## the three bets

1. **verifier-first rewards** — where a check exists (function name, json schema, exec, math), use it as a free non-gameable reward; the llm judge is a fallback.
2. **the regression suite is a shipped artifact** — every failure becomes a permanent capability-tagged test case; deployment gates on a per-capability vector, not an aggregate.
3. **an open prior over pipeline trajectories** — every run can publish `(spec, model, recipe, eval_deltas)` to a public hf dataset; the planner retrieves from it. closed companies cannot compound across customers.

## license

apache-2.0. see [LICENSE](LICENSE).
