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

> status: beta. all eight phases of the [plan](PLAN.md) are landed. three example tasks ship in [SHOWCASE.md](SHOWCASE.md).

## fire it up

one command. handles uv, python 3.12, the venv, the right hardware extra
(mlx on apple silicon, cuda on nvidia), the daemon, and opens your browser
at the onboarding wizard.

```sh
git clone https://github.com/sangwansahil/elmo-harness && cd elmo-harness
./up
```

then to stop:

```sh
./down
```

`make up` / `make down` / `make restart` / `make logs` / `make status` /
`make clean` work too. the daemon writes its pid to `.elmo/daemon.pid` and
its log to `.elmo/daemon.log`, so it stays out of your terminal.

## what you do next

the wizard at http://127.0.0.1:7777/#/onboard walks you through it:

1. diagnose your hardware (chip, ram, gpu, backend)
2. pick a base model — "elmo's choice" is amber-outlined
3. watch the download progress bar
4. send one prompt to confirm the model loads
5. describe in one sentence what the model should be expert at
6. review the discovered capabilities + acceptance gates elmo inferred
7. watch live training counters (accepted / rejected / step / score),
   then **save to elmo model hub**

no terminal required after `./up`.

## cli (after `./up`)

once the venv is set up, the elmo binary is in `.venv/bin/`. activate or
prefix:

```sh
source .venv/bin/activate

elmo run examples/function-calling.yaml     # headless anchor task
elmo run examples/math.yaml                 # gsm8k
elmo run examples/json-structured.yaml      # structured extraction
elmo runs                                   # list past runs
elmo hub list                               # base + fine-tuned models
elmo regression list                        # permanent test cases
elmo trajectory list                        # local public-prior corpus
elmo preset apply free-openrouter           # one-line free-tier wiring
elmo doctor                                 # what is wired on this machine
elmo system                                 # hardware probe
```

## install standalone (no clone)

if you just want the package without the dev scripts:

```sh
# apple silicon
pip install "elmo-harness[mlx,ui] @ git+https://github.com/sangwansahil/elmo-harness"

# nvidia
pip install "elmo-harness[cuda,ui] @ git+https://github.com/sangwansahil/elmo-harness"
```

then `elmo onboard` directly.

a single run prints a before/after number, writes artifacts to `./runs/<id>/`, and logs the run to a local sqlite database (`./.elmo/elmo.db`).

## what works today

**phase 0 — rails**

- task spec parser (yaml + pydantic contract)
- mlx-lm lora fine-tune driver for apple silicon
- xlam function-calling dataset loader
- minimal bfcl-style verifier (function name + json args)
- sqlite run state, jsonl artifact log

**phase 2 — closed loop**

- multi-iteration `execute` loop: foundry (with diagnose-informed brief) →
  train → eval → regression-suite eval → per-capability gate → promote
  failures → diagnose
- regression suite (`runs/<task>.regression.jsonl`) is monotonic, idempotent
  on (capability, query), tracks `first_seen_iter` and `fixed_in_iter`, and
  ships alongside the model as a proof of what cannot regress
- gate compares the new capability vector against the best-so-far, not
  aggregate; blocked if any gated capability regresses by more than epsilon
- diagnose clusters failures by capability (deterministic) then asks the
  planner for a one-paragraph summary + corrective brief per cluster, fed
  to the next iteration's foundry call
- early-stop on target hit, plateau, or budget exhaustion

**phase 3 — web ui**

- `elmo serve` starts a FastAPI daemon at `http://127.0.0.1:7777`
- rest endpoints: `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/events`,
  `/api/regression/{task}`, `/api/health`
- websocket `/api/runs/{id}/live` streams stage events while a run is active
- single-page vanilla-js ui (no build step), "instrument paper" design
  language, dark "phosphor" + light "paper" via prefers-color-scheme

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
