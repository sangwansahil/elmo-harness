# contributing

elmo is in early alpha (Phase 0 — the rails). The roadmap lives in [PLAN.md](PLAN.md). If you want to help, the highest-leverage spots right now:

- **Phase 1 — data foundry**: planner → generator → judge split, verifier-first filter, hard-negative mining.
- **Phase 2 — closed loop**: failure clustering, the monotonic regression suite, per-capability gating.
- **Phase 4 — GRPO**: TRL trainer for verifier-backed capabilities (and an MLX rejection-sampling fallback).
- **Web UI** per [DESIGN.md](DESIGN.md).

## dev setup

```sh
git clone https://github.com/sangwansahil/elmo-harness && cd elmo-harness
pip install -e ".[mlx,dev]"     # or .[cuda,dev]
pytest -q
ruff check .
```

## design rules

- prefer editing existing files to creating new ones
- no comments unless the *why* is non-obvious
- verifier-first: if a check exists, use it instead of an llm judge
- every automated decision should leave a receipt — the user must be able to ask "why" and get a real answer
- lowercase ui copy. one accent color (amber). no emoji.

## license

Apache-2.0 — by submitting a PR you agree your contribution is licensed under the same.
