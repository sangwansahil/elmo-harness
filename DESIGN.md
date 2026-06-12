# elmo design system — "instrument paper"

Retro-modern and subtle. A lab notebook crossed with a flight instrument: warm paper surfaces, ink text, hairline rules, mono numerals, a single amber accent. Calm by default — color appears only when meaning changes.

## Principles

1. **Receipts, not logs.** Every agent decision is a bordered card: what it did, why, what it cost. The "why" line is mandatory.
2. **One accent.** Amber means *live/running*. Green appears only for *pass*, brick red only for *regression*. Nothing else is colored, ever.
3. **Mono numerals everywhere.** All data, scores, costs, and identifiers are set in mono; prose is sans.
4. **Hairlines and dotted leaders.** 1px borders; dotted rules connect label → value, receipt-style.
5. **Quiet motion.** 150ms ease on everything; the only looping animation is the cursor blink on the live stage.
6. **prompt → model is the spine.** The run screen reads top-to-bottom as one continuous story: prompt, contract, stages, result.

## Tokens

| Token | light "paper" | dark "phosphor" |
|---|---|---|
| --bg | #F2EEE5 | #14110D |
| --surface | #FAF7F1 | #1C1814 |
| --ink | #1B1916 | #E9E2D4 |
| --ink-muted | #6C665D | #968D7E |
| --hairline | #DAD3C4 | #2E2922 |
| --accent (live) | #B45309 | #E0A458 |
| --pass | #4E7A51 | #8AB48D |
| --fail (regression only) | #AE4A32 | #D06A50 |

- **Type:** UI/prose = Inter (or Geist); data/code/numerals = JetBrains Mono. Micro-labels: 11px mono, +0.08em tracking, lowercase.
- **Radius:** 6px cards, 2px chips; progress bars squared.
- **Grid:** 8px base; page max-width 1040px; dot-grid background at 24px, ~4% opacity.
- **Headings:** lowercase, plain language.

## Components

- **Stage rail** — the seven stages (spec · plan · data · train · eval · diagnose · export) with glyph states: ◇ pending, ◈ running (amber, blinking caret), ◆ done. Dotted leaders between stages.
- **Receipt card** — header (stage · timestamp · cost), body, and a muted "why:" footer with the agent's rationale and the models/tokens used.
- **Progress bar** — 3px, squared ends, mono percentage right-aligned.
- **Capability vector** — one labeled bar per capability with ▲/▼ delta chips against the previous iteration.
- **Benchmark track** — a single line: baseline tick → current fill (amber) → target tick, with mono labels.
- **Iteration strip** — compact history chips (`01 +9.3`, `02 +4.6`, `03 ◈`), future iterations ghosted.
- **Log feed** — timestamped mono lines, each tagged with its stage.
- **Provenance drawer** — any dataset row expands to: generator model, prompt hash, verifier verdict, judge score (if any).

## Screens

1. **new run** — prompt composer; live-parsed contract preview (capabilities, verifiers, budget); base-model picker with a plain-language note on what this machine can train.
2. **run** — the core screen. Top: prompt as a receipt. Middle: stage rail + current-stage detail + capability vector + benchmark track. Bottom: iteration strip + log feed.
3. **trajectories** — tree of attempted pipelines, scored and color-coded by outcome.
4. **regression** — capability × iteration grid of the permanent test suite; every cell is a pass/fail dot.
5. **export** — artifacts (GGUF / MLX / LoRA), eval report, regression suite download, public-prior publish toggle.

## Voice

Lowercase headings. Plain language. Honest numbers. Error states say what the agent will try next, not just what broke. The mascot is allowed exactly one appearance per screen.
