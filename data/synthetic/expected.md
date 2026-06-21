# data/synthetic/diagnostic.txt — what the crystallization loop should find

This is the **corpus-level analog of the Phase E symbolic-synthesis test**: a
hand-built corpus with *planted* structure of known kinds, interleaved with
known-structureless noise. A healthy crystallization loop should compress the
structured sections into low-MDL modules and find nothing in the noise. It is
the diagnostic that makes Bets 2 and 3 (`ARCHITECTURE.md` Part V) measurable
before committing to a multi-day real-corpus run.

Each section below states the planted pattern, what the loop should do, and
which mechanism it exercises.

## Section 1 — Periodic phrase (period-1 n-gram)

The same sentence repeated verbatim. The single highest-structure region.

- **Expected:** the loop notices near-zero prediction error after the first
  occurrence; a single crystallized module reproduces the phrase at minimal
  MDL. `J/nat` over this span should fall sharply versus the noise span.
- **Exercises:** the basic notice → buffer → distill → crystallize path, and
  Bet 2 (crystallization produces net predictive gain).

## Section 2 — Shift rule

Each line is the previous line shifted by one position in the alphabet — the
planted "shift" relation.

- **Expected:** symbolic synthesis (`src/crystal/synth.rs`) recovers a **`Shift`**
  program; the cluster's ternary matrix is reducible to a short program and
  attaches a `symbolic_hint`. This is the corpus-level version of the Phase E
  planted-pattern recovery unit test.
- **Exercises:** the symbolic DSL (Shift primitive), Head S suitability
  (Phase P), and eventual symbolic-at-inference execution (Phase Q).

## Section 3 — Period-2 alternation

Three independent period-2 sequences (`up down`, `plus minus`, `on off`).

- **Expected:** a period-2 / `Negate`-like or `Mask` structure is detected;
  predictions lock onto the alternation after one period. Low residual bits.
- **Exercises:** the Negate / Mask DSL primitives and the distill MDL math
  (`src/crystal/distill.rs`).

## Section 4 — High-entropy noise

Fixed but structureless token soup. The negative control.

- **Expected:** the loop finds **no** compressible structure here — clustering
  yields high residual, no module passes the compression gate, and no
  `symbolic_hint` attaches. If a module *does* crystallize from this section,
  the MDL gate is too loose (a false-positive regression to investigate).
- **Exercises:** the forget/no-regression discipline (Bet 3) and the
  compression-gate threshold.

## Section 5 — Structure interleaved with noise

The Section-1 phrase embedded between noise tokens at varying offsets.

- **Expected:** the loop isolates the recurring phrase from the surrounding
  noise — the crystallized module fires on the structured span and stays quiet
  on the noise, rather than memorizing whole noisy lines.
- **Exercises:** routing/activation selectivity and the active-selection
  acquisition score (`max(N − C, 0) / joules_per_token`).

## How to run it

```bash
clob synth   --seed 1 --output /tmp/diag.clob --config small
clob ingest  --seed 1 --model /tmp/diag.clob \
    --input data/synthetic/diagnostic.txt --memory-dir /tmp/diag_eps \
    --metrics-out /tmp/diag.jsonl
clob crystal --model /tmp/diag.clob --memory-dir /tmp/diag_eps --modules-dir /tmp/diag_mods
clob metrics --path /tmp/diag.jsonl
```

> On a **random-weight** synth model these signals are still flat — the value
> of this fixture is as the first target once the model has actually been
> trained (Phase L corpus + Phase N first run). It documents *what success
> looks like* so a real run can be judged against ground truth rather than
> vibes.
