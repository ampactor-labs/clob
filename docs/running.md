# Running clob: the runbook

This document walks through a cold start — fresh clone to first
crystallized module — with expected numbers at each step. If your
numbers diverge significantly from what's here, the diagnostic
ladder at the bottom is where to look.

## Cold start

```bash
git clone <repo> clob && cd clob
cargo build --release
cargo test --release              # expect: 50+ passed
```

## The canonical pipeline

Seeds default to `1` everywhere that takes `--seed`. Every command is
reproducible: the same pipeline run twice on the same commit produces
byte-identical artifacts (see Phase G verification).

### 1. Seed model

```bash
./target/release/clob synth \
    --output seed.clob --config small --seed 1
```

Expected: `seed.clob` (~0.80 MB for `small` config, d=128 L=4 V=1024),
`seed.clob.manifest.toml` sidecar capturing seed + git commit.

### 2. Calibrate confidence + meta critics

```bash
./target/release/clob calibrate-confidence \
    --model seed.clob \
    --corpus tests/data/eval_corpus.txt \
    --output conf.bin --meta-out meta.bin \
    --seed 1 --epochs 3 --meta-refresh 300
```

Expected on the smoke corpus (1.5 KB, random weights):
- Head MSE trends ~29 → ~13 → ~6 across 3 epochs.
- Meta MSE trends ~22 → ~3 → ~0.5 (faster convergence; tracks a
  simpler signal).
- Pearson correlation with NLL ~0 (random weights carry no signal —
  **this flips to positive on a real model**).
- Sidecar manifest emitted next to both `.bin` files.

### 3. Train routers

```bash
./target/release/clob train-router \
    --model seed.clob \
    --corpus tests/data/eval_corpus.txt \
    --output routers.bin --seed 1 --cf-rate 0.15 --max-tokens 1500
```

Expected:
- 1 MoE layer detected (layer 2 under `small` config).
- Counterfactuals ≈ 15% of tokens (matches `--cf-rate`).
- L2 drift ≈ 15–30% of the original weight norm.
- `routers.bin` serializes the trained router weights + manifest.

### 4. Ingest with adaptive compute

```bash
./target/release/clob ingest \
    --model seed.clob --input tests/data/eval_corpus.txt \
    --memory-dir episodes \
    --confidence-head conf.bin --meta-critic meta.bin --routers routers.bin \
    --adaptive-compute --adaptive-max-extra 2 --adaptive-z 0.5 \
    --metrics-out m.jsonl --metrics-window 500
```

Expected:
- mean NLL ≈ 7.0 (near log(vocab)=~6.9 on random weights — no real
  learning yet).
- Some `extra_steps` (100s) and `meta_suppressed` (tens) events
  per window.
- ~300-1500 tok/s depending on adaptive settings.

### 5. Read the dashboard

```bash
./target/release/clob metrics --path m.jsonl
```

Expected: a table with one row per window, baseline from window 0,
ΔJ/nat%, Δmean_nll. On the smoke corpus with random weights, J/nat
is ~0.0003–0.007 depending on adaptive settings.

### 6. Drain episodes into modules

```bash
./target/release/clob crystal \
    --memory-dir episodes --modules-dir modules --model seed.clob
```

Expected: may produce zero modules on the smoke corpus (clusters
aren't coherent at d=128 with ~280 episodes). On a real corpus,
expect a few modules per 10K tokens with `symbolic_hint` attached
to the tiny minority that are reducible to the DSL.

## Diagnostic ladder

### Synth produces a different sha256 on two runs

You're building from a tarball without git — `CLOB_GIT_COMMIT` is
"unknown" but seeds are still deterministic. The `.manifest.toml`
differs (its `git_commit` field) but the `.clob` payload is
byte-identical. Confirm by hashing the `.clob` files alone.

### `test result: FAILED` on `cargo test --release`

If only doctests fail: you've added pseudo-code to a module header
as a raw indented block — Rust tries to compile it. Wrap in a ` ```text `
fence. See `src/metrics/mod.rs` and `src/perceive/active.rs` for
the pattern.

### J/nat rises after enabling `--adaptive-compute`

Expected on random weights (adaptive head isn't calibrated;
suppression gate is paying the cost of checking without the benefit
of skipping). If it *also* rises after training on a real corpus,
the adaptive gate is trading joules for no accuracy — either Head C
needs more epochs or `--adaptive-z` is too permissive.

### `calibrate-confidence` Pearson stays near 0

Model is under-trained. Random-weights models have no predictive
structure in their hidden states, so the head cannot learn to
predict NLL. Train the base model first (Phase K's `bench-suite`
will surface this; until then `calibrate` the energy critic to get
a trained Head N as a baseline).

### `train-router` drifts by >50% in a single run

Learning rate too high. Default is `1e-3`; try `3e-4`. Or
`--cf-rate` is too low — with too few counterfactuals, the router
overfits to its current favorite.

### `crystal` produces zero modules

Either the cluster coherence gate is rejecting everything (check
`token_entropy_bits` — must be below 2.0 by default) or the MDL
gate is (variance too high). Small corpora with random weights
hit both limits. Expect real modules on corpora ≥ 10K tokens with
trained weights.

## Beyond this document

- `ARCHITECTURE.md` for the philosophical frame.
- `PLAN.md` for what's shipping next.
- `docs/subsystems/` for the per-subsystem deep dives.
- `experiments/` for your own run logs.
