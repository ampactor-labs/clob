# clob

A ternary recurrent intelligence kernel in Rust. The floor hardware is a
Lenovo T490 (i7-8550U, 16 GB RAM); if it doesn't run there, it doesn't
count.

The bet: intelligence is compression efficiency per joule. The mechanism:
a crystallization loop that turns prediction errors into compiled ternary
modules and discards the raw episodes once the pattern is captured. Every
improvement should cost less energy, not more; if `J/nat` trends up, the
design is failing on its own terms.

## Quick start

```bash
cargo build --release

# 1. Create a random seed model (byte-reproducible given --seed)
./target/release/clob synth --output seed.clob --config small --seed 1

# 2. Tune the confidence + meta critics against a corpus
./target/release/clob calibrate-confidence \
    --model seed.clob \
    --corpus tests/data/eval_corpus.txt \
    --output conf.bin --meta-out meta.bin --seed 1 --epochs 3

# 3. Train MoE router weights via REINFORCE with counterfactuals
./target/release/clob train-router \
    --model seed.clob \
    --corpus tests/data/eval_corpus.txt \
    --output routers.bin --seed 1 --cf-rate 0.15 --max-tokens 1500

# 4. Stream a corpus with adaptive decode + metrics
./target/release/clob ingest \
    --model seed.clob --input tests/data/eval_corpus.txt \
    --memory-dir episodes \
    --confidence-head conf.bin --meta-critic meta.bin --routers routers.bin \
    --adaptive-compute --metrics-out m.jsonl

# 5. Read the dashboard (J/nat, tokens/sec, ΔCE)
./target/release/clob metrics --path m.jsonl

# 6. Drain stored episodes into crystallized modules
./target/release/clob crystal --memory-dir episodes --modules-dir modules --model seed.clob
```

## Core commands

| Command | What it does |
|:--|:--|
| `synth` | Generate a random seed model for a given config. |
| `ingest` | Stream a corpus through the model, record novel episodes, optionally log metrics and use adaptive compute. |
| `eval` | One-shot cross-entropy / perplexity evaluation. |
| `calibrate` | Train the energy critic (Head N) online against NLL. |
| `calibrate-confidence` | Train Head C + MetaCritic (target-network). |
| `train-router` | Train MoE routers with REINFORCE + counterfactuals. |
| `crystal` | Cluster stored episodes, distill patterns, emit ternary modules. |
| `active-ingest` | Choose inputs by acquisition score `max(N-C, 0) / joules_per_token`. |
| `metrics` | Tail + render a metrics jsonl as a table. |
| `compile` | Demo: lower a ternary matrix to x86-64 + dlopen. |

Every write command emits a `<output>.manifest.toml` sidecar with the
seed, git commit, and sha256 of every input, so any run can be reproduced
and audited.

## Deeper reading

- `ARCHITECTURE.md`: the design manifesto. Part V names five
  falsifiable bets and what would break each.
- `PLAN.md`: the current forward roadmap, session-resume dossier,
  and tiered ambition ladder (G–CC).
- `docs/running.md`: the runbook with expected numbers and a
  diagnostic ladder.
- `docs/subsystems/`: one page each on the crystallization loop, the
  multi-head critic, the compile pipeline, and symbolic synthesis.

## Current state

The supporting machinery is built and has been exercised locally: a
multi-head critic, adaptive decode, active input selection, a meta-critic
with a target network, REINFORCE-trained MoE routers, and symbolic
synthesis through a ternary-reducible DSL.

The core, though, is still random and untrained. The first real run
crystallized zero modules, and the reservoir probe found no contextual
gain from frozen hidden states beyond a trained unigram null. The current
track is Path B: untie and train the recurrent core first, then evaluate
crystallization again. See `docs/experiments/reservoir-probe.md` and
`docs/plans/path-b-core-training-build.md`.
