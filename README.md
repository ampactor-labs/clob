# clob

A small Rust language model whose core weights are all -1, 0 or +1, built to
test whether a model can improve by compressing its own prediction errors. The
repo calls it a ternary recurrent intelligence kernel: ternary for those
weights, recurrent because it reads one token at a time through a fixed-size
state. Its crystallization loop turns clusters of those errors into small
add-on modules. In one small test on real text, a trained version predicted as
well as a full-precision copy, but the loop did not improve it.

**Status: prototype.** The pipeline and the trainer run end to end, but the crystallization loop failed its first test on a trained model, and the main runtime still runs on untrained core weights.

## Quick start

Needs a Rust toolchain; tested on x86-64 Linux with Rust 1.94.1.

```bash
cargo build --release
./target/release/clob synth --output seed.clob --config small --seed 1
./target/release/clob ingest --model seed.clob --input tests/data/eval_corpus.txt \
    --memory-dir episodes --metrics-out m.jsonl
./target/release/clob metrics --path m.jsonl
./target/release/clob crystal --memory-dir episodes --modules-dir modules --model seed.clob
```

`synth` writes a random `small` model, byte-identical for the same `--seed`.
`ingest` streams a 1,589-byte smoke corpus through it and stores surprising
tokens as episodes, and `crystal` tries to turn them into modules. With random
weights expect a loss near 7.0 nats (a uniform guess over 1,024 tokens scores
6.93), about 300 episodes and 0 modules. `metrics` prints (timings vary):

```text
 idx   tokens      sec          J      J/nat    ΔJ/nat%   mean_nll  Δmean_nll    tok/s
   0      500     0.91     13.699     0.0039     +0.00%     7.0693    +0.0000    547.4
   1      500     0.92     13.762     0.0039     +0.37%     7.0757    +0.0064    545.0
   2      500     0.84     12.533     0.0035     -8.84%     7.0950    +0.0257    598.4
   3       88     0.16      2.334     0.0038     -2.86%     7.0455    -0.0238    565.5

baseline J/nat = 0.0039 | latest J/nat = 0.0038 | ΔCE vs baseline = -0.0238
```

## Usage

### Commands

| Command | What it does |
|:--|:--|
| `synth` | Generate a random seed model for a config (`tiny`, `small`, `seed`). |
| `calibrate-confidence` | Train the confidence head (Head C) and, with `--meta-out`, the MetaCritic. |
| `train-router` | Train mixture-of-experts routers with REINFORCE (a policy-gradient method) and counterfactual sampling. |
| `ingest` | Stream a corpus through the model, record novel episodes, optionally log metrics and use adaptive compute. |
| `crystal` | Cluster stored episodes, distill patterns, write ternary modules. |
| `train` | Train a dense core by backpropagation through time (Path B). |
| `eval-dense` | Held-out loss of a trained `.dense` file, in its ternary or f32 view. |
| `crystal-dense` | The Bet 2 test: held-out loss of a trained core with and without crystallized modules. |

[docs/commands.md](docs/commands.md) lists all 24 subcommands. Seven, `synth`
and `train` among them, write a `<output>.manifest.toml` sidecar with the seed,
git commit, arguments and input hashes. [docs/running.md](docs/running.md) runs
the full pipeline with trained critics and routers.

### Train a dense core

```bash
./scripts/acquire_corpus.sh
N_MERGES=3000 ./scripts/tokenize_corpus.sh
./target/release/clob train --corpus data/corpus/train.tokens \
    --tokenizer data/corpus/tokenizer.bin --window 32 --lr 2e-3 \
    --weight-decay 0 --qat --steps 500 --output qat.dense
./target/release/clob eval-dense --dense qat.dense --window 32 \
    --corpus data/corpus/holdout.tokens --tokenizer data/corpus/tokenizer.bin
```

`N_MERGES=3000` matches the 3,260-token vocabulary of the Phase 6 runs in
Benchmarks (the default is 4,000). `eval-dense` prints held-out loss and the
unigram baseline. Phase 6 trained for 20,000 steps
([run.sh](experiments/2026-07-10_phase6_real_qat/run.sh)).

## How it works

The thesis is that intelligence is compression efficiency per joule: each
improvement should lower J/nat (joules per nat of loss), and a rising J/nat
means the design fails on its own terms. "Kernel" means the bottom layer of the
larger system that [ARCHITECTURE.md](ARCHITECTURE.md) plans. [PLAN.md](PLAN.md)
has the roadmap and [IF_FOUND.md](IF_FOUND.md) is a hand-over note.

**The model.** `CoreModel` (`src/model`, `src/nn`) runs each token through
blocks made of a selective state-space layer (a recurrent layer whose update
depends on the input, in the style of Mamba-2) and a gated channel mixer, which
on some layers is a mixture of experts with a learned router. Block weights are
packed at 2 bits each with one f32 scale per row (`src/tensor`), and AVX2
kernels with a scalar fallback do the additions and subtractions (`src/simd`);
the embedding table stays f32. Linear probes on the hidden state predict the
next token's loss: Head N drives novelty detection, and Head C adds passes
through the blocks on hard tokens.

**The loop.** `ingest` stores tokens with unusually high Head N scores as
episodes in a memory-mapped ring buffer. `crystal` clusters them (by hidden
state, or with `--causal` by what came next), keeps clusters whose next tokens
agree and whose pattern is cheaper to store than the episodes, and turns each
into a rank-1 ternary correction. A module adds its output when the hidden
state resembles its signature, and the buffer can then overwrite the episodes.
[docs/subsystems/](docs/subsystems/) covers each part.

**Path B.** A probe found no usable context in the random core's state
([reservoir probe](docs/experiments/reservoir-probe.md)), so the current track
trains the core. `src/learn` has a dense copy of the model with its own output
table, backpropagation through time checked against finite differences, and
AdamW on f32 weights. `--qat` (quantization-aware training) runs the forward
pass through the ternary weights, so training shapes the weights that get
deployed ([build log](docs/plans/path-b-core-training-build.md)).

Two decisions shaped it. Ternary weights turn matrix products into additions
and subtractions, so a laptop CPU is a realistic target: the floor hardware is
a Lenovo T490 (i7-8550U, 16 GB RAM), and if it does not run there, it does not
count. Each bet also has pass and kill thresholds fixed before the run that
judges it (ARCHITECTURE.md Part V, [ATTRACTOR.md](ATTRACTOR.md) for Bets 6 to
8), so a result can fail instead of being reinterpreted.

## Benchmarks

Pre-registered Phase 6 and Bet 7 results from `experiments/`. Loss is mean
negative log-likelihood per token in nats on the held-out last 10% of
*Moby-Dick* (lower is better; token frequencies alone score 4.7765). The model
is the dense `small` preset (d_model 64, 3 layers, 579,032 parameters, seed 1)
at step 15,000 of 20,000. The run records do not name the hardware.
[docs/results.md](docs/results.md) has every run and how to repeat it.

| Measurement | Result | Pass / kill | Verdict |
|:--|:--|:--|:--|
| Bet 1: held-out loss, ternary QAT core vs matched f32 core | 3.7972 vs 3.8016 (ratio 0.9989) | ≤ 1.25 / > 1.5 | pass |
| Bet 1: recurrent-core weight memory | 12.65× smaller than f32 | ≥ 10× / < 8× | pass |
| Whole-model memory, with the f32 embedding and output tables | 1.34× smaller than f32 | not judged | |
| Bet 2: held-out loss with crystallized modules, ternary core | 0 modules, 0.00% change | ≥ 2% lower / ≤ 0% | kill |
| Bet 2 on the f32 core | 2 modules, 3.61% worse | not judged | |
| Bet 7: Spearman ρ between memory length (λ₁) and capability | −0.27 over 9 checkpoints | ≥ 0.6 / \|ρ\| < 0.2 | indeterminate |

The 0.9989 ratio is parity on one seed. Bet 7 expected capability to rise as
the state's memory grew, but the best checkpoints forget fastest, which
falsifies that hypothesis even though the literal verdict is indeterminate.

## Data

The real corpus is *Moby-Dick* (Project Gutenberg #2701, public domain).
`scripts/acquire_corpus.sh` downloads it, strips the Gutenberg boilerplate and
table of contents, and splits it by line into train (first 90%) and holdout
(last 10%); `scripts/tokenize_corpus.sh` trains a byte-pair tokenizer and
caches both splits as `.tokens` files. The repo commits only checksums
([data/corpus/README.md](data/corpus/README.md)), and a download on 2026-09-26
did not match them. On a fresh clone the script overwrites the committed
checksums instead of stopping. Exact Phase 6 numbers need the original bytes,
which the repo does not hold. `data/synthetic/` holds a corpus with [planted
patterns](data/synthetic/expected.md).

## Testing

Run `cargo test --release`. It and `cargo test` each passed all 123 tests on
2026-09-26: 120 unit tests in the library and 3 in `tests/data_split.rs`. They
cover gradient checks for each layer and the whole dense model, a trainer smoke
test that needs memory to pass, DSL synthesis, causal refinement, the distill
gates, the Lyapunov estimator, the Bet 2 harness, checkpoints, manifests and
the train/holdout contamination check. The CLI commands, SIMD kernels, x86-64
emitter and network code have no tests, and there is no CI. No test can say
whether a bet holds; the experiments do that.

## Limitations

The central mechanism does not work yet. In its first test on a trained model,
crystallization built no modules from the deployed ternary core, and on the
full-precision (f32) version the two modules it built raised the loss on unseen
text by 3.6%. The trained core lives in a separate training path, so `ingest`,
`crystal` and `think` still use random core weights. All trained results so far
come from one small model trained with one seed on one book.

- Only the recurrent-core matrices are ternary. The f32 embedding and output
  tables hold 72% of the dense model's parameters, so it is 1.34× smaller.
- J/nat is an estimate from wall-clock time, a 15 W TDP and the load average.
- `compile` lowers a random matrix with a scalar emitter, and nothing calls the
  dlopen loader, so Bet 4 (compiled modules beat a generic kernel) is untested.
  Bets 3 and 5 (forgetting and long-run convergence) are untested too.
- Symbolic hints on modules never run, trained routers stay f32, `boot`
  ignores `--listen` and `--peer`, and the code marks mDNS discovery as a
  placeholder.
- `.cargo/config.toml` builds for the host CPU (`target-cpu=native`), the
  experiment scripts hardcode the author's paths, and logging is `eprintln!`.

## Roadmap

1. **Redesign crystallization.** The Bet 2 run names two suspects: capture
   keeps only tokens with error above 0.5, the least coherent ones, and
   distillation has no held-out check. Each fix needs its own pre-registered
   run, and neither is written yet.
2. **Test Bet 7 on long-range text.** Fast forgetting may be specific to one
   novel. The test needs a corpus with long-range dependencies, which the repo
   does not have yet.

## License

No license chosen yet.
