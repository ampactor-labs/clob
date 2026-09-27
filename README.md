# clob

A Rust experiment that trains a ternary recurrent language model (each weight is -1, 0 or +1) and tests whether it can improve by compressing its own errors. Everything must run on a Lenovo T490 laptop (i7-8550U, 16 GB RAM) with no GPU. The idea under test is a crystallization loop that distills clusters of high-error predictions into small ternary modules. In a small first test the trained ternary core matched a same-size 32-bit float model on held-out text, but the loop added no gain on top of it.

**Status: prototype.** The central bet is unproven, and the first real run crystallized zero modules.

## Quick start

You need a Rust toolchain; these commands were checked with Rust 1.94 on Linux x86-64. They build the binary, then run the crystallization pipeline end to end on a randomly initialized model and the 1,589-byte smoke corpus in `tests/data/`, writing their outputs to the current directory.

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

On a 4-core Linux container the build took 90 seconds, and the six steps took 20 to 35 seconds across two runs. The core's weights stay random, so this run exercises the plumbing only. `ingest` reports a mean NLL (negative log-likelihood) of about 7.1. NLL measures the model's surprise at each next token in nats (natural-log units), and lower is better; a uniform guess over the model's 1,024-token vocabulary scores 6.9. `ingest` also stores a few hundred surprising predictions as episodes (281 in the check run), and `crystal` ends with `0 new modules saved`.

## Usage

`./target/release/clob --help` lists all 24 subcommands, and `clob <command> --help` shows each one's flags. The main ones:

| Command | What it does |
| :-- | :-- |
| `synth` | Write a randomly initialized `.clob` model in the `tiny`, `small` or `seed` config. |
| `ingest` | Stream a corpus through a `.clob` model and store surprising predictions as episodes; it can also log metrics, run extra decode passes on uncertain tokens (adaptive compute) and checkpoint. |
| `eval` | Report the mean NLL and perplexity (e to the power of the NLL) of a `.clob` model on a corpus, with or without modules loaded. |
| `calibrate` | Train the energy critic (Head N), a linear probe that predicts the next token's loss from the hidden state. |
| `calibrate-confidence` | Train the confidence head (Head C), which triggers extra decode passes, and the meta-critic that suppresses them where Head C is unreliable. |
| `train-router` | Train the mixture-of-experts routers, which choose the expert networks that process each token, with REINFORCE (a reinforcement-learning method), sometimes sending a token to its second-choice expert. |
| `crystal` | Cluster stored episodes, distill the coherent clusters and write ternary modules. |
| `active-ingest` | Choose the next input source by `max(N - C, 0) / joules_per_token`, where N is Head N's predicted surprise and C is Head C's predicted loss. |
| `metrics` | Print a metrics `.jsonl` file as a table. |
| `bench-suite` | Evaluate every on/off combination of adaptive compute, modules and trained routers for each seed given, and write a TSV. |
| `train` | Train a dense core on windows of tokens; `--qat` trains through the ternary weights. |
| `eval-dense` | Report a trained core's per-token loss next to a unigram baseline, which predicts each token from its overall frequency. |
| `regime` | Estimate how fast the core's state forgets a small change (its largest Lyapunov exponent), which bounds how many tokens of context survive. |
| `crystal-dense` | Run the equal-compute crystallization test (Bet 2) on a trained core. |
| `compile` | Demo: lower a random 64×64 ternary matrix to x86-64 and write it as an ELF shared object (a Linux `.so` library). |

The others cover the boot loop and REPL (`boot`, `think`), a throughput benchmark (`bench`), inspection (`info`, `status`), tokenizers (`train-tokenizer`, `encode`) and the peer-to-peer layer (`spore`, `net`).

### Train a core

Trained models come from a separate dense training path. These commands train a small core on the smoke corpus, score it, and measure how fast its state forgets:

```bash
./target/release/clob train --corpus tests/data/eval_corpus.txt --config mini --qat --output mini.dense
./target/release/clob eval-dense --dense mini.dense --corpus tests/data/eval_corpus.txt
./target/release/clob regime --dense mini.dense --corpus tests/data/eval_corpus.txt
```

`--qat` (quantization-aware training) runs the forward and backward passes through the ternary weights, so the ternary version is the model that learns. On the same container training took 5 to 7 seconds. `eval-dense` printed an NLL of 1.80 against 3.02 for the unigram baseline, and `regime` reported a memory horizon of 3.5 tokens, the span over which a small change to the state shrinks by a factor of e. The NLL is measured on the training text, so it shows the trainer works; held-out results are under Benchmarks.

### Reproducibility

Two `synth` runs and two `train` runs with the same flags produced byte-identical files. `synth`, `calibrate`, `calibrate-confidence`, `train-router`, `train`, `bench-suite` and `regime --out` also write a `<output>.manifest.toml` next to their output. It records the git commit, the arguments, the seed where there is one, and the sha256 of the main input files. `ingest` writes one only inside checkpoints, and `crystal`, `crystal-dense`, `encode` and `train-tokenizer` write none.

## How it works

The design bets that intelligence is compression efficiency per joule: predictive quality counts only relative to the energy spent on it. I chose ternary weights and a recurrent core because the target is a CPU without a GPU. A matrix product over weights of -1, 0 and +1 needs only additions and subtractions, plus one scaling multiply per output row. A recurrent core reads one token at a time and carries a fixed-size state forward, so its memory per token stays constant however long the text runs.

**The core.** Weights are packed at 2 bits each with one scale per row, and the kernels use AVX2 vector instructions when the CPU has them, with a scalar fallback. Each block pairs a selective state space model (SSM), a recurrent layer that updates a fixed-size state at every token, with a gated layer that mixes features within each token. Some blocks swap that second layer for a mixture of experts. There are two model paths. `synth`, `ingest`, `crystal` and the critic commands use a `.clob` model whose core weights stay at their random initialization. `train` runs a separate dense path, a core with no mixture-of-experts layers because their routing has no gradient. It uses truncated backpropagation through time (gradients flow back through a fixed window of tokens), with the AdamW optimizer updating 32-bit shadow weights that are projected to ternary. The tests check each layer's backward pass against finite differences.

**The loop.** `ingest` streams text through the model. The energy critic predicts each token's loss from the hidden state, and an adaptive threshold on that prediction flags surprising tokens. Each one is saved as an episode: the context, hidden state, top predictions and the 8 tokens that actually came next. `crystal` clusters episodes by hidden state with k-means, and `--causal` then splits and merges clusters by what follows them. Distillation keeps a cluster only if two tests pass: its next tokens must mostly agree (at most 2 bits of entropy), and a minimum description length (MDL) test must find that a pattern plus residuals takes under 95% of the bits of the raw episodes. Each kept cluster becomes a module: a ternary matrix built from the cluster's average input and average correction. The model adds a module's output to the hidden state, scaled by the state's cosine similarity to the module's signature, whenever that similarity passes a threshold. Later cycles skip the episodes a module consumed. A separate compiler lowers ternary matrices to x86-64 ELF shared objects, but nothing loads them yet.

**Energy.** `ingest --metrics-out` logs J/nat for each window of tokens: estimated joules divided by the summed per-token loss in nats. The joules come from wall-clock time, a 15 W TDP (the i7-8550U's rated power) and the 1-minute load average per core; no power meter is read.

**Pre-registered bets.** [ARCHITECTURE.md](ARCHITECTURE.md) and [ATTRACTOR.md](ATTRACTOR.md) list eight bets, each with a pass line and a kill line committed before the run that judges it, and a threshold may change only in a commit that does not also report a result. I chose this so that a run can fail on the record, which is how Bet 2 was killed.

### Further reading

- [ARCHITECTURE.md](ARCHITECTURE.md): the design, Bets 1 to 5 with their thresholds (Part V), and a status note per bet.
- [ATTRACTOR.md](ATTRACTOR.md): the second thesis, compressing the past into the model's state, with Bets 6 to 8.
- [PLAN.md](PLAN.md): the roadmap, the session-resume notes and the phase ladder from G to CC.
- [IF_FOUND.md](IF_FOUND.md): a plain-language account of what works and what does not.
- [docs/running.md](docs/running.md): the runbook for the random-model pipeline, with expected numbers per step and a diagnostic ladder.
- [docs/subsystems/](docs/subsystems/): one page each on the crystallization loop, the critics, the compiler and symbolic synthesis.
- [docs/experiments/reservoir-probe.md](docs/experiments/reservoir-probe.md) and [docs/plans/path-b-core-training-build.md](docs/plans/path-b-core-training-build.md): why the random core failed as a fixed reservoir, and how the core is now trained.

## Benchmarks

Each test's pass and kill lines were committed before its result, in ARCHITECTURE.md, ATTRACTOR.md or the experiment's `notes.md`. The numbers come from the files in the linked directories. A `runinfo.txt`, where present, records the commit and the binary's sha256; no file records the machine. The real-corpus runs used Moby-Dick (see Data), the dense `small` preset (width 64, 3 layers), seed 1, 32-token windows and up to 20,000 training steps. In the Bet 1 runs, the checkpoint and the better of two optimizer settings were chosen on the same held-out split that the scores use, as registered. To rerun a test, edit the paths at the top of its `run.sh`, which point at the author's checkout, and expect the corpus mismatch described in Data.

| Test | Measured | Pass / kill line | Verdict |
| :-- | :-- | :-- | :-- |
| [Bet 1](experiments/2026-07-10_phase6_real_qat/): held-out NLL of the QAT ternary core vs a same-size 32-bit core | 3.797 vs 3.802 (0.9989×) | ≤ 1.25× / > 1.5× | pass |
| [Bet 1](experiments/2026-07-10_phase6_real_qat/): size of the core's weight matrices vs 32-bit, computed from their shapes | 12.65× smaller (1.34× for the whole model) | ≥ 10× / < 8× | pass |
| [Bet 2](experiments/2026-07-10_phase6_bet2/): held-out NLL with modules loaded vs cleared, at equal compute | 0 modules formed on the ternary core; 0.0% change | ≥ 2% lower / not lower | kill |
| [Bet 2](experiments/2026-07-10_phase6_bet2/) context row: the same test on the 32-bit core | 2 modules; NLL 3.6% higher | reported only | loss |
| [Bet 7](experiments/2026-07-08_bet7/): Spearman ρ between λ₁ and capability across checkpoints | ρ = −0.27 over all 9; −0.81 over the 8 trained ones | ρ ≥ 0.6 / ρ between −0.2 and 0.2, each with a λ₁ condition | indeterminate |
| [Planted patterns](experiments/2026-07-08_qat_planted_projection/): whole-corpus NLL of a QAT core on `diagnostic.txt` | 0.094 vs 3.153 for unigram | ≤ 0.75× unigram plus per-section lines / not below unigram | pass |

λ₁ is the largest Lyapunov exponent from `regime`: the rate per token at which a small change in the model's state grows (positive) or dies out (negative). ρ is the Spearman rank correlation, and capability is the negative of held-out NLL. Bet 7 predicted that capability would rise as λ₁ moved up toward zero. It rose as λ₁ fell, a case the registered kill line did not cover, so the literal verdict is indeterminate while the hypothesis itself is falsified.

## Data

- **Moby-Dick** (Project Gutenberg #2701, public domain) is the real corpus. `scripts/acquire_corpus.sh` downloads it, strips the Gutenberg header, footer and table of contents, and puts the last 10% of lines in the held-out split. `scripts/tokenize_corpus.sh` then trains a byte-pair-encoding (BPE) tokenizer with 4,000 merges and encodes both splits. The text is not committed; `data/corpus/*.sha256` pins the expected bytes.
- **`data/synthetic/diagnostic.txt`** (1,946 bytes) plants patterns with known answers, listed in `data/synthetic/expected.md`, to check the loop before real text.
- **`tests/data/eval_corpus.txt`** (1,589 bytes) is the smoke corpus for the Quick start.

The pins no longer match the source. Gutenberg's file was last modified on 2026-09-07, after the pins were committed on 2026-07-03, and a download on 2026-09-27 produced different hashes for both splits. On a fresh clone the script overwrites the committed pins without comparing them, so check `git diff data/corpus` after running it. [data/corpus/README.md](data/corpus/README.md) has the provenance and the variables for using another text.

## Project layout

```text
src/main.rs     every subcommand
src/model/      the .clob inference core: blocks, router, adaptive decode
src/nn/         layers and the critic heads
src/learn/      the dense training path: backward pass, AdamW, QAT
src/crystal/    clustering, causal refinement, distillation, modules, program synthesis
src/compile/    ternary matrix to IR, x86-64 and ELF, plus a dlopen loader
src/dynamics/   the Lyapunov regime instrument
src/eval/       bench-suite and the Bet 2 harness
src/net/        peer discovery, module exchange, spores
docs/           runbook, subsystem notes, the Path B plan
experiments/    one directory per registered run
scripts/        corpus download and tokenizing, first_run.sh, diagnostic_run.sh
```

## Testing

```bash
cargo test --release
```

This runs 123 tests: 120 unit tests in the library and 3 in `tests/data_split.rs`, with no doctests. On a 4-core Linux container it took 81 seconds, nearly all of it compiling. The tests check ternary packing, each trainable layer's backward pass and the full dense model's against finite differences, a training run on a pattern that only a working recurrence can learn, recovery of short matrix programs from planted matrices, causal refinement, the distillation gates, the Lyapunov estimator on maps with known exponents, the Bet 2 harness, seeding, manifests, checkpoints, `.tokens` files and the train/holdout overlap check. The overlap check on the real corpus skips itself when `data/corpus/` has no text.

Nothing runs in CI: the repository has no workflow files. The x86-64 emitter, the ELF writer and loader, the SIMD kernels, the episode store, the `.clob` inference stack and the network layer have no tests of their own. No test can say whether the central bet holds; the experiments do that (see Benchmarks).

## Limitations

The central bet is unproven. The design depends on a crystallization loop that should improve the model by turning its prediction errors into small ternary modules, but the first real run crystallized zero modules. The first test on a trained core did no better: its ternary version formed zero modules again, and the two modules that formed on a matching 32-bit core raised held-out loss by 3.6%.

- **The `.clob` pipeline runs on random weights.** `synth` only initializes a `.clob` model, and no command trains its core weights. The working trainer, `train`, writes a separate `.dense` artifact that `ingest`, `crystal` and the critic commands cannot load.
- **Bet 1 passed at small scale only.** It used one seed and one corpus with a 64-wide core. The embedding and readout tables stay 32-bit in both arms, so the whole model is only 1.34× smaller.
- **Only Bets 1, 2 and 7 have been run.** Bet 3 (forgetting without losing accuracy), Bet 4 (compiled modules beating generic kernels), Bet 5 (long-run improvement), Bet 6 (causal clustering beating plain clustering) and Bet 8 (a training loss over several future tokens, which `train` does not implement) are untested.
- **Forgetting is a flag.** Crystallization marks the episodes a module used as consumed, and they stay on disk until the fixed-size store overwrites them in oldest-first order, consumed or not.
- **The compiler is outside the loop.** The x86-64 emitter produces scalar code (one `addss` or `subss` per nonzero weight), the AVX2 speed target in ARCHITECTURE.md is unmeasured, and nothing loads compiled modules.
- **J/nat does not measure power.** The joules come from a TDP proxy, and the ratio divides them by total loss, so a model that predicts better at the same cost reports a higher J/nat.
- **Two parts stay outside the ternary compression.** A module that reduces to a short program in the matrix language (shifts, negations, reversals and masks) keeps the program only as a hint and still runs its matrix, and trained routers stay 32-bit.
- **The peer-to-peer layer is untested.** Discovery, a handshake, module exchange and spore packaging exist, but no test or recorded run uses them.
- **The pinned corpus cannot be re-downloaded.** Gutenberg's copy changed after the pins were made (see Data).

## Roadmap

1. **Redesign crystallization.** The Bet 2 kill points at two parts. Capture keeps only tokens where the core put less than 0.5 probability on the right answer, which may leave out coherent structure, and distillation has no held-out check, so it built modules from clusters the core gets flatly wrong. Each change needs its own pre-registered run, and none is registered yet.
2. **Test Bet 7 on long-range text.** Next-token prediction on Moby-Dick is mostly local, so the Bet 7 result may not carry over. The test needs a corpus with long-range dependencies, which the project does not have yet.
3. **Add CI and leveled logging (Phase M).** PLAN.md puts this after the experiments above, because a signal on the central bet outranks code polish.

## License

No license chosen yet.
