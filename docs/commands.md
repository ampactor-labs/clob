# Commands

Every `clob` subcommand, grouped by what it works on. `clob <command> --help`
lists each one's flags and defaults. Paths below assume the release binary at
`./target/release/clob`.

## Seed model pipeline (`CoreModel`, `.clob` files)

These run on a randomly initialized core; only the critics, routers and
crystallized modules are trained. [running.md](running.md) walks through them
in order with expected numbers.

| Command | What it does |
|:--|:--|
| `synth` | Generate a random seed model for a config (`tiny`, `small`, `seed`). The same `--seed` gives the same bytes. |
| `info` | Show a model file's config and size. |
| `calibrate` | Train the energy critic (Head N) online against per-token loss and save it to a side file. |
| `calibrate-confidence` | Train the confidence head (Head C); with `--meta-out`, also the MetaCritic that learns where Head C is unreliable. |
| `train-router` | Train mixture-of-experts routers with REINFORCE and counterfactual sampling. |
| `ingest` | Stream a corpus through the model, record novel episodes, optionally log metrics, checkpoint and use adaptive compute. |
| `active-ingest` | Choose inputs among several sources by acquisition score `max(N-C, 0) / joules_per_token`. |
| `eval` | One-shot cross-entropy and perplexity evaluation, optionally with crystal modules loaded. |
| `metrics` | Render a metrics jsonl file from `ingest --metrics-out` as a table. |
| `crystal` | Cluster stored episodes, distill patterns, write ternary modules. `--causal` refines clusters by what came next. |
| `status` | Show the episode memory and crystal status. |
| `bench-suite` | Run the ablation grid (adaptive on/off, modules present or absent, pristine or trained routers, seeds) and write a TSV. |
| `bench` | Measure throughput for a config. |
| `think` | Interactive sampling REPL. |
| `boot` | Create or load a seed model, open memory and enter the main loop. Ignores `--listen` and `--peer` today. |

## Tokenizer

| Command | What it does |
|:--|:--|
| `train-tokenizer` | Train a byte-pair tokenizer from a text file. |
| `encode` | Cache a corpus as a `.tokens` file so later runs skip tokenizing. |

## Path B: trained dense core (`.dense` files)

| Command | What it does |
|:--|:--|
| `train` | Train a dense core by truncated backpropagation through time with AdamW. `--qat` trains through the ternary weights. |
| `eval-dense` | Held-out loss of a `.dense` file next to the unigram baseline, in its ternary or f32 view. |
| `regime` | Estimate the largest Lyapunov exponent λ₁ of a `.clob` or `.dense` core: how fast its state forgets a small perturbation. |
| `crystal-dense` | The Bet 2 test: crystallize a trained core's errors on the train split, then score the holdout with modules loaded and cleared. |

## Early and demo code

| Command | What it does |
|:--|:--|
| `compile` | Demo: lower a random 64×64 ternary matrix to x86-64 and write it as a `.so`. It does not read crystallized modules. |
| `spore` | Package the binary, a seed model and a bootstrap script into one file. |
| `net discover`, `net connect` | Peer discovery over mDNS (marked a placeholder in the code) and a direct connection handshake. |

## Manifests

`synth`, `calibrate`, `calibrate-confidence`, `train-router`, `train`,
`bench-suite` and `regime --out` write a `<output>.manifest.toml` sidecar with
the kernel version, git commit, seed, arguments and the sha256 of every input,
so a run can be reproduced and audited. `ingest` writes the same manifest into
each checkpoint directory. Commands that draw random numbers take a `--seed`.
