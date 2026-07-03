# PLAN.md — clob readiness and ambition roadmap

This document is the project's forward plan *and* the session-resume
dossier. It is the single source of truth a new Claude (or new human
contributor) should read to pick up exactly where the project left off.

---

# Session-resume dossier (read this first)

## What `clob` is

A self-improving ternary-recurrent AI kernel written in Rust, targeting
a Lenovo T490 (i7-8550U, 16 GB RAM) as its floor hardware. The core
thesis: intelligence = compression efficiency per unit energy. The
mechanism: a crystallization loop that turns prediction errors into
compiled ternary modules and discards the raw episodes once compressed.
See `ARCHITECTURE.md` for the long-form manifesto (now pruned of the
"unsurpassable" rhetoric of earlier drafts).

## State (updated 2026-07-03 — first real run + reservoir probe completed)

**Branch:** `main`, 18 commits ahead of `origin/main` (not pushed), with
local cleanup/docs work in progress.

**Commit history (most recent first):**
- `d0e900a` — Add IF_FOUND.md: the kernel's honest self-record
- `271ab2a` — Add scripts/first_run.sh: one-command unattended real-data run
- `a70dbce` — Harden the .tokens cache and peer handshake; audit runtime unwraps
- `cca14b0` — Pre-register bet thresholds and a ground-truth-first run discipline
- `e562162` — Phase L scaffolding: .tokens cache, encode subcommand, split-check, corpus scripts
- `9dfb4aa` — Refresh PLAN.md state block and Phase K scope to current reality
- `88827f4` — Phase K: bench-suite ablation harness (trajectory.rs deferred to L)
- `c280808` — Phase J: checkpoint + resume for ingest
- `41f1142` — Phase I: docs layer — README, runbook, subsystem deep dives, CLAUDE.md
- `615e139` — Phase H: manifests + experiments convention
- `6f4988b` — Phase G: model save path + SeedTree for reproducibility
- `8369dd3` — Phase E: symbolic crystallization (DSL synthesis, hint-only)
- `b8c23fd` — Phase F (2/2): router crystallization via REINFORCE
- `f7388a9` — Phase F (1/2): MetaCritic with target-network
- `413f0f1` — Phase D: active input selection
- `64626ac` — Phase B+C: Head C (confidence) + adaptive decode
- `abaa160` — Phase A: metrics subsystem + J/nat dashboard
- `827c890` — (pre-supercharge) Sharpened ARCHITECTURE.md + fixed distill MDL math

**Test suite:** 70+ tests are expected to pass (`cargo test --release`). Includes unit
tests for distill/MDL, router policy gradient, meta-critic target-network
refresh, symbolic-synth planted-pattern recovery, active selection
acquisition score, metrics window serialization, byte-identical model
save (Phase G), manifest round-trip (Phase H), checkpoint resume
(Phase J), bench-suite summarization (Phase K), split enforcement
(Phase L), and more.

**Tier 1 phase status:** G ✓ · H ✓ · I ✓ · J ✓ · K ✓ · L partial · N
partial. The first real run completed locally on the default Moby-Dick corpus:
it trained heads/router, ingested the corpus, checkpointed, ran bench-suite,
and crystallized **zero modules**. A follow-up reservoir probe over frozen-core
hidden states measured **zero contextual gain** over a trained unigram null.
The next gate is no longer "run first_run"; it is **Path B core training**.
See `docs/experiments/reservoir-probe.md` and
`docs/plans/path-b-core-training-build.md`.

**End-to-end pipeline verified** on synthetic weights against
`tests/data/eval_corpus.txt` (1.5 KB smoke corpus):
```
synth → calibrate-confidence → train-router → ingest (adaptive + meta + routers) → crystal → active-ingest → metrics
```
Every subcommand composes; signals are uninformative only because the
model is random-weights, not because the plumbing is broken.

## Critical files a fresh session should read

In rough order:

1. **`PLAN.md`** (this file) — orientation, state, roadmap.
2. **`ARCHITECTURE.md`** — the design manifesto, Part V now names five
   falsifiable bets and what would break each.
3. **`src/lib.rs`** — module tree; every subsystem has its own dir.
4. **`src/main.rs`** — every subcommand (Cli + command dispatch).
5. **`src/crystal/{distill,crystallize,engine,synth,module}.rs`** — the
   crystallization loop.
6. **`src/nn/{confidence,energy}.rs`** — the multi-head critic + meta.
7. **`src/model/{stack,router,block}.rs`** — core + MoE router +
   adaptive decode.
8. **`src/metrics/{mod,window}.rs`** — the dashboard substrate.
9. **`src/perceive/active.rs`** — active input selection.
10. **`docs/experiments/reservoir-probe.md`** — why Path A is currently
    treated as failed.
11. **`docs/plans/path-b-core-training-build.md`** — the next build track.
12. **`/home/suds/.claude/projects/-home-suds-Projects-clob/memory/project_clob.md`**
    — long-lived project memory with shipped-state summary; update it
    at the end of every major work session.

## Quick-start verification (run after any pull)

```bash
# 1. Smoke-build
cargo build --release

# 2. Smoke-test (must show all tests passed; 70+ expected)
cargo test --release 2>&1 | grep "test result"

# 3. End-to-end pipeline (takes ~1 min)
cd /tmp && rm -rf clob_verify && mkdir clob_verify && cd clob_verify
/path/to/clob/target/release/clob synth --output seed.clob --config small --seed 42
/path/to/clob/target/release/clob calibrate-confidence \
    --model seed.clob --corpus /path/to/clob/tests/data/eval_corpus.txt \
    --output conf.bin --meta-out meta.bin --meta-refresh 300 --epochs 3
/path/to/clob/target/release/clob train-router \
    --model seed.clob --corpus /path/to/clob/tests/data/eval_corpus.txt \
    --cf-rate 0.15 --max-tokens 1500 --output routers.bin
/path/to/clob/target/release/clob ingest \
    --model seed.clob --input /path/to/clob/tests/data/eval_corpus.txt \
    --memory-dir episodes \
    --confidence-head conf.bin --meta-critic meta.bin --routers routers.bin \
    --adaptive-compute --adaptive-max-extra 2 --adaptive-z 0.5 \
    --metrics-out m.jsonl --metrics-window 500 --max-tokens 1500
/path/to/clob/target/release/clob metrics --path m.jsonl
```

Expected: clean run, 3 windows of metrics, J/nat ≈ 0.006, mean_nll ≈ 7.0,
some counterfactuals and some meta-suppressions.

## Known gaps remaining

Phases G–L/N closed the original blockers: the model-save path, top-level
seed threading (no `thread_rng` in production paths), config manifests,
checkpoint/resume, bench-suite, token caches, split checks, and first-run
script all exist now. Still open:
- The core does not learn. `src/learn` has pieces, but there is no supervised
  backward pass, no untied readout, and no `train` command. **→ Path B.**
- The first real run crystallized zero modules because the frozen random core
  did not expose useful structure. Crystallization should wait for a trained
  core before being judged again.
- Scalar x86 emitter (`addss`/`subss` per trit); AVX2 is aspiration,
  not measurement. **→ Phase O.**
- `eprintln!` everywhere (171 calls, 158 in `main.rs`); no leveled
  logging, no CI. **→ Phase M.**
- Symbolic crystallization is hint-only metadata; `Program::apply` is
  implemented but unwired — not on the inference hot path. **→ Phase Q.**
- MoE routers stay dense f32 after REINFORCE training; not
  re-ternarized into the same compression regime as experts. **→ Phase R.**
- All five falsifiable bets remain untested/instrumented — there is no
  real-data signal yet. **Unblocked by Phase L.**

## Global conventions

- **Commits:** no Claude attribution (`~/.claude/CLAUDE.md` is strict
  about this). Plain messages, user authorship, no Co-Authored-By.
- **Tests:** every phase lands with its own unit tests; doctests in
  module headers must use `text` blocks (Rust tries to compile them).
- **Schema evolution:** every new serde field gets `#[serde(default)]`
  so old artifacts load cleanly — Phase E's `symbolic_hint` sets the
  pattern.
- **Subcommand additions:** follow the existing pattern (Commands enum +
  match arm in main + `cmd_<name>` function). Keep flags scoped per
  subcommand; no cross-cutting globals.

## The memory directory

`/home/suds/.claude/projects/-home-suds-Projects-clob/memory/` contains
`MEMORY.md` (index) and individual entries. At session start a fresh
Claude should read `MEMORY.md` (auto-loaded) and follow links as
relevant. At session end, update `project_clob.md` with the shipped
state + any new gaps discovered.

## Companion plan file

A mirror of this plan also exists at
`/home/suds/.claude/plans/write-a-plan-that-cosmic-aho.md` (the
plan-mode workspace). PLAN.md here is canonical; the other is a
working copy from the session that generated this.

---

# The plan itself

## Context

The five-phase supercharge (Phases A → B+C → D → F → E, commits
`abaa160` through `8369dd3`) shipped a working multi-head critic,
adaptive decode, active selection, meta-critic target network, router
policy-gradient training, and symbolic crystallization. Every subsystem
is verified on synthetic weights against a 1.5 KB smoke corpus; the
plumbing is honest but the signals are uninformative because nothing
has been trained on real data.

Before committing to a real run — the first ~day+ where the
crystallization loop meets a meaningful corpus and J/nat becomes
diagnostic — the project needs operational scaffolding that does not
yet exist: reproducibility primitives, checkpoints, a save path, a
real evaluation harness, data-acquisition scripts, docs, and the
supporting experiment conventions. That's Tier 1.

Tier 2 closes the remaining 50% of Phases E and F that the original
commits cut for diff hygiene. Tier 3 expands the architecture into
genuine cognitive-architecture territory (composition, dual-process,
federation, DSL expansion). Tier 4 reaches for "ultimate" — embodied
perception, actor mode, self-modifying compiler, counterfactual
imagination, introspection, hardware autotuning, and self-documentation.

## Guiding principles

- **Reproducibility comes first.** Nothing else matters if the same
  inputs don't produce the same outputs. Seeds, config manifests,
  checksums, deterministic iteration — these are blockers, not polish.
- **Reuse the existing skeleton.** `WeightAccum` at
  `src/io/synth.rs:10-64` is the model-save scaffold. `io::format`
  writes headers; `io::loader` reads them. The save path is almost
  write-once code; don't overbuild.
- **Each phase ships as one commit, self-contained, with a smoke test.**
  Nothing half-wired; no "will be populated later" fields that aren't
  already populated.
- **Defer until blocked.** Don't build bench-suite infrastructure until
  the model-save path exists (needed for A/B snapshots). Don't build
  `scripts/first_run.sh` until every command it calls is stable.
- **Watch for drift from the five-phase commits.** Phase E attached
  `symbolic_hint` with `#[serde(default)]` so old module files load.
  Every new schema addition maintains that discipline.
- **Tier discipline.** Land Tier 1 before Tier 2. Tier 2 before Tier 3.
  Tier 3 before Tier 4. Shortcuts across tiers produce unmaintainable
  systems that look ambitious and behave brittle.
- **Signal before polish.** *Within* Tier 1, the first real-corpus signal
  outranks code-quality polish. As soon as Phase L lands a corpus, do an
  exploratory real-data run — even ad-hoc, before Phase N's polished
  `first_run.sh` — and defer Phase M (logging, CI, the eprintln sweep)
  until after. M is deferrable; not knowing whether the core bet holds is
  not. A green logger on a model that has never seen real data proves
  nothing. (Run the planted-pattern `scripts/diagnostic_run.sh` first of
  all — it has known answers in `data/synthetic/expected.md`.)

## Tier 1 — Readiness (Phases G–N)

### Phase G — Model save + top-level seed (the blockers)

Without these, nothing downstream is coherent: no checkpointing without
save, no reproducibility without seed threading, no bench-suite without
model snapshots.

**Create:**
- `src/io/writer.rs` — `pub fn save_model(model: &CoreModel, path: &Path)
  -> io::Result<()>`. Mirrors `generate_synthetic` structure using
  `WeightAccum` (`src/io/synth.rs:10-64`): for each subsystem, push its
  weights in the same names `loader.rs` expects. Extract the per-layer
  naming into a small helper so save and load cannot drift. Round-trip
  test: `save_model(m) → load_model(p) → assert_model_eq`.
- `src/util/seed.rs` — `pub struct SeedTree` deriving child RNGs
  deterministically from a top-level u64. `SeedTree::child("critic")`,
  `SeedTree::child("router")`, etc. — every RNG construction in the
  codebase pulls from the tree, never from `thread_rng`.

**Modify:**
- Every subcommand that takes a seed gets `--seed u64` as the single
  source of entropy. Audit: `main.rs` currently seeds with `42`, `7`,
  `11`, `13`, `thread_rng`, and (via `rand::rngs::StdRng::seed_from_u64`)
  ad-hoc integers. All of those become `SeedTree` children.
- `cmd_synth` takes `--seed`. Passes it to both model init and the
  manifest.
- `cmd_calibrate_confidence` previously hardcoded `seed_from_u64(42)` —
  takes `--seed`.
- `cmd_train_router` previously hardcoded `seed_from_u64(7)` — takes
  `--seed`.
- `NoveltyDetector::new()` and all critic/head initializations take a
  seed from the tree. No `thread_rng` in production code paths.

**Verify:** `synth --seed 1 && save load && save load` produces
byte-identical files on two runs. `calibrate --seed 1` twice produces
byte-identical critic.bin. `train-router --seed 1` twice produces
byte-identical routers.bin.

### Phase H — Config manifests + experiment directories

Once Phase G ships, runs become comparable. Now make them *traceable*:
every artifact carries its manifest, and every non-trivial run lives in
its own directory.

**Create:**
- `src/config/manifest.rs` — `pub struct RunManifest` with: kernel
  version (from `CARGO_PKG_VERSION`), git commit (via `env!`), seed,
  config name, tokenizer sha256, corpus sha256, subcommand args, UTC
  timestamp. Serialize as TOML (add `toml = "0.8"` to Cargo.toml —
  small, widely used, belongs here).
- `src/util/sha.rs` — tiny wrapper over `sha2` (new dep) for file
  hashes. Used by manifest + split-enforcement.
- `experiments/.gitkeep` and `experiments/README.md` explaining the
  convention.

**Modify:**
- Every write-subcommand (`synth`, `calibrate`, `calibrate-confidence`,
  `train-router`, `ingest` with metrics, `crystal`) emits a
  `<output>.manifest.toml` alongside its primary output.
- Every read-subcommand that consumes artifacts from prior commands
  cross-checks manifest hashes. Mismatch: warn by default, `--strict`
  to fail.
- `experiments/` convention: `scripts/new_experiment.sh <name>` scaffolds
  `experiments/YYYY-MM-DD_<name>/` with `manifest.toml`, `notes.md`
  template, and a `run.sh` stub.

**Verify:** Running `synth --seed 1` twice from identical commits
produces the same `seed.clob.manifest.toml`. Loading a critic with a
manifest from a different model file prints a clear warning and, under
`--strict`, aborts.

### Phase I — Docs layer

Now that runs are reproducible and traceable, docs become worth writing.
Before this phase, docs would describe incomplete behavior.

**Create:**
- `README.md` — 500 words. Hook sentence. "The kernel is a ternary
  recurrent intelligence that compiles experience into code." Quick
  start: `cargo build --release && ./scripts/first_run.sh`. Core
  commands list with one-line descriptions. Link to ARCHITECTURE.md
  (the manifesto) and docs/running.md (the runbook).
- `docs/running.md` — the runbook. Cold start from fresh clone to
  first crystallized module. Expected numbers per step ("After
  `calibrate-confidence --epochs 3` on the default corpus expect
  Pearson around X ± Y. If below Z, check …"). Diagnostic ladder
  for common failure modes.
- `docs/subsystems/crystallization.md` — the loop, each step, the
  files, the reusable utilities. Cross-reference source.
- `docs/subsystems/critic.md` — Head N, Head C, MetaCritic,
  target-network trick, NoveltyDetector wrapping. Cross-reference
  source.
- `docs/subsystems/compile.md` — IR, x86 emitter (and its scalar
  status), ELF, dlopen path. Cross-reference source.
- `docs/subsystems/symbolic.md` — DSL, BFS search, compression gate,
  symbolic_hint. Cross-reference source.
- `CLAUDE.md` at repo root — short standing instructions for AI
  contributors: commit style, test discipline, "read before editing",
  current architecture snapshot. Kept minimal.

**Modify:**
- `ARCHITECTURE.md` — append a "Current Status" section at the end
  listing which bets in Part V have evidence so far. Keep the rest
  untouched.

**Verify:** A fresh reader who has never seen this repo can go from
`git clone` to a first successful `clob ingest` run in under 20 minutes
using only README + runbook. Tested by re-reading as if blind.

### Phase J — Checkpointing + resume + crash safety

With manifests in place, checkpoints become tractable. This is the
phase that makes multi-day runs possible.

**Create:**
- `src/io/checkpoint.rs` — `pub struct Checkpoint` bundling model,
  energy critic, confidence head, meta critic, routers, episodic
  memory pointer, and a manifest. `save(path)` writes a directory, not
  a file: `checkpoint_00042/{model.clob, conf.bin, meta.bin,
  routers.bin, memory_ptr.bin, manifest.toml}`. `load(path, &mut model)`
  restores in place.
- `src/util/watchdog.rs` — `pub struct ResourceGuard { max_rss_mb,
  min_toks_per_sec }`. On a background thread (one-shot, not rayon),
  samples `/proc/self/status` VmRSS and the current tok/s estimate.
  Trips a `AtomicBool` on violation; the ingest loop checks each token.

**Modify:**
- `cmd_ingest`, `cmd_train_router`, `cmd_calibrate_confidence`: gain
  `--checkpoint-every <N>` (tokens) and `--resume-from <dir>`.
  Checkpoint directories are numbered sequentially; `--resume-from
  latest` picks the highest number.
- `cmd_ingest` gains `--max-rss-mb` and `--min-toks-per-sec`; when set,
  spawns `ResourceGuard`.

**Test:**
- Crash-safety fuzz for `src/memory/ring.rs`: start a write, inject a
  panic, reopen, assert the ring is consistent. New test in
  `tests/ring_crash.rs`.

**Verify:** Kill a long-running `ingest` with SIGKILL; restart with
`--resume-from latest`; confirm J/nat picks up at the previous window
idx (within ε) and mean NLL continues the trajectory.

### Phase K — Evaluation harness + per-head trajectories

The current `eval` subcommand is a one-shot. Real evaluation is an A/B
matrix with repeated seeds. Also: log the internal trajectories that
are currently stderr-only and lost.

**Create:**
- `src/eval/bench_suite.rs` — runs the matrix: `{adaptive on/off} ×
  {modules loaded/cleared} × {router trained/pristine} × seeds`. Emits
  `<out>/bench_suite.tsv` with columns `run_id, seed, adaptive,
  n_modules, router_state, mean_nll_holdout, j_per_nat, tokens_per_sec`.
  Cross-cell baseline comparison: the cell with lowest J/nat *and*
  lowest CE wins.
- `clob bench-suite --model X --eval-corpus Y --seeds 1,2,3 --out Z` —
  new subcommand wrapping the above.
- `src/eval/trajectory.rs` — per-training-step jsonl writer for head
  MSE, router update norms, Pearson against NLL. Consumed by `clob
  traj` diagnostic subcommand that renders a text plot.

**Modify:**
- `cmd_calibrate_confidence` writes a trajectory jsonl alongside the
  head bin when `--trajectory-out <path>` is set.
- `cmd_train_router` writes per-step advantage + update-norm to
  trajectory jsonl.

**Verify:** `bench-suite` on the synth model produces a coherent table
where `modules-loaded` doesn't regress against `modules-cleared` (it
should match today because modules are empty). Trajectories from
`calibrate-confidence` show the head's MSE monotonically decreasing per
step window.

**Shipped (commit `88827f4`):** `src/eval/bench_suite.rs` + the `clob
bench-suite` subcommand — the matrix over {adaptive}×{modules}×{router}×
{seeds} emitting `bench_suite.tsv` with a `summarize_by_axis` pivot.
**Deferred to Phase L:** `src/eval/trajectory.rs`, the `clob traj`
diagnostic subcommand, and the `--trajectory-out` modifications to
`calibrate-confidence` / `train-router`. These were cut from K for diff
hygiene and now ride with the real-data work (trajectories are only
diagnostic once a real corpus produces a non-flat NLL curve).

### Phase L — Real data: acquisition, tokenization, split enforcement

Up to now every verification has used the 1.5 KB smoke corpus. This
phase replaces it with pinned, reproducible artifacts.

**Create:**
- `scripts/acquire_corpus.sh` — downloads a pinned slice (initial
  candidate: Project Gutenberg's "A Brief History of Time" or a
  Wikipedia Wiki-40B train slice at a specific commit). Writes
  `data/corpus/train.txt` and `data/corpus/holdout.txt` with `.sha256`
  sidecar files. Idempotent — re-running re-verifies checksums, doesn't
  re-download.
- `scripts/tokenize_corpus.sh` — runs `clob train-tokenizer` with
  pinned merges count (4000), then encodes train + holdout into
  `data/corpus/train.tokens` and `data/corpus/holdout.tokens` (compact
  `u32` arrays). Cached on disk; ingest can skip the tokenize step.
- `data/corpus/README.md` — sources, licenses, provenance.
- `data/synthetic/diagnostic.txt` — hand-built corpus with *planted
  structural patterns*: repeated bit-shift rules, periodic n-grams,
  known high-entropy noise interleaved with high-structure sections.
  Accompanied by `data/synthetic/expected.md` documenting what the
  crystallization loop should find. This is the Phase E end-to-end test
  at the corpus level.
- `src/util/split_check.rs` — sentence-level hash check: no train
  sentence may appear in holdout. `tests/data_split.rs` enforces this
  in CI.
- `src/eval/trajectory.rs` + `clob traj` (**rolled over from Phase K**):
  per-training-step jsonl writer for head MSE, router update norms, and
  Pearson against NLL, rendered as a text plot. `calibrate-confidence`
  and `train-router` gain `--trajectory-out`. Now diagnostic because the
  real corpus produces a non-flat learning curve.

**Modify:**
- `cmd_ingest` accepts tokenized input when the path ends in `.tokens`,
  skipping tokenization entirely.
- Add a default `tokenizer.bin` to the repo (small enough; ~10-100 KB)
  so fresh clones don't require training.

**Verify:** `acquire_corpus.sh && tokenize_corpus.sh` twice on two
machines produces byte-identical `.tokens` files. `tests/data_split.rs`
detects an injected overlap between train and holdout.

### Phase M — Code quality: logging, error handling, CI, dead code

Until now, `eprintln!` was fine because nothing was long-running and
nothing was parsed programmatically. With Phase J's long runs and Phase
K's automated bench-suite, that changes.

**Create:**
- `src/util/log.rs` — a minimal leveled logger (info/warn/error/debug)
  with a `--log-format={plain,jsonl}` switch. No external dep needed; a
  thin wrapper over eprintln.
- `.github/workflows/ci.yml` — build + test + clippy on push. `cargo
  fmt --check` too. Small and cheap.

**Modify:**
- Replace every `eprintln!("[subsystem] ...")` with a logger call.
  Grep-and-pattern sweep across `src/`. This is mechanical but touches
  every file — do it as its own commit.
- Error handling consistency: add `thiserror` dep. Each module with a
  failure mode gets a `<Module>Error` enum. `main.rs` uses `anyhow` at
  the top level only.
- Delete dead code flagged by the Phase G–M work:
  - `Program::apply()` in `src/crystal/synth.rs` is unreached today;
    either wire into a symbolic-runtime path (Phase Q) or delete.
  - Any `pub fn` accessor added in Phase F for "future use" that still
    has no caller after this plan lands — delete it, re-add when
    actually used.

**Verify:** `cargo clippy -- -D warnings` is clean. CI passes on a
fresh push. Grep for `eprintln!` returns zero results in `src/`.

### Phase N — First-run rehearsal: scripts/first_run.sh

The culminating Tier 1 phase. Everything above exists so this script
can run.

**Create:**
- `scripts/first_run.sh` — opinionated end-to-end pipeline:
  ```
  set -euo pipefail
  : "${SEED:=1}"
  : "${EXPERIMENT_NAME:=first_run}"
  EXP_DIR="experiments/$(date +%Y-%m-%d)_${EXPERIMENT_NAME}"
  mkdir -p "$EXP_DIR"
  ./scripts/acquire_corpus.sh
  ./scripts/tokenize_corpus.sh
  clob synth       --seed "$SEED" --output "$EXP_DIR/seed.clob"
  clob calibrate   --seed "$SEED" --model "$EXP_DIR/seed.clob" ...
  clob calibrate-confidence --seed "$SEED" --meta-out ...
  clob train-router --seed "$SEED" --checkpoint-every 10000 ...
  clob ingest --seed "$SEED" --adaptive-compute ... --metrics-out "$EXP_DIR/m.jsonl"
  clob crystal --model ...
  clob bench-suite --out "$EXP_DIR/bench.tsv"
  echo "Experiment complete at $EXP_DIR"
  ```
- `scripts/smoke_test.sh` — same shape, but uses
  `tests/data/eval_corpus.txt` and tiny configs, finishes in < 60s.
  Used by CI as the end-to-end smoke signal.

**Verify:** `scripts/smoke_test.sh` on a fresh clone succeeds in under
60s and produces a non-empty `bench.tsv` in an `experiments/` directory.
`scripts/first_run.sh` with a moderate `--max-tokens` runs to
completion on the acquired corpus and produces J/nat numbers that
visibly differ from the synth-weight baseline.

### Tier 1 critical files

- **Create:** `src/io/writer.rs`, `src/util/seed.rs`, `src/util/sha.rs`,
  `src/util/watchdog.rs`, `src/util/log.rs`, `src/util/split_check.rs`,
  `src/config/manifest.rs`, `src/io/checkpoint.rs`,
  `src/eval/bench_suite.rs`, `src/eval/trajectory.rs`,
  `scripts/acquire_corpus.sh`, `scripts/tokenize_corpus.sh`,
  `scripts/first_run.sh`, `scripts/smoke_test.sh`,
  `scripts/new_experiment.sh`, `README.md`, `CLAUDE.md`,
  `docs/running.md`,
  `docs/subsystems/{crystallization,critic,compile,symbolic}.md`,
  `data/corpus/README.md`,
  `data/synthetic/{diagnostic.txt,expected.md}`,
  `experiments/README.md`, `.github/workflows/ci.yml`,
  `tests/data_split.rs`, `tests/ring_crash.rs`.
- **Modify:** `src/main.rs` (every subcommand: `--seed`, manifests,
  checkpoints), `src/io/synth.rs` (factor shared naming with writer),
  `src/io/loader.rs` (factor shared naming with writer), every
  `eprintln!` site, `Cargo.toml` (new deps: `toml`, `sha2`, `thiserror`,
  `anyhow`), `ARCHITECTURE.md` (status appendix).

### Tier 1 reused utilities (do not duplicate)

- `WeightAccum` + `format::write_header` (`src/io/synth.rs:10-64`,
  `src/io/format.rs:43`) — the save-path skeleton.
- `EpisodicMemory::stats()` and `mark_consumed()` — already idempotent
  and crash-tolerant in design; verify in Phase J test rather than
  rebuild.
- `MetricsWindow` and `JsonlWriter` — reused by `bench_suite.rs` and
  `trajectory.rs` as the logging substrate.
- `NoveltyDetector` — reused anywhere we want an adaptive z-score
  threshold (logs, probes, watchdogs).

### Tier 1 end-to-end verification gates

1. **Reproducibility gate** (post-G): `synth --seed 1` + `calibrate
   --seed 1` + `train-router --seed 1` each produce byte-identical
   outputs across two runs on the same commit.
2. **Manifest gate** (post-H): every artifact carries a TOML manifest;
   loading a mismatched manifest under `--strict` aborts with a clear
   diagnostic.
3. **Docs gate** (post-I): a reader new to the project can run the
   smoke test using only README + runbook.
4. **Resume gate** (post-J): SIGKILL mid-run + `--resume-from latest`
   continues the J/nat trajectory.
5. **Eval gate** (post-K): `bench-suite` produces a table where
   adaptive+meta outperforms adaptive-alone on real weights on a real
   corpus (the first verification that B+C+F actually earn their keep,
   not just "work").
6. **Data gate** (post-L): `acquire_corpus.sh` + `tokenize_corpus.sh`
   produce byte-identical artifacts across machines; split enforcement
   blocks a planted overlap.
7. **CI gate** (post-M): `cargo clippy -- -D warnings`, `cargo fmt
   --check`, and `scripts/smoke_test.sh` all pass on a fresh push.
8. **First-run gate** (post-N): `scripts/first_run.sh` on the acquired
   Phase L corpus produces an `experiments/<date>_first_run/` directory
   with a coherent `bench.tsv`. Phase B+C+F subsystems either show
   their earn-keep win or the regression is visible.

### Tier 1 effort estimate

~2 weeks of focused work. Phase G ~1 day, H ~1 day, I ~1 day, J ~2
days, K ~2 days, L ~1 day (corpus dependent), M ~1 day, N ~half day.
Durations collapse if phases with shared code paths run in parallel.

## Tier 2 — Previously deferred items, now in scope (Phases O–R)

These are the items the five-phase commits explicitly cut to keep diffs
honest. Mechanical engineering with well-defined success criteria.

### Phase O — AVX2 (and AVX-512) vectorized ternary emitter

The single largest constant factor on the table. `src/compile/x86.rs`
today emits scalar `addss`/`subss` per nonzero trit; an AVX2 emitter
using packed sign ops and `vpaddd`/`vpsubd` over 8 f32 lanes at a time
delivers an expected 4–8× throughput gain on the T490. AVX-512 (where
available) widens to 16 lanes.

**Create:**
- `src/compile/x86_avx2.rs` — emitter that groups ternary trits into
  AVX2-sized chunks. For each row: zero an 8-wide accumulator, for each
  batch of 8 input lanes load into `xmm`, apply a sign mask built from
  the row's trits (`vpsignd` or equivalent), accumulate.
- `src/compile/x86_avx512.rs` — same, 16 lanes. Runtime gated by CPUID.
- `src/compile/dispatch.rs` — selects the emitter based on CPU
  capabilities at module-compile time. Default scalar for
  correctness-first debugging via env flag.

**Modify:**
- `src/compile/jit.rs::compile_and_load` — consults the dispatcher.
- `src/simd/mod.rs` — already has runtime CPUID; reuse the existing
  feature flags rather than re-detect.
- Benchmark suite (Phase K's `bench-suite`) gains
  `kernel={scalar,avx2,avx512}` as an ablation dimension.

**Verify:** A compiled module runs identical outputs (bit-exact for
ternary since there's no floating-point reordering) across scalar /
AVX2 / AVX-512 emitters on the same inputs. Bench-suite shows the
expected throughput ladder. ARCHITECTURE.md's "32 ops/clock" claim
becomes measurable.

### Phase P — Head S: symbolic-suitability probe

Phase E ships `symbolic_hint` as a post-hoc check against every
crystallized ternary matrix. Phase P completes the plan's original
design: a learned probe that predicts, *from the cluster's hidden state
distribution*, whether symbolic synthesis will succeed before the
ternary matrix is even built.

**Create:**
- `src/nn/symbolic_head.rs` — `pub struct SymbolicHead` — a third
  linear probe over hidden state, identical machinery to
  ConfidenceHead. Target label: 1 if the cluster's crystallized matrix
  would admit a program under `try_synthesize`, 0 otherwise. Trained
  offline over historical clusters.

**Modify:**
- `src/crystal/engine.rs::cycle` — after clustering, for each cluster
  run Head S on the centroid. Record the prediction into the cluster
  record. When Head S fires above a threshold, budget extra synthesis
  depth (up to 6 ops); when it misfires, skip synthesis entirely —
  saving the BFS cost on clusters that won't reduce.
- `src/main.rs` — `calibrate-symbolic` subcommand, mirrors
  `calibrate-confidence`. Streams historical episodes, clusters on the
  fly, records (cluster hidden centroid, synth_success) pairs, trains
  Head S.

**Verify:** On the synthetic diagnostic corpus (Phase L) with planted
structural patterns, Head S's ROC against ground truth reaches AUC >
0.8. Ablation: without Head S, BFS runs on every cluster; with it, ~80%
of unproductive BFS calls are skipped.

### Phase Q — Symbolic execution at inference

Today `symbolic_hint` is metadata. Complete the loop by actually
running programs as the hot path for symbolic modules:
`Program::apply` is already implemented — wire it into
`CrystalModule::apply`.

**Create:**
- `src/crystal/module.rs::ModuleKind` enum with
  `Ternary(TernaryMatrix)` and `Compiled { program: Program, fallback:
  TernaryMatrix }`. `apply()` branches. The fallback stays so symbolic
  failure falls through gracefully.
- **Storage optimization:** the Compiled variant serializes only the
  program (tens of bytes), reconstructing the fallback matrix on load
  via `program.materialize_trits`. `#[serde(skip)]` on the runtime
  fallback; hydrate lazily. This is the MDL storage win Phase E always
  promised.

**Modify:**
- `src/crystal/store.rs` — load path hydrates Compiled variants.
- `src/crystal/crystallize.rs` — when a hint is found with acceptable
  compression, construct ModuleKind::Compiled directly instead of
  wrapping Ternary.

**Verify:** A module saved as Compiled and reloaded produces identical
`apply()` outputs to its Ternary counterpart. Disk footprint for a
planted-shift module drops from ~4 KB (d=128) to ~30 bytes.

### Phase R — Router re-ternarization

Phase F (2/2) leaves routers as dense f32 after training. The plan's
original "once converged, re-ternarize" step makes the router a
first-class citizen of the same compression regime the experts live
under — the conceptual through-line of the whole architecture.

**Create:**
- `src/model/router_ternary.rs` — wraps `ExpertRouter` in a
  `LatentWeights` (reuse `src/learn/grad.rs:11-75`). Training flow:
  (a) train as f32 via REINFORCE (existing path); (b) mirror to latent;
  (c) re-ternarize via `LatentWeights::re_ternarize`; (d) at inference,
  route against the ternary version, maintaining the same logit
  computation but with packed weights.

**Modify:**
- `src/main.rs::cmd_train_router` — gains `--ternarize-after <steps>`.
  Post-training, calls `LatentWeights::re_ternarize()`, writes the
  ternary version to the routers side-file, verifies route decisions
  match on a held-out batch within ε.

**Verify:** Post-ternarization CE delta on the held-out slice stays
within 2% of pre-ternarization. Router weight file drops from ~2 KB
per layer to ~300 bytes.

### Tier 2 effort estimate

~3–4 weeks. Phase O ~1 week (vectorized codegen is fiddly). Phase P
~3–4 days. Phase Q ~3–4 days. Phase R ~3 days.

## Tier 3 — Architectural expansions (Phases S–V)

These require design work, not just engineering. Each extends the
architecture into territory the current commits could not cover without
breaking the focus on a specific phase. All are grounded in existing
primitives.

### Phase S — Mycelium: federated crystallization across peers

`src/net/exchange.rs` ships a wire protocol for module exchange and a
simple acceptance heuristic. Phase S completes the mycelium story:
multiple `clob` instances observe different input distributions,
crystallize specialized modules, and barter them via the existing
protocol into a distributed library.

**Core mechanisms:**
- **Domain specialization:** each node tracks per-module activation
  rates. Modules with high activation rates locally but unknown to
  peers become "offerable" — they capture local expertise.
- **Counterpart acceptance:** when a peer offers a module whose domain
  signature complements an uncovered region of this node's input
  distribution (measured via rolling hidden-state coverage), accept it.
  Use `NoveltyDetector` over the domain-similarity distribution.
- **Provenance chains:** every accepted module records its peer-origin
  chain. A module that propagates across N peers is evidence of
  generalizable structure, not a local artifact.

**Create:**
- `src/net/federation.rs` — the orchestration layer: discovery,
  offering schedule, acceptance policy.
- `src/net/coverage.rs` — maintains a sketch (count-min or similar) of
  hidden-state regions this node has covered; used by the acceptance
  heuristic.
- `clob federate --peers host1,host2` subcommand.

**Verify:** A three-node testbed on localhost with three disjoint
corpora (technical, literary, conversational). After N rounds of
federation, each node's module library reflects all three domains with
differential weighting by local input distribution.

### Phase T — Dual-process cognition (System 1 / System 2)

Adaptive compute today is "iterate k extra times when Head C is
uncertain." The full idea: two distinct inference modes.

**System 1 — fast, ternary matmul, single pass, no reflection.** The
existing `decode_step`.

**System 2 — slow, involves (a) adaptive iteration (existing), (b)
cross-module composition (new), (c) symbolic program execution with
branching (new), (d) counterfactual forward passes against held-out
module ensembles.**

Head C gates the switch. Low uncertainty → System 1. High uncertainty →
System 2 with a proportional compute budget.

**Create:**
- `src/model/cognition.rs` — dispatches `decode_step` among multiple
  modes based on uncertainty + a budget.
- `Mode::System2` composes sub-components:
  - `compose_modules(hidden, modules, depth)` — runs top-K modules in
    sequence, selecting at each step based on intermediate state.
  - `counterfactual_ensemble(hidden, held_out)` — samples K runs with
    different module subsets, aggregates by lowest cross-entropy.

**Verify:** On tokens flagged high-uncertainty, System 2 achieves lower
NLL than System 1 — the test is not whether it's always better (it
isn't; it's more expensive) but whether the gating spends the extra
compute where it pays off. Bench-suite cell: `{System1_only,
System2_on_uncertain, System2_always}`.

### Phase U — Module composition algebra (abstraction growth)

Modules today are leaves. Phase U lets frequently-co-activated pairs
become a single *composed* module, and that composition becomes
crystallizable in turn. The mechanism for the system to grow
abstractions rather than just experts.

**Create:**
- `src/crystal/composition.rs` — tracks co-activation counts per-token
  across modules. When pair (A, B) fires together with high frequency
  AND their composed output consistently reduces cross-entropy, emit a
  compose candidate.
- `Program::Composed(Box<Program>, Box<Program>)` variant — not a
  matrix product but a sequential application, exposing the hierarchy.

**Verify:** After running on the Phase L diagnostic corpus, modules
that capture "shift then negate" patterns emerge from the composition
of a Shift module and a Negate module observed to co-activate — not
from re-crystallizing the whole pattern from scratch.

### Phase V — DSL expansion: branches, nonlinearities, memory

Phase E's DSL is pure linear composition. Real symbolic structure often
needs conditionals and nonlinearities. Phase V expands the DSL without
sacrificing the ternary-reducibility trick.

**Primitives to add:**
- `Sign` — `output[i] = sign(input[i])` via a two-ternary-matrix
  decomposition (ternary fallback). Breaks linearity but stays
  ternary-realizable.
- `Where(cond, A, B)` — conditional composition; `cond` is a ternary
  mask derived from another program.
- `Accumulate` — running sum across a window. Implementable as a
  lower-triangular ternary matrix.
- `Delay(k)` — last-k-steps memory. Requires the runtime to carry
  state; introduces statefulness to Programs, parallel to the SSM's
  recurrent state.

**Verify:** Planted counter-incrementing pattern in diagnostic corpus
(e.g., "position i in the sequence is +1 from position i-1") is
discovered by the expanded DSL under depth 6.

### Tier 3 effort estimate

~6–8 weeks. Phase S ~2 weeks. Phase T ~2 weeks. Phase U ~1.5 weeks.
Phase V ~1.5 weeks.

## Tier 4 — Frontier (Phases W–CC): the ultimate architecture

Speculative and deliberately bold. Each extends the architecture into
territory with no current code path. All are grounded in existing
primitives — none require abandoning the ternary / crystallization
thesis — but each adds a qualitatively new capability. Flagged as
exploratory; each warrants its own design review before
implementation.

### Phase W — Embodied perception: beyond text tokens

Right now `clob` reads text. The T490 has a camera, a microphone, a
screen, a keyboard. Phase W makes the kernel *perceive its own
machine*.

**Create:**
- `src/perceive/screen.rs` — framebuffer capture, quantized to
  patch-tokens via a fixed VQ codebook (itself crystallizable).
- `src/perceive/audio.rs` — ALSA capture, spectrogram patches tokenized
  similarly.
- `src/perceive/input.rs` — keyboard + mouse event stream as tokenized
  actions.
- A unified `Percept::Multimodal` that carries the modality tag so the
  crystallization loop can learn cross-modal correspondences.

**Why ultimate:** a language-only intelligence is a brain in a jar. A
kernel that sees what its human sees — by reading the framebuffer —
has grounded its compression against the same reality we inhabit. The
novelty is not the modalities (everyone has these) but the
ternary-constrained compression applied uniformly across them. A
cross-modal crystallized module that fires on the same abstract pattern
whether presented as text, pixels, or audio is *the* signal that
compression is finding actual structure, not modality-specific
artifacts.

### Phase X — Actor mode: closed-loop action and consequence

Perception without action is observation. Phase X gives the kernel a
way to propose actions — keystrokes, mouse events, shell commands —
then observe the consequences as new percepts, crystallizing the
action→outcome mapping.

**Create:**
- `src/actor/mod.rs` — an action head: given hidden state + task token,
  emit an action token. Trained via policy gradient (reuse the REINFORCE
  machinery from Phase F's router training, abstracted into a reusable
  utility).
- `src/actor/safe_shell.rs` — a sandboxed action executor: only runs
  shell commands that match an allowlist regex and log every action. No
  destructive ops without explicit user ack.
- Episode schema extension: `(prev_hidden, action, new_hidden, reward)`
  tuples that the crystallization engine consumes alongside prediction
  episodes.

**Why ultimate:** the crystallization loop's compression target expands
from "predict next token" to "predict environment response to this
action." This is the minimal loop for agency: observe, act, observe,
compress. It's also the Achilles' heel — actor mode is where safety
matters most; the sandbox design is doing real work.

### Phase Y — Self-modifying compiler

The `src/compile/` pipeline emits x86. Phase Y closes the ouroboros:
the crystallization engine crystallizes patterns in *compile-time*
profiles (what instruction sequences frequently co-occur, what register
patterns recur) and emits a *better compiler* as a new module. Not a
theoretical "fixed point of self-improvement" but a measurable one:
after N cycles, the compiler emits tighter code than its initial
version.

**Create:**
- `src/compile/profile.rs` — instruments the emitter, records a
  byte-level histogram of emitted instructions per module.
- `src/crystal/compiler_module.rs` — a specialized crystallization path
  whose input episodes are compile profiles, whose output is a new
  `x86` emitter module (emitted as a `Compiled` CrystalModule that the
  JIT loads in place of the default emitter for matching IR patterns).

**Why ultimate:** this is the literal fixed-point argument made
concrete and testable. If it works, the architecture improves its own
machinery in the same way it improves its predictions — by
crystallizing structure. If it doesn't, we learn exactly where the
argument breaks.

### Phase Z — Introspection and legibility layer

A kernel that can answer "why did you activate module #42 on that
token?" is qualitatively different from one that cannot. Phase Z builds
a reflection layer that can describe its own state.

**Create:**
- `src/introspect/mod.rs` — a query engine that takes a natural-language
  question about the kernel's state, parses it into a probe plan (which
  modules fired when, what the critic said, what the episodic memory
  contains), and emits a response grounded in actual recent activity.
- `clob ask "..."` — a subcommand that runs the query engine.
- A memoir append log: every non-trivial decision the kernel makes
  (module crystallized, route changed, head trained) gets a one-line
  entry in `MEMOIR.md`, which the kernel can itself read.

**Why ultimate:** the hardest critique of any neural system is "it
works but we don't know why." Legibility isn't ornament — it's a
precondition for trusting the kernel to act in Phase X's actor mode,
and for letting Phase Y's self-modifying compiler mutate its own
innards. The reflection layer is the safety rail that scales with the
kernel's capability.

### Phase AA — Counterfactual imagination / dreaming

Human learning consolidates during sleep via hippocampal replay — not
just of what happened, but of variants, counterfactuals, near-misses.
Phase AA implements the analog: during idle periods, run the core
forward on *synthetic* inputs sampled from the episodic buffer, varying
one dimension at a time, crystallizing any pattern that emerges from
the variations.

**Create:**
- `src/crystal/imagine.rs` — idle-scheduler job that wakes during low
  utilization, samples episode clusters, perturbs hidden states along
  principal axes, runs the model forward, records synthetic episodes
  as if real.
- A distinction between "observed" and "imagined" episodes, to avoid
  feedback-loop contamination where the kernel learns to predict its
  own hallucinations.

**Why ultimate:** the crystallization loop's bottleneck is novelty per
unit time. Imagination *creates* novelty from structure already
present. The existing system is episodic, passive, and bandwidth-limited
by the outside world's rate of interesting events. Imagination breaks
that ceiling — carefully.

### Phase BB — Hardware-aware autotuning

The T490 is one machine. Every deployment will be different. Phase BB
makes the kernel profile its hardware at boot, crystallize optimal
kernel shapes (AVX2 vs AVX-512 thresholds, cache block sizes, thread
assignments) for *this specific machine*, and treat those shapes as
normal crystallizable modules.

**Why ultimate:** "efficient on a T490" is the starting point.
"Autotunes to every machine it lands on" is the mycelium story with
teeth.

### Phase CC — Memoir: the architecture documents itself

Every commit, every crystallized module, every benchmark number, every
experiment directory — append-only into `MEMOIR.md`. The kernel itself
can read it. This is not documentation for humans (though humans can
read it); it's the kernel's own long-term memory of its own
development, usable as context for future crystallization cycles.

**Why ultimate:** an architecture that remembers its own history has a
substrate for meta-reflection on what has and hasn't worked. The only
honest way to evaluate the five-phase plan's bets across the lifetime
of the project: not via one-shot benchmarks but via the kernel's own
reading of its memoir.

### Tier 4 effort estimate

Exploratory. Each phase is a 2–6 week project on its own. Pick one at
a time, land it fully, evaluate before committing to the next. The
tier doesn't need to ship in any particular order except that Phase Z
(introspection) ideally lands before Phase X (actor mode) so the
actor's decisions are legible from day one.

## Dependency graph across all tiers

```
Tier 1 (G..N) — readiness
    │
    ├── Phase O — AVX2/AVX-512 emitter
    │       └── benefits all downstream work (constant factor)
    │
    ├── Phase P — Head S probe
    │       └── Phase Q — symbolic at inference
    │               └── Phase V — DSL expansion
    │
    ├── Phase R — router re-ternarization
    │       └── closes Phase F's conceptual loop
    │
    ├── Phase S — federation / mycelium
    │       └── prerequisite for any cross-machine work
    │
    ├── Phase T — dual-process cognition
    │       ├── requires Phase U (module composition) for full effect
    │       └── requires Phase Q (symbolic runtime)
    │
    ├── Phase U — composition algebra
    │       └── enables Phase Y (compiler self-modification)
    │
    ├── Phase W — embodied perception
    │       └── prerequisite for Phase X (actor mode)
    │
    ├── Phase X — actor mode
    │       └── hard-depends on Phase Z (introspection) for safety
    │
    ├── Phase Y — self-modifying compiler
    │       └── depends on Phase U, Phase O's profile infrastructure
    │
    ├── Phase Z — introspection layer
    │       └── benefits every downstream phase
    │
    ├── Phase AA — counterfactual imagination
    │       └── depends on Phase Q (symbolic runtime) for structured variation
    │
    ├── Phase BB — hardware autotuning
    │       └── depends on Phase O + Phase Y
    │
    └── Phase CC — memoir / self-documentation
            └── lightweight, land early and continuously
```

## What this plan is, honestly

Tier 1 is a 2-week engineering sprint before the first real run. It's
load-bearing and should happen in order.

Tier 2 is the other 50% of Phases E and F that the original commits
cut for diff hygiene. Necessary to complete the architectural thesis.

Tier 3 is where `clob` stops being a kernel and becomes a *cognitive
architecture* — composing modules, running two processes, expanding
the DSL, federating across machines. Each phase here is individually
ambitious but tractable.

Tier 4 is where the project reaches for "ultimate" in the sense the
user meant it: not mere performance, but *architectural completeness*.
Embodiment, action, self-modification, dreaming, introspection, self-
documentation. No guarantee any one of these works out; all of them
together is the shape of an architecture that could genuinely be a
fixed point of self-improvement rather than one that *describes itself*
as one.

The ordering is deliberate. Shortcuts across tiers produce
unmaintainable systems that look ambitious and behave brittle.

**Land Tier 1 first. Everything follows.**

---

# Session-handoff footer

When ending a work session that makes material progress against this
plan:

1. Commit all code changes with the existing commit-message style (no
   Claude attribution).
2. Update `State as of this document's creation` at the top — move
   completed phases from "not yet" to "shipped", add their commit
   hashes.
3. Update the memory file at
   `/home/suds/.claude/projects/-home-suds-Projects-clob/memory/project_clob.md`
   with the new state.
4. If any phase's scope expanded or contracted, edit the phase's entry
   here rather than leaving it stale.
5. If blocked on something unknown at plan time, add a **Known
   blockers** subsection with the specific question.

When starting a new work session:

1. Read this file top-to-bottom.
2. Read `ARCHITECTURE.md` for philosophical context (skimmable after
   first read).
3. Run the quick-start verification commands.
4. Pick the next phase by dependency order — Tier 1 first, G before H,
   etc.
5. Review the last 3 commits to catch any context not yet in this doc.
