# CLAUDE.md — standing instructions for AI contributors

## Architecture snapshot

`clob` is a ternary-recurrent AI kernel. The design thesis and
five falsifiable bets live in `ARCHITECTURE.md`. The active forward
roadmap (with per-phase deliverables, verification gates, and
session-resume info) lives in `PLAN.md`. Read both before making
material changes.

## Commit conventions

- **No Claude attribution.** No `Co-Authored-By: Claude`, no "🤖
  Generated with..." footers, no signatures. Commits read as if the
  human maintainer wrote them.
- Use a short imperative subject line (one line), then a blank line,
  then a detailed body. Existing history is the style reference.
- One logical change per commit. When a phase has two independent
  halves, split them (see `f7388a9` + `b8c23fd`).

## Code conventions

- Every new `serde` field gets `#[serde(default)]` so old on-disk
  artifacts still load. Phase E's `symbolic_hint` set the pattern.
- Doctests: pseudo-code in module docs goes in `text` fences, not
  raw indented blocks. Raw blocks are compiled as Rust and fail.
- New subcommands: add to `Commands` enum → match arm in `main()` →
  `cmd_<name>` function. Never reach into `main()` from deep code;
  the subcommand orchestrates, the modules execute.
- Seeds: no `thread_rng()` or `from_entropy()` in production code
  paths. Use `SeedTree::child("purpose")` from a top-level `--seed`.
- Manifests: every write-subcommand emits a sidecar via
  `RunManifest::new(...).with_seed(...).with_input(...).save_sidecar`.

## Test discipline

- Run `cargo test --release` before every commit; 50+ tests are
  expected to pass.
- New code ships with its own tests in the same commit, not a
  "tests coming later" commit.
- End-to-end smoke test for any phase that touches the runtime
  pipeline. See how Phase G verified byte-identical artifacts.

## Read before editing

- `ARCHITECTURE.md`, `PLAN.md`, and the auto-loaded memory file
  (`/home/suds/.claude/projects/-home-suds-Projects-clob/memory/project_clob.md`)
  are the orientation triad.
- Before touching any file, read it at least once. The Write/Edit
  tools refuse to write a file not yet read in the session.

## Don't

- Don't push to origin without explicit instruction.
- Don't skip pre-commit hooks (`--no-verify`) unless the user asks.
- Don't rename `SeedTree::child` labels casually — they're part of
  the reproducibility contract.
- Don't introduce `eprintln!` in new code; Phase M will sweep
  those into a leveled logger.
- Don't change the on-disk format without bumping `format::VERSION`
  and preserving the ability to load v1 files.

## When in doubt

Describe what you're about to do in one sentence, then do it. If
it's risky or reversible only by running a command the user didn't
authorize (push, force-push, reset --hard, rm -rf on shared state),
pause and ask.
