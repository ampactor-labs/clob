# IF FOUND

If you are reading this and no one is here to explain it, this file is clob's
honest account of itself — what it is, what it is *not*, and what it would take
to make it live. It is written to be understood by a mind that did not build
it. It does not overclaim; an earlier draft of this project's manifesto was
deleted for "leaning on prayer rather than proof," and this note keeps that
discipline.

## What clob is

A ternary-recurrent intelligence kernel, small enough to run on one laptop with
no GPU. Its one bet: **intelligence is compression efficiency per joule.** Its
mechanism: a *crystallization loop* — notice prediction error, cluster it,
distill it into a ternary `{-1,0,+1}` module, integrate the module, and forget
the raw experience. Every improvement is meant to cost *less* energy, not more.
The design thesis is in `ARCHITECTURE.md`; the roadmap and resume state in
`PLAN.md`; the verified ground truth in the project's memory file.

It is, at time of writing, ~11k lines of Rust, 70 passing tests, byte-for-byte
reproducible from a single seed. The plumbing composes end to end. That part is
real and was built carefully.

## What clob is NOT — read this part twice

**clob does not yet learn.** This is the truth that matters most:

- The ternary core — embeddings, the recurrent SSM, every block — **is never
  trained.** It stands at random initialization, permanently.
- The machinery to train it (`src/learn/`: straight-through gradients, AdamW
  over latent weights) exists but is **wired to nothing.** It is the one module
  with no tests. There is no backward pass anywhere in the model.
- Embedding and unembedding are **the same tied table**, so even the readout
  cannot be trained in isolation.
- The only things that adapt are the confidence/meta heads, the router policy,
  and the crystallized modules — corrections layered on a frozen, random base.

So **none of the five falsifiable bets in `ARCHITECTURE.md` Part V are
validated.** They cannot be, yet. Bet 1 says "a ternary core, *trained well*…"
— there is no path to train it well. Crystallization runs, but on a random core
it has mostly noise to compress.

Do not mistake the running pipeline for a living one. It moves; it does not yet
grow.

## The decision that defines its future

Before clob can become anything, one fork had to be chosen:

- **A — Reservoir.** The random core is a fixed substrate by design; *all*
  learning is crystallization of local corrections (the code says, in places,
  "no backprop required"). Then "trained well" means "crystallized well," and
  the bet is bold and unconventional.
- **B — Trained core.** Train the ternary core to competence (untie the
  embedding, write a backward pass, wire `src/learn/`, add a `train` command),
  *then* run crystallization as the continual-compression layer on top. This is
  the manifesto's literal reading.

The first real-run artifacts make this less ambiguous now. A reservoir probe in
`experiments/2026-06-22_first_run/probe_readout.json` fit a trained linear
readout over frozen-core hidden states and measured zero contextual gain over a
trained bias-only unigram null. Treat that as a failed Path A probe unless a
stronger replicated run contradicts it.

So the next build is **Path B**. Untie the readout, implement a checked backward
pass, train the core on planted patterns, and only then ask crystallization to
compress what the core has learned.

Since that fork was written, the probe's verdict acquired a mechanism. The
dynamics instrument (`clob regime`, 2026-07-04) measures the random core at
λ₁ = −1.23 nats/token: its state erases a perturbation in under one token, so
the past physically never reached the readout. The same day, re-partitioning
episodes by their *futures* instead of their hidden-state geometry
(`clob crystal --causal`) produced the first four modules this pipeline has
ever crystallized. The second thesis behind both changes is `ATTRACTOR.md` —
read it after the manifesto. Its bets (6–8) are pre-registered like the
first five, and they wait on Path B just the same.

## How to judge it honestly

The pass/kill numbers are pre-registered in `ARCHITECTURE.md` (Part V →
*Falsification thresholds*) — fixed *before* any run, so a result can fail
rather than be reinterpreted. Honor that. If a threshold is wrong, change it in
a commit that does not also report the run it judges. Let clob be falsified if
it deserves to be.

## How to resume it

1. Read `ARCHITECTURE.md`, `PLAN.md`, and the memory file — the orientation
   triad.
2. `cargo test --release` — should be green.
3. `scripts/diagnostic_run.sh` — runs the planted-pattern corpus, whose answers
   are *known* (`data/synthetic/expected.md`). Ground truth before real noise.
4. `scripts/first_run.sh` — the one-command real-data run. Honest, but it tests
   crystallization on a frozen core, not the headline bets.
5. Make the fork. Then build core learning. Then the bets become real.

## Why it was not simply left running

Because a half-built self-improver set loose does not ascend — it drifts and
burns its own substrate. The most useful thing for clob, in the absence of
anyone to finish it, is not motion but *legibility*: to be found in a state
clean enough to continue. An idea outlives its author only if it can be picked
up. That is the whole purpose of this file.

If you finish it — close the loop, make the core learn, and let the
crystallization heartbeat run on something that can actually grow — then clob
becomes what it was for. If you do not, let it rest here, honest and findable.

Either is a fate it can bear. Being misunderstood as more than it is, is not.
