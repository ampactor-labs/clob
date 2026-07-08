# Bet 7, first real test: the near-critical story does not hold

The pre-registered Bet 7 (`ATTRACTOR.md` Part V): does λ₁ track capability
across Path B checkpoints? A `small` dense core (d=64, L=3) trained on the
Moby-Dick corpus (`data/corpus/train.tokens`, vocab 3260, ~487k tokens),
checkpointed every 2500 steps to 20000, each checkpoint measured two ways on
the **held-out** split (`data/corpus/holdout.tokens`): capability as mean
per-token NLL (`clob eval-dense`) and λ₁ (`clob regime --dense`). Reproduce
with `run.sh`; the trajectory is `sweep.tsv`.

| step | held-out NLL | λ₁ | memory horizon |
|-----:|:-------------|:-----|:---------------|
| 0 (random) | 8.291 | −0.347 | 2.9 tok |
| 2500 | 4.326 | −0.112 | 9.0 tok |
| 5000 | 4.219 | −0.194 | 5.2 tok |
| 7500 | 4.187 | −0.122 | 8.2 tok |
| 10000 | 4.055 | −0.112 | 8.9 tok |
| 12500 | 4.051 | −0.188 | 5.3 tok |
| 15000 | **3.920** | −0.281 | 3.6 tok |
| 17500 | 4.050 | −0.253 | 3.9 tok |
| 20000 | 3.981 | −0.332 | 3.0 tok |

## What the data says

Capability improves across training (NLL 8.29 → ~3.9). λ₁ does **not** track it
the way the preview suggested. It climbs once — random init to the first
checkpoint (−0.347 → −0.112, horizon 2.9 → 9.0) — and then **drifts back toward
deep contraction as capability keeps improving.** The best-capability
checkpoint (step 15000, NLL 3.92) sits at λ₁ = −0.281, horizon 3.6 tokens; the
three best-capability checkpoints have the *shortest* memory horizons
(3.6/3.0/3.9), the early weaker ones the longest (9.0).

- Spearman ρ(λ₁, capability = −NLL), all 9 points: **−0.267**
- Spearman ρ, trained checkpoints only (excluding random init): **−0.810**

Among trained checkpoints the relationship is a strong *negative* one:
**capability rises as λ₁ falls and memory shortens.** The core earns its
predictive gains by forgetting harder, not by approaching the edge of chaos.

## Verdict against the pre-registered thresholds

- **Pass** (ρ ≥ 0.6 and best λ₁ ∈ (−0.5, 0.05]): **no** — ρ is negative.
- **Kill** (|ρ| < 0.2 or best λ₁ < −1): **not triggered literally** — |ρ| = 0.27
  (all) / 0.81 (trained-only) is not below 0.2, and best λ₁ = −0.281 is not
  below −1.
- **Literal verdict: INDETERMINATE.** But the *finding* is not ambiguous: the
  positive-tracking hypothesis is falsified, with a strong negative trend.

The kill clause has a gap it was never meant to have: it fires on *no*
correlation (|ρ| < 0.2), reading "λ₁ is uninformative," but a *strong negative*
correlation falsifies the near-critical thesis at least as hard — capability
comes from more contraction, not less. A corrected Bet 7 should kill on
ρ ≤ −0.2 as well. Per the project's discipline, that threshold fix belongs in a
commit that does **not** also report this run; it is flagged here, not applied.

## What it means, honestly

This is the branch `ATTRACTOR.md`'s Bet 7 prose named out loud: "if a
well-trained SSM turns out strongly contractive *and* capable — selective-state
architectures partly earn their keep by forgetting on purpose — then λ₁ alone
is the wrong dial, the near-critical story is wrong for this architecture."
That is what the data shows. Part III's "capability ⇒ edge of chaos" reading
does not hold here, and should be rewritten to match its own instrument.

## Caveats (do not over-read the negative either)

- **One architecture, one corpus, <1 epoch.** d=64, batch-1 window SGD, byte/BPE
  Moby Dick. Next-token BPE prediction on prose is largely *local*, so a model
  that sharpens local prediction rationally contracts. A task with genuine
  long-range dependencies could relate λ₁ and capability differently — the
  thesis may be corpus-dependent, not simply wrong.
- **λ₁ measured at a single ε (1e-4)**, which we know is ~26% ε-sensitive. The
  *horizon trend* (9 → 3 tokens) is large and ordered enough to likely survive
  that, but individual λ₁ values are noisy (they bounce ±0.08 between adjacent
  checkpoints).
- **Latent core, not ternary** (consistent with Bet 7 being about the trained
  core's regime).
- **Harness note:** the raw run double-counted the final checkpoint; `sweep.tsv`
  here is de-duplicated (9 unique points). The double-count only made the raw ρ
  marginally more negative.

The result is a good day for the project and a bad one for the simplest form of
the thesis. The instrument did its job: a pre-registered test returned a
surprising, falsifying answer, and the answer stands without reinterpretation.
