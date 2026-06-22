# Design: the optimal next run — does the ternary core actually learn?

**Status:** design / pre-registered spec (not yet run) · **Phase:** Path B, step 2

The reservoir probe resolved the fork: the frozen random core carries no
decodable context, so the core must be **trained** (`IF_FOUND.md`,
`docs/experiments/2026-06-22-reservoir-probe.md`). Step one — an untied
trainable output head — shipped. This is the design for the run that tests the
real question: **can the ternary recurrent core be trained to learn at all?**

The design follows the probe's hard-won lesson: a result is only as trustworthy
as the thing that produced it. A wrong backward pass "trains" to garbage just as
convincingly as a right one, so correctness is *gated*, not assumed, and the
first run is the cheapest one that can fail against a **known answer**.

## Stage 0 — the gate: a gradient check (no run until this passes)

Before any training run, verify the backward pass by finite differences on a
tiny model: for a handful of parameters in *each* layer type, the analytic
gradient must match `(L(w+ε) − L(w−ε)) / 2ε` to ~1e-4 relative.

- Layer types to cover: ternary linear (via the straight-through estimator),
  RMSNorm, the GLU channel mixer, and the selective-SSM recurrence (the hard
  one — gradients must flow correctly through `h_t = Ā·h_{t-1} + B̄·x_t`).
- This is a **unit test**, not a compute run, and it is the single most
  important deliverable. Until it is green, no NLL curve means anything.

Rationale: the probe's first verdict came from a *diverging optimizer* that
happened to be right for the wrong reason. The analog here is a plausible-but-
wrong gradient. The gradient check is the only thing that rules it out.

## Stage 1 — the optimal next run: small core on the planted-pattern corpus

Train a **small** core (`KernelConfig::small`: d=128, L=4, V=1024) on
`data/synthetic/diagnostic.txt`, whose structure is planted and documented in
`data/synthetic/expected.md`:

- **Section 1** — the same sentence repeated verbatim. After the first
  occurrence this is *deterministic*; a working core must drive next-token NLL
  to ≈ 0 here.
- **Section 3** — three period-2 alternations. Predictable after one period;
  NLL must collapse to ≈ the planted (near-zero) entropy.
- Noise sections — NLL should stay near the noise entropy (the core must *not*
  hallucinate structure where there is none).

Why this is the optimal first run: it is the smallest, fastest experiment that
distinguishes "the trainer works" from "the trainer is broken," against an
answer we already know. A core that cannot learn to predict a sentence it has
seen ten times is broken — and that costs minutes to discover here versus a
multi-day real-corpus run to discover later. "Ground truth before real noise."

**Training setup:**
- Untie the embedding; train embed + core + unembed jointly. (The probe already
  established the readout must be untied; reuse that machinery.)
- Straight-through estimator over the existing `LatentWeights` f32 shadow
  weights, `AdamW` from `src/learn/optimizer.rs`, periodic `re_ternarize`.
- **Truncated BPTT**, window T = 64 tokens (detach SSM state at window
  boundaries) — bounds memory and is the standard recurrent-training choice.
- Gradient clipping by global norm; warmup then cosine/step lr decay.
- `SeedTree` for every RNG; a run manifest; byte-reproducible from `--seed`.

**Pre-registered pass / kill (locked before the run):**
- **PASS** — next-token NLL on the deterministic sections falls below ~0.2 nats
  (the core demonstrably learns known structure), and overall NLL falls well
  below the diagnostic corpus's unigram marginal.
- **KILL** — after the compute budget the core cannot beat the unigram marginal
  on the structured sections. That means training does not work; fix the
  trainer, do **not** spend compute on the real corpus.

## Stage 2 — follow-on (not this run): real corpus + Bet 1

Only after Stage 1 passes. Train the ternary core **and a matched f32 core**
(same architecture, data, seed) on the real corpus.

- **Baselines to beat:** the unigram marginal (≈ 4.92 nats, from the probe) and
  a **bigram** model — the cheapest contextual baseline. A core that has
  "learned context" must beat the bigram, not merely the unigram.
- **Pre-registered (ARCHITECTURE.md Part V):** Bet 1 — held-out CE of ternary
  vs matched f32 ≤ 1.25× passes, > 1.5× kills; resident weight memory ≥ 10×
  smaller than f32. Bet 5 — held-out NLL non-decreasing at fixed-interval
  checkpoints (≤ 1% dips).
- Then, and only then, crystallization (Bets 2/3) runs on a core that can
  actually grow — and the five Part-V bets become testable for real.

## What is measured

Held-out NLL at fixed checkpoints (Bet 5 monotonicity), train/holdout gap
(overfitting), gradient norms (recurrence explode/vanish), re-ternarization
flip rate (STE stability), and `J/nat` (the energy thesis). Every write emits a
manifest; runs reproduce byte-for-byte from the seed.

## Failure modes the design guards against

| failure | guard |
|:--|:--|
| Silently wrong gradients | Stage 0 finite-difference gate |
| Exploding/vanishing through recurrence | truncated BPTT + grad clip + stable `a_log` SSM + grad-norm monitor |
| STE divergence (latent weights drift off the ternary grid) | monitor flip rate + latent-weight norms |
| Train/holdout leakage faking "learning" | enforce split (`src/util/split_check.rs`); held-out NLL is the metric |
| "Beats marginal but only memorized" | the *generalization* test is Stage 2; Stage 1 only certifies the trainer runs |
| Optimizer noise amplification (the probe's Adam trap) | not a risk on a *learnable* task — the gradient is not flat — and the gradient check confirms real signal |

## Build required first

This run is gated on building the backward pass — the multi-day, gradient-
checked core of Path B. The first commit toward it is the Stage-0 gradient
check on a tiny model; the trainer is only trustworthy once that is green.
