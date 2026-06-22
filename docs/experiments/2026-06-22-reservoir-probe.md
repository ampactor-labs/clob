# Experiment: Is clob's frozen core a usable reservoir?

**Date:** 2026-06-22 · **Status:** complete · **Verdict:** reservoir dead → Path B (train the core)

## The question

`clob` is a ternary-recurrent kernel whose core — embeddings, the selective
SSM, every block — stands at **random initialization and is never trained**.
The design left one fork unmade (`IF_FOUND.md`, *"the one decision that defines
its future"*):

- **A — Reservoir.** The random core is a *fixed substrate by design*; all
  learning is crystallization of local corrections. "Trained well" means
  "crystallized well." Bold, unconventional — and it rests on a hidden
  assumption: that a random, recurrent, ternary projection *exposes* the
  input's predictive structure the way an echo-state network's reservoir does.
- **B — Trained core.** Train the ternary core to competence first, then run
  crystallization as the continual-compression layer on top. The manifesto's
  literal reading.

Everything downstream depends on which is true. This experiment settles it
empirically instead of by argument.

## Why it couldn't be answered before

Reservoir computing needs a *trainable readout* to decode the reservoir's
state. clob has none: embedding and unembedding are **one tied table**, and
it is random. So the model's own output is a random projection of a random
projection — on the first real-data run it scored a holdout NLL of **8.506
nats (perplexity ≈ 4940)** over a 4096-token vocab, *worse than uniform
chance* (`ln 4096 = 8.32`). You cannot ask "does the reservoir carry signal?"
through a readout that is itself noise.

## Method

The `probe-readout` subcommand supplies the missing readout as a diagnostic:

1. Stream a corpus through the **frozen** core (recurrent, in order) and cache
   the hidden state after each token — exactly the vector the tied unembedding
   reads.
2. Fit a dense f32 readout `logits = W·h + b` over those features by
   multinomial logistic regression (a convex problem).
3. Measure held-out NLL and compare against the right baselines.

If the frozen core linearly exposes contextual structure, a trained readout
will beat a context-free baseline. If it does not, it cannot.

**Why linear is the right probe — not too weak.** The obvious objection is
"maybe a *nonlinear* readout would find signal a linear one misses." But a
linear readout is exactly the test the reservoir hypothesis calls for: the
whole premise of reservoir computing (echo-state networks, extreme learning
machines) is that a fixed random nonlinear projection lifts the input into a
space where the relevant structure is **linearly** separable, so that only a
*linear* readout need be trained. If the structure were recoverable only by a
nonlinear readout, the "reservoir" would be doing none of the work — the
readout would be. So a linear readout finding nothing is not a weak result; it
is the hypothesis failing on its own terms.

## Three decisions that make the result trustworthy

A naive version of this probe gives a confident **wrong** answer three
different ways. Each was caught and fixed:

### 1. Momentum SGD, not Adam — because the objective is flat

The first run used AdamW and "diverged": training loss *rose* monotonically
from the warm start (6.20 → 6.53), and it reported DEAD. That verdict was
right by accident, for the wrong reason. On a **flat** objective — which a
dead reservoir produces — the true gradient is ≈ 0, and Adam divides that ≈ 0
gradient by an ≈ 0 second moment, normalizing *pure noise* into O(1) steps. It
random-walks away from the optimum and manufactures a false "divergence"
precisely in the dead case. Plain momentum SGD has the property the diagnostic
needs: its step scales with the gradient, so a dead reservoir takes vanishing
steps and *stays at the baseline*, while a live one descends below it.

### 2. The trained-bias-only null — not the hand-set marginal

Warm-starting the bias at the Laplace unigram and reporting gain over *that*
miscredits **bias-fitting** as if it were context. A weight-decay sweep made
this visible: the apparent gain over the Laplace marginal *grew monotonically
as the weight `W` was crushed toward zero* — i.e. the improvement came from the
one thing weight decay doesn't touch (the trainable bias finding a better token
marginal than add-one smoothing), not from the features. The correct null is a
**bias-only readout** (W frozen at 0) trained under the identical objective.
The contextual contribution is then exactly `bias_only − full_readout`, with
the marginal advantage divided out.

### 3. Standardization, a regularization sweep, and a convergence guard

Features are standardized (per-dim, from train stats — information-preserving)
so the learning rate isn't at the mercy of the random core's output
magnitudes. A weight-decay sweep + best-epoch selection means a weak signal
can't hide behind a single under- or over-regularized fit. A convergence guard
refuses to call a result "dead" if the fit's own training loss ran away from
the warm start.

All three are encoded as unit tests: a separable signal must beat the null,
pure noise must not, and the bias-only null must be unable to exploit features
at all.

## Result

Authoritative run: 150,000 train tokens, 52,271 held-out tokens, 3-way weight
decay sweep, 4 epochs (`experiments/2026-06-22_first_run/probe_readout.json`).
Holdout NLL, lower is better:

| readout | holdout NLL | perplexity | note |
|:--|--:|--:|:--|
| uniform floor `ln(vocab)` | 8.318 | 4096 | — |
| **tied random readout** (model's own) | **8.506** | **4943** | worse than chance |
| Laplace unigram marginal | 4.917 | 137 | fixed bias |
| **trained-bias-only null** (W ≡ 0) | **4.917** | 137 | proper context-free baseline |
| full trained readout (best of wd ∈ {0.01, 0.1, 1}) | 4.925 | 138 | adding W *hurts* |
| **contextual gain over null** | **0.000 nats (0.0%)** | — | the deciding number |

The finding is identical across every scale tested (5k, 100k, 150k). Two
details make it airtight:

- **The trained-bias-only null equals the Laplace marginal exactly** (4.917).
  At scale the trained bias converges to the marginal, so the small bias-fitting
  advantage that appeared at 100k vanishes — which is precisely why the
  bias-only control, not the hand-set marginal, is the right baseline.
- **Every full-readout fit is *worse* than the null** (4.925, 4.952, 4.953 vs
  4.917). Giving the decoder the core's hidden state doesn't merely fail to
  help — it strictly degrades held-out NLL. The features are pure overfitting
  noise; the best a regularized readout can do is fall back to W = 0.

A trainable linear decoder extracts **no predictive information from the frozen
core's hidden state beyond the token marginal**. The random ternary SSM does
not expose linearly-readable context.

## Verdict and decision

**Reservoir dead.** Path A — learning purely by crystallizing corrections onto
a fixed random core — is falsified for this architecture as built. The fork
resolves to **Path B: train the core.** (`IF_FOUND.md` updated accordingly.)

The probe also produced an immediately useful artifact: a *trained readout*
that takes the model from 8.51 NLL (worse than chance) to the marginal (4.93),
measured end-to-end on the real inference path — **perplexity 4943 → 138, a
35.7× improvement**. That is shipped as the **trained-readout sidecar**
(`probe-readout --save-readout`, installed via `eval --readout`) —
Path B, step one: the output head is now untied and trainable. It does not make
the core learn, but it fixes a genuinely broken component and is the substrate
the recurrent-core training will build on.

## What's next (Path B roadmap)

Making the *core* learn is the real prize and a deliberate next phase, not a
thing to half-land:

1. **Untie the output head** — done, as the sidecar above.
2. **Backward pass through the stack** — straight-through gradients for the
   ternary linears (scaffolded in `src/learn/`), through GLU/RMSNorm and the
   selective SSM, with backprop-through-time for the recurrence. This is the
   careful, gradient-checked core of the work.
3. **`train` subcommand** — wire `src/learn/` (AdamW over latent f32 weights,
   `re_ternarize`) into a streaming trainer; demonstrate the core's held-out
   NLL falling *below* the marginal — the first time clob actually learns.
4. **Then** crystallization runs on a core that can grow, and the five Part-V
   bets become testable for real.

## Reproduce

```
clob probe-readout \
  --model seed.clob --train train.tokens --holdout holdout.tokens \
  --tokenizer tokenizer.bin --max-tokens 150000 --epochs 4 \
  --weight-decays "0.01,0.1,1.0" --seed 1 \
  --out probe_readout.json --save-readout readout.bin

# Demonstrate the shipped readout on the real inference path:
clob eval --model seed.clob --corpus holdout.txt --tokenizer tokenizer.bin            # nll 8.506, ppl 4943
clob eval --model seed.clob --corpus holdout.txt --tokenizer tokenizer.bin --readout readout.bin   # nll 4.930, ppl 138
```
