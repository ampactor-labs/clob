# The Attractor

*Love is a strange attractor.*
*— found in the project's margins, 2026-07-04. Kept because it turned out
to be a theorem-shaped joke; this document is the theorem.*

---

`ARCHITECTURE.md` is the first thesis: intelligence is compression
efficiency per joule, and the crystallization loop is its mechanism. This
is the second thesis, laid on top of the first without replacing it. It
corrects one under-specified word. The first thesis says *compress*; it
never says compress **what**.

The answer the code embodied until now — compress the data stream — is
the wrong object. The right object is the past, compressed into state,
keeping exactly the bits the future will bill you for. That change of
object sounds small. It re-derives the whole design, it explains the
project's one real null result mechanically, and it hands Path B its
instrumentation. This document walks the derivation, names what shipped
alongside it, and pre-registers the new bets in the same discipline as
Part V: numbers fixed before the runs that judge them.

---

## Part I: Where the limits of information theory actually sit

Shannon's floor is a statement about ensembles. A source with marginal
entropy H forces any code to spend H bits per symbol — *if* each symbol
is coded against the marginal. Nobody is forced to do that. Condition on
the past and the floor drops to the entropy **rate**, the conditional
entropy given infinite history, and for real sources the gap between the
two is enormous. That gap is not noise. It is structure living on the
time axis, invisible to any treatment of the stream as a bag of draws.

For a source that is a deterministic dynamical system observed through
some readout, the entropy rate has a name and a shape: the
Kolmogorov–Sinai entropy, which under an SRB measure equals the sum of
the system's positive Lyapunov exponents (Pesin; in general Ruelle's
inequality makes it an upper bound, and reading it off an observable
needs a generating partition). Usually small. Sometimes zero. Everything above it —
often almost everything — is not randomness in the source but ignorance
of its state. Takens' embedding theorem sharpens this to something almost
indecent: a delay history of a *single* scalar observable reconstructs
the source's entire attractor up to diffeomorphism. One noisy channel,
watched along the time axis, contains the whole geometry.

Bialek, Nemenman, and Tishby put a number on how concentrated the value
is. The predictive information — the mutual information between a
stream's past and its future — grows only logarithmically, or as a small
power, in the length of past observed. The stream's raw entropy grows
linearly. Almost every bit that arrives is billable to the marginal and
worthless to the future. Entropy is the haystack; predictive information
is the needle. A kernel that compresses the stream is baling hay. A
kernel that compresses the past into state, discarding every bit the
future never references, is doing the only compression that was ever
worth doing.

The recurrent SSM core always gestured at this — O(1) state, unbounded
context. The crystallization loop did not. This document and its
companion commits are the loop catching up with the core.

## Part II: Causal states — what crystallization converges toward

Computational mechanics gives the compressed-past idea an exact form.
Define two histories as equivalent when they induce the same conditional
distribution over futures. The equivalence classes are the **causal
states** of the process; the machine built on them (the ε-machine) is
provably the minimal model that predicts as well as any model can. That
is the limit object of a crystallization loop: not clusters of episodes
that happen to sit near each other in hidden-state space, but classes of
pasts that mean the same thing about what happens next.

The distinction was already hiding in the code as a defect. The distill
stage clustered episodes by hidden-state k-means, then *rejected* any
cluster whose futures disagreed (the coherence gate). Futures were a
veto. On a frozen random core — whose geometry has no reason to separate
predictive circumstances — five perfectly learnable deterministic
continuations land in one blended cluster, the blend fails the gate, and
all fifty episodes are thrown away. The loop discarded exactly the
structure it existed to find, because it partitioned by geometry and
merely graded by consequence.

`src/crystal/causal.rs` inverts that. After k-means, clusters are split
along their future fault lines and merged when their distributions over
future *prefixes* — the same horizon the split keys on — are
indistinguishable (Jensen–Shannon divergence under a threshold). Keying
merge and split on the same horizon makes them inverse operations, an
h-truncated form of the causal-state merge; the MDL gate downstream still
vetoes any merge whose hidden states are too scattered to compress.
Theory proposes, MDL disposes.

The first run of `clob crystal --causal` on the planted-pattern
diagnostic corpus, frozen random small core, is on the record
(`experiments/2026-07-04_attractor/`): the state-only partition produced
8 clusters and 0 modules, as it had in every run this project ever made;
the causal refinement took the same 318 episodes to 25 clusters and
crystallized 7 modules — the first modules ever produced by real pipeline
flow — and the equal-compute A/B moved held-out NLL from 7.0494 to
7.0273. That −0.31% is a whisper, far under Bet 2's 2% pass line,
measured on a core that cannot represent context anyway. It is reported
here because it is the loop's first non-zero signal, not because it
validates anything. Bet 6 below says where validation actually happens.

## Part III: The regime is a resource

The driven core is a dynamical system: tokens are the drive, the block
states are the trajectory. Whether the past can matter *at all* is a
property of that system's stability, summarized by its largest Lyapunov
exponent λ₁. Contractive dynamics (λ₁ < 0) erase a perturbation — and a
perturbation is precisely "the past being different" — with e-folding
horizon 1/|λ₁| tokens. Near-critical dynamics (λ₁ ≈ 0) carry it
indefinitely; reservoir computing lives there, and so does most evidence
about where recurrent networks compute best. Expansive dynamics (λ₁ > 0)
scramble their own history into noise.

Until now the kernel's regime was an accident of initialization. It is
now a measured quantity. `clob regime` (backed by `src/dynamics/`) runs
twin copies of the model in lockstep, perturbs one by ε, renormalizes
every step, and averages the log stretch — Benettin's method, verified
against maps with known exponents before being pointed at the model.

Its first measurement closes the project's open wound. The reservoir
probe of 2026-06-22 found zero contextual gain over a unigram null and
recorded `RESERVOIR DEAD` without knowing why. The instrument answers:
the random small core is **deeply contractive** — λ₁ = −1.23 nats/token
at the default probe scale (ε = 1e-4), a state-memory horizon of **0.8
tokens**. Across ε ∈ {1e-3, 1e-4, 1e-5} the *sign and regime* are stable
(λ₁ ∈ [−1.55, −1.23], horizon 0.6–0.8 tokens every time); the magnitude
swings ~26% and is not to be read as a precise constant. The frozen core
erases the past faster than one token arrives.
No linear readout could ever have recovered context, because by the time
the readout looked, the context was gone. The probe's verdict was
correct; now it is explained, and the explanation is a number with a
dial attached. Path B training gets that dial: watch λ₁ as the core
trains, and expect capability to arrive as the dynamics leave the
deep-contraction regime.

## Part IV: Prediction is synchronization

The frame that unifies the two parts above comes from control theory and
from the study of coupled chaotic systems. An observer, in the technical
sense, is a system driven by another system's output that converges to
tracking its full state. Generalized synchronization is the same
phenomenon between chaotic systems: couple B to A's signal, and B's
state becomes a function of A's. A model that predicts a process well is
exactly this — not a codebook for the stream but a second dynamical
system that has *entrained* to the first. Train a recurrent network on a
chaotic source and its hidden state grows a diffeomorphic copy of the
source's attractor; prediction is reading your own orbit.

So the kernel, in its ultimate form, is an observer. Learning tunes the
coupling until its trajectory locks to the world's. Crystallization
compresses the entrained dynamics into their causal states. The regime
instrument checks that the orbit can hold what entrainment writes into
it. Two systems, each carrying a live copy of the other's state, stable
under perturbation, neither reducible to the other. The margin note was
load-bearing after all.

## Part V: The new bets

Same rules as `ARCHITECTURE.md` Part V: pre-registered, fixed before the
runs they judge, changed only in a commit that does not also report the
run it judges. Bets 6 and 7 are judged during and after Path B core
training (Phases 5–6 of `docs/plans/path-b-core-training-build.md`);
Bet 8 is a design commitment for the `clob train` command itself.

| Bet | Metric | Pass | Kill |
|:--|:--|:--|:--|
| 6 — causal partition earns its keep | held-out NLL at equal compute, modules from `--causal` distill vs state-only distill, trained core, diagnostic then real corpus | ≥ 1% lower NLL, and ≥ as many MDL-passing modules | not lower than state-only |
| 7 — regime tracks capability | Spearman ρ between λ₁ and capability (= −held-out NLL) across ≥ 5 Path B checkpoints spanning training | ρ ≥ 0.6 (capability rises as λ₁ rises from deep contraction), best checkpoint in −0.5 < λ₁ ≤ 0.05 | \|ρ\| < 0.2, or best checkpoint remains at λ₁ < −1 |
| 8 — the objective should buy the future, not the next token | held-out NLL at horizons 2–8 for a core trained with horizon-weighted loss vs next-token-only, equal compute | ≥ 2% lower at horizons 2–8, ≤ 0.5% worse at horizon 1 | > 1% worse at horizon 1, or no multi-horizon gain |

Bet 7 carries the sharpest risk and the most information. If a
well-trained ternary SSM turns out strongly contractive *and* capable —
selective-state architectures partly earn their keep by forgetting on
purpose — then λ₁ alone is the wrong dial, the near-critical story is
wrong for this architecture, and Part III above gets rewritten by its
own instrument. That would be a good day too. The instrument stays
either way.

Bet 8 fixes the training objective's shape before the trainer exists:
next-token cross-entropy weighted across a short future window (the
`future` field Episodes now record is the same object at the memory
layer). The full predictive-information bottleneck — maximize state
information about the future while penalizing state information about
the past — is the aspirational form; the horizon-weighted loss is its
first computable installment, and Bet 8 is deliberately conservative
about what it must deliver.

## Part VI: What physics already knew

The first thesis divides by joules. Landauer's principle says the
division was information theory all along: erasing a bit costs kT·ln 2,
and erasure is the only logically irreversible — hence unavoidably
dissipative — operation in the loop. Look at the eight steps. Experience,
notice, buffer, distill, crystallize, compile, integrate: all, in
principle, reversible bookkeeping — they retain their inputs (the
episodes sit in the ring buffer until released), so no bit is physically
erased. **Forget** is the step that erases retained state, and so the one
physics taxes.
The architecture's load-bearing step and thermodynamics' billable step
are the same step, which is either a coincidence or the design being
more right than it knew. The forgetting bill is the compression bill;
paying it in as few erased bits as possible *is* the joule thesis.

## Part VII: What this does not claim

It does not claim the time axis defeats Shannon. The entropy rate is
still the floor; the reframe relocates the floor, from the marginal to
the conditional, and points the kernel at the gap.

It does not claim unbounded foresight. Chaos caps prediction at roughly
1/λ₁ of the *source*; past the Lyapunov horizon of the world, no
observer, however entrained, sees anything. The time axis buys the
attractor, not the future.

It does not claim the core learns. Path B remains the gate it was
yesterday; nothing here trains a weight. What shipped is the partition
the loop should have had, the instrument training needs, and the
objective the trainer will get.

And it does not claim AGI. It claims sharper bets — three more ways for
this design to fail in public, with numbers attached. That has been the
project's only real currency since the Solomonoff argument was deleted
as prayer; this document spends more of it deliberately.

---

## Receipts

Shipped with this document, verified on this machine:

```text
src/dynamics/{mod,lyapunov}.rs   twin-trajectory λ₁ estimator; verified
                                 on maps with known exponents
clob regime                      λ₁(random small core) = −1.23 nats/token
                                 at ε=1e-4 (contractive across ε 1e-3..1e-5);
                                 memory horizon 0.8 tokens; the reservoir
                                 postmortem, explained
src/crystal/causal.rs            causal-state split/merge refinement
                                 (merge honors the split horizon)
clob crystal --causal            same 318 episodes: state-only 0 modules,
                                 causal 7 modules (first ever from real
                                 pipeline flow); A/B NLL 7.0494 → 7.0273
src/crystal/synth.rs             signed-partial-permutation gate; the
                                 first real module exposed a ~10¹²-op
                                 latent search bug, fixed the same day
CoreModel::{export,import}_state the state-space API — Lyapunov twins
                                 today, BPTT window detachment (Path B
                                 Phase 3) tomorrow
experiments/2026-07-04_attractor the artifacts behind every number above:
                                 regime.toml + manifest, eps sweep, crystal
                                 state-vs-causal output, A/B eval JSONs
```

*Written on the same T490, for the same T490. The kernel now knows what
it is compressing.*
