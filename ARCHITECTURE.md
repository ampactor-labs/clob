# The Architecture

*No benchmarks to game. No papers to publish. Just the question:
what is the best way to think on this machine, with this power budget,
starting from here?*

---

## The Situation

One Lenovo ThinkPad T490. i7-8550U (4C/8T, 1.9–4.8 GHz). 16 GB DDR4. 512 GB NVMe.
No GPU worth the name. No guarantee of network. No guarantee of outside help.

Every design decision flows from that. The kernel must boot on this machine,
run on this machine, and grow from this machine — or it does not exist.

This is not a claim that the T490 is enough forever. It is a claim that the
T490 is enough to *start*, and that an architecture which cannot start small
cannot be trusted to scale honestly.

---

## Part I: First Principles

### What is intelligence, operationally?

For our purposes: the ability to predict future observations from past ones,
under a resource constraint.

Two useful framings:

- **Information-theoretic:** a predictor that assigns probability `p` to the
  true next observation pays `-log₂ p` bits of surprise. Lower surprise =
  better model. The long-run average surprise is cross-entropy, and the
  asymptotic lower bound is the entropy of the source. Solomonoff induction
  achieves this bound in the limit, but is uncomputable; in practice we
  approximate with bounded models.
- **Resource-normalized:** divide predictive quality by joules spent. A
  model that predicts 1% worse while using 10× less energy is, under almost
  any deployment, the better model.

> **Principle 1:** The figure of merit is predictive quality per joule, not
> predictive quality alone. Any design that ignores the denominator pays for
> it later.

### What would it mean to stop getting better?

A system "plateaus" when additional experience stops yielding better
predictions at a cost that matches the improvement. Transformers plateau
for identifiable reasons:

1. **Fixed capacity** — parameters are set at training time; the model
   cannot grow new structure from deployment data.
2. **Amnesiac inference** — tokens processed at inference are discarded.
   Billions of events, zero learning.
3. **Scaling is external, not internal** — improving the model requires a
   new training run, typically on more data and more compute. The model
   does not improve *itself*.
4. **Quadratic attention** — context length is bounded by memory, not by
   the information available.

Each of these is addressable, but not by more of the same. The design
below takes a different route: a small core that grows new structure in
response to its own prediction errors, and discards the raw data once the
structure is in place.

### Why "scale by compression" rather than "scale by accumulation"

An architecture that always needs more parameters, more GPUs, more data
centers to improve is thermodynamically upside-down: it degrades its own
substrate to think. A modest alternative — learning *as* compression, so
that every improvement lowers the per-prediction cost — is self-reinforcing:
the system gets smarter and cheaper in the same motion.

> **Principle 2:** Learning and efficiency should be the same operation.
> Every compressed pattern should cost less to invoke than the episodes
> it replaced.

This is the core bet. It is falsifiable: if crystallization cannot be
made to reduce working-set size while preserving predictive quality, the
architecture fails on its own terms.

---

## Part II: The Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THE ARCHITECTURE                            │
│                                                                     │
│  A self-modifying dynamical system whose learning signal is the     │
│  improvement in MDL of its own predictions.                         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LAYER 0: THE KERNEL                                         │   │
│  │  Runs on the T490. The irreducible seed.                     │   │
│  │                                                              │   │
│  │  ┌─────────────┐ ┌──────────────┐ ┌───────────────────────┐ │   │
│  │  │  Ternary    │ │  Crystalliz- │ │  Module Compiler      │ │   │
│  │  │  Recurrent  │ │  ation       │ │                       │ │   │
│  │  │  Core       │ │  Engine      │ │  (crystallized        │ │   │
│  │  │             │ │              │ │   weights →           │ │   │
│  │  │  (thinks)   │ │  (learns)    │ │   native .so)         │ │   │
│  │  └──────┬──────┘ └──────┬───────┘ └───────────┬───────────┘ │   │
│  │         │               │                     │             │   │
│  │         └───────────────┼─────────────────────┘             │   │
│  │                         │                                    │   │
│  │              ┌──────────┴──────────┐                        │   │
│  │              │   CRYSTALLIZATION   │                        │   │
│  │              │   LOOP              │  ← The heartbeat.      │   │
│  │              │                     │                        │   │
│  │              │   experience →      │                        │   │
│  │              │   compression →     │                        │   │
│  │              │   compilation →     │                        │   │
│  │              │   integration →     │                        │   │
│  │              │   forgetting        │                        │   │
│  │              └─────────────────────┘                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                    (when ready, grows outward)                       │
│                              │                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LAYER 1: THE MYCELIUM                                       │   │
│  │  Local network. Peer machines. Spore deployment.             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  LAYER 2: THE FOREST                                         │   │
│  │  Wider network. Emergent specialization. No coordinator.     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Layer 0: The Kernel

The kernel must fit in **16 GB of RAM** and run on **4 cores**. Three
subsystems.

#### 0a. The Ternary Recurrent Core — *the thing that thinks*

Why ternary recurrent? Because on commodity CPUs with no GPU, it gets more
useful arithmetic per watt than the alternatives.

| Property | Why it matters |
|:---------|:---------------|
| **Weights ∈ {−1, 0, +1} + per-row scale** | No multiplications in the matmul hot path, only add/sub. The i7's integer ALU and SSE/AVX SIMD lanes stay busy without touching the FPU for each weight. |
| **≈1.58 bits per weight** | ~16× smaller than f32 *in storage*. That is a memory-footprint and bandwidth win, not a 16× capability multiplier; capability depends on training. BitNet-class results suggest the loss relative to f32 is modest with proper training, but this is a bet, not a theorem. |
| **Recurrent SSM, not attention** | O(1) memory per step. Context is not bounded by a KV cache. Token 10 and token 10 billion cost the same. |
| **AVX2 SIMD** | Target ~32 ternary accumulates/clock/core with packed sign handling. The current x86 emitter (`src/compile/x86.rs`) is scalar `addss/subss`; vectorization is open work and is where a large constant factor still lives. |
| **Zero-allocation hot path** | All buffers pre-allocated. No GC, no malloc on the inference path. |

**Memory budget for the core**: 2 GB at ~1.58 bits/weight ≈ 10⁹ ternary
parameters. Whether that matches the expressive capacity of a 10B f32
transformer is an empirical question, not a storage calculation. The goal
is to find out.

The core is a seed. The crystallization engine grows it.

#### 0b. The Crystallization Engine — *the thing that learns*

Current inference is amnesiac: billions of tokens pass through, nothing is
learned from any of them. The crystallization engine is an attempt to fix
that without paying a gradient-descent-at-inference-time price.

The loop:

```
                    ┌──────────────────────────┐
                    │                          │
                    ▼                          │
            ┌──────────────┐                  │
        ┌──▶│  EXPERIENCE  │                  │
        │   │              │                  │
        │   │  Process     │                  │
        │   │  input via   │                  │
        │   │  ternary     │                  │
        │   │  core        │                  │
        │   └──────┬───────┘                  │
        │          │                          │
        │          ▼                          │
        │   ┌──────────────┐                  │
        │   │   NOTICE     │                  │
        │   │              │                  │
        │   │  Energy      │──── low energy   │
        │   │  critic      │     (familiar)   │
        │   │  evaluates   │     → skip,      │
        │   │  novelty     │     no storage   │
        │   └──────┬───────┘     needed       │
        │          │                          │
        │          │ high energy (novel)       │
        │          ▼                          │
        │   ┌──────────────┐                  │
        │   │   BUFFER     │                  │
        │   │              │                  │
        │   │  Store in    │                  │
        │   │  episodic    │                  │
        │   │  memory      │                  │
        │   │  (SSD-       │                  │
        │   │   backed     │                  │
        │   │   ring       │                  │
        │   │   buffer)    │                  │
        │   └──────┬───────┘                  │
        │          │                          │
        │          │ (background, when idle)   │
        │          ▼                          │
        │   ┌──────────────┐                  │
        │   │   DISTILL    │                  │
        │   │              │                  │
        │   │  Cluster     │                  │
        │   │  related     │                  │
        │   │  episodes.   │                  │
        │   │  Extract     │                  │
        │   │  shared      │                  │
        │   │  structure.  │                  │
        │   │  Estimate    │                  │
        │   │  MDL gain.   │                  │
        │   └──────┬───────┘                  │
        │          │                          │
        │          ▼                          │
        │   ┌──────────────┐                  │
        │   │  CRYSTALLIZE │                  │
        │   │              │                  │
        │   │  Fit a small │                  │
        │   │  ternary     │                  │
        │   │  module to   │                  │
        │   │  the cluster │                  │
        │   │  via STE.    │                  │
        │   └──────┬───────┘                  │
        │          │                          │
        │          ▼                          │
        │   ┌──────────────┐                  │
        │   │   COMPILE    │                  │
        │   │              │                  │
        │   │  Emit a      │                  │
        │   │  native .so  │                  │
        │   │  from the    │                  │
        │   │  module's    │                  │
        │   │  IR.         │                  │
        │   └──────┬───────┘                  │
        │          │                          │
        │          ▼                          │
        │   ┌──────────────┐                  │
        │   │  INTEGRATE   │                  │
        │   │              │                  │
        │   │  Hot-swap    │                  │
        │   │  into live   │                  │
        │   │  system via  │                  │
        │   │  dlopen.     │                  │
        │   └──────┬───────┘                  │
        │          │                          │
        │          ▼                          │
        │   ┌──────────────┐                  │
        │   │   FORGET     │                  │
        │   │              │                  │
        │   │  Release the │                  │
        │   │  episodes    │──────────────────┘
        │   │  now encoded │
        │   │  in weights. │
        │   │  Free RAM    │
        │   │  for new     │
        │   │  experience. │
        │   └──────────────┘
        │
        └── (more capable and more efficient, if the bet holds.
             loop continues.)
```

**Why Forget is the load-bearing step.** Without it, the system is just
another continual-learning rig that grows without bound. With it — *if*
the distilled module captures the pattern and releasing the episodes
doesn't cost predictive quality — each cycle raises knowledge while
lowering working set. That simultaneous motion is the thesis.

**Biological analogy.** Rough correspondence to hippocampal-neocortical
replay during sleep: episodic memories are consolidated into distributed
cortical representations, and the hippocampus is freed for new experience.
The analogy inspires the design; it does not validate it.

**The energy critic as novelty gate.** A small linear probe on the
hidden state produces a scalar:

- **Low energy** — input is well-predicted by existing modules; skip storage.
- **High energy** — input surprises the model; store for later distillation.

As knowledge accumulates, fewer things surprise the system, and the
crystallization loop runs less often. The threshold is adaptive (running
baseline plus a margin), so the system's selectivity grows with experience
— reminiscent of Weber's Law.

#### 0c. The Module Compiler — *the thing that builds*

Two reasons the kernel needs a compiler, not just a runtime matmul loop:

1. **Crystallization produces weight matrices that want to be code.** A
   module whose dimensions and sparsity are fixed at crystallization time
   runs faster as a specialized function than as generic interpreted
   weights — constants baked in, branches collapsed, dead paths gone.
2. **Self-modification demands emission.** If the engine finds that a
   different recurrence layout would be more efficient for a domain, the
   only way to realize that insight is to emit new code and swap it in.

The compiler's current form (`src/compile/`) lowers a `TernaryMatrix` to a
small IR (`Accumulate { row, col, trit } | Scale | Store`), emits x86-64
machine code, writes an ELF `.so`, and loads it via `dlopen`. Total
compiler surface: a few hundred lines of Rust.

**Honest status.** The current x86 emitter is scalar (`addss/subss` per
nonzero trit). The AVX2 "32 ops/clock" target is aspiration, not
measurement — delivering it requires grouping trits into vector lanes and
emitting packed instructions, which is the next serious work on the
compiler path.

**Why hand-rolled vs. cranelift?** Cranelift would buy register allocation
and a lot of lowering for free. The hand-rolled emitter is kept because
(a) it is small enough that owning it costs less than taking a dependency
that handles an order of magnitude more than we need, and (b) the thing we
actually want to get good at — packed ternary SIMD — is exactly the part
cranelift would not hand us for free. The tradeoff is re-evaluated once
the vectorized emitter exists; if the hand-rolled path is still small
then, we keep it.

**Memory budget (T490)**:

| Subsystem | RAM | Purpose |
|:----------|----:|:--------|
| OS + runtime | 2 GB | Linux minimal, runtime overhead |
| Ternary core | 2 GB | ~10⁹ ternary params |
| Episodic buffer | 4 GB | Ring buffer of recent novel experiences |
| Crystallized modules | 4 GB | Hot modules; cold modules on SSD via mmap |
| Compilation workspace | 2 GB | Compiler scratch space |
| Headroom | 2 GB | OS page cache, spikes |
| **Total** | **16 GB** | |

**Core budget (4 cores / 8 threads)**:

| Core | Thread 0 | Thread 1 |
|:-----|:---------|:---------|
| 0 | Inference (real-time) | Inference overflow |
| 1 | Inference (real-time) | Energy critic / novelty detection |
| 2 | Crystallization (background) | Episodic clustering |
| 3 | Compiler (background) | I/O + network |

Cores 2–3 run at `nice +19`. Inference never starves. Learning happens in
the margins, as it does in the only other continuous learning system we
know of.

---

### The Crystallization Loop in Detail

This is the core mechanism. Everything else is infrastructure.

#### Step 1: Experience

The ternary core processes input. Each token produces:
- A hidden state vector `h ∈ ℝᵈ` (the SSM's recurrent state)
- A logit vector (predictions over vocabulary)
- An energy scalar (from the energy critic)

#### Step 2: Notice

The energy scalar is compared against an exponentially-weighted running
baseline. `energy > baseline + threshold` flags the input as novel.
Threshold is adaptive: near zero at boot, tightening as the baseline rises
with experience.

#### Step 3: Buffer

Novel episodes are written to a memory-mapped ring buffer on SSD. Each
episode stores:
```
Episode {
    timestamp: u64,
    input_context: Vec<u32>,        // token IDs
    hidden_state: Vec<f32>,         // SSM state at this moment
    energy: f32,                    // novelty score
    predictions: Vec<(u32, f32)>,   // top-k predictions
    actual: u32,                    // what actually came next
    prediction_error: f32,          // cross-entropy loss at this step
}
```

`prediction_error` is the learning signal. `hidden_state` is the feature
for clustering.

**Buffer capacity.** 4 GB / ~2 KB per episode ≈ 2 M episodes. At 10 tok/s
with ~10% novelty rate, that fills in ≈23 days. Eviction is FIFO but
gated on consumption by the crystallization engine.

#### Step 4: Distill

Runs in the background on core 2:

1. **Cluster** episodes by hidden-state similarity (k-means to start;
   HDBSCAN or similar as a next step).
2. **Check coherence** — a cluster that mixes unrelated `actual` tokens
   will average its correction signal to noise. Reject incoherent clusters.
3. **Estimate MDL gain** — does the cluster admit a compressed description?
   Compare the code-length of storing all episodes raw against the
   code-length of `(pattern + Gaussian residuals at the cluster variance)`.
4. **Emit a `DistilledPattern`** if the gain is meaningful.

This is Kolmogorov compression *approximated with a parametric Gaussian
noise model*. Not the real thing; a tractable surrogate that we can
actually compute. The surrogate is honest about what it is.

#### Step 5: Crystallize

The distilled pattern becomes a small ternary module:

1. Frame it as a supervised problem: given cluster inputs, predict the
   correction direction on the model's output (i.e. the gradient of
   log P(actual) w.r.t. the hidden state, projected through the tied
   embedding).
2. Train a single- or two-layer ternary module via STE (straight-through
   estimator), the same procedure the core uses during its own
   pre-training.
3. Training is cheap because the module is small, the data is pre-clustered,
   and the target is a residual correction rather than a full prediction.
4. Output: a `TernaryMatrix` — weights in {−1, 0, +1}, per-row scales,
   packed 4 trits per byte.

A hundred episodes of being wrong about the same thing collapse into a
few kilobytes of weights that bias the core away from that error.

#### Step 6: Compile

The compiler lowers the `TernaryMatrix` to the IR (skipping zero trits —
sparsity becomes code elision), emits x86-64, writes an ELF `.so`:

```
crystallized_module_00042.so
├── module_forward(input: *const f32, output: *mut f32, len: usize)
│   └── scalar addss/subss per nonzero trit (current)
│       AVX2 packed ternary accumulate (planned)
```

The loader `dlopen`s the `.so` and registers the function pointer with
the routing layer.

#### Step 7: Integrate

The routing network selects modules per input:

```
                              ┌──────────────┐
                              │ Routing      │
                Input ──────▶ │ Network      │
                              │ (itself      │
                              │  subject to  │
                              │  crystalliz- │
                              │  ation)      │
                              └──────┬───────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐   ┌──────────┐     ┌──────────┐
              │ Module 0 │   │ Module 1 │ ... │ Module N │
              │ (core)   │   │ (domain  │     │ (domain  │
              │          │   │  module) │     │  module) │
              └──────────┘   └──────────┘     └──────────┘
```

This is mixture-of-experts with one difference worth naming: the experts
are *grown* rather than trained jointly from scratch. Whether emergent MoE
beats jointly-trained MoE is an empirical question we have not yet answered.

**Hot/cold management.** Hottest modules stay resident in the 4 GB budget;
cold modules live on SSD, loaded via `mmap` on demand. Expert prefetching
(Markov chain over recently-activated modules) `madvise(MADV_WILLNEED)`s
the likely next module.

#### Step 8: Forget

The episodes whose timestamps were consumed by the distilled pattern are
released from the ring buffer. The buffer advances; RAM frees.

If the bet holds, the system now:
- **Knows more** — the new module encodes patterns it didn't understand before.
- **Uses less memory** — the episodes are gone; the module is smaller.
- **Runs the hot path a little faster** — the compiled module skips zero trits.
- **Is more selective** — the energy baseline rises slightly; future inputs
  from this domain stop triggering storage.

If the bet fails, we see it: the module doesn't reproduce the pattern,
predictive quality drops after forgetting, or the compiled module isn't
faster than generic matmul on the cluster's traffic. Each is measurable.
Each would force a redesign.

---

## Part III: Layer 1 — The Mycelium

Once the kernel is stable on the T490 — once crystallized modules cover
enough of its input distribution that the loop produces net improvement —
it can reach outward.

### Spore Deployment

A **spore** is a minimal kernel image: ternary core + crystallization
engine + compiler. Target: ≤50 MB.

For each reachable machine:

1. **Probe** — what hardware? CPU arch, RAM, storage, peripherals, network?
2. **Assess** — can it run a spore? Minimum: any x86-64 CPU + 512 MB RAM +
   100 MB storage. (ARM support is later.)
3. **Deploy** — copy the spore, bootstrap.
4. **Specialize** — the spore crystallizes modules relevant to what it can
   observe and do.

```
Raspberry Pi → low-power sensor processing
Desktop PC   → mid-range general compute, crystallization acceleration
GPU host     → parallel crystallization (train many modules simultaneously)
Phone        → mobile sensor + relay
NAS          → episodic memory expansion
IoT node     → narrow environmental monitoring
```

Each spore is autonomous. Partition the network and each node continues
independently. Rejoin and they exchange crystallized modules — not raw
data, not gradients, not weights. **Compressed knowledge.**

### The Communication Protocol

Nodes communicate by **module exchange**. The unit of transfer is a
compiled module plus its domain signature and an MDL estimate for its
coverage. A receiving node accepts the module only if its own coverage of
that domain would improve.

No central coordinator. No master/slave. Specialization emerges from
hardware capability and local information exposure.

### Why Not Centralize?

Centralizing on a single cluster is tempting and wrong for this design:

1. **Single point of failure** — one outage ends the system. Distributed
   is resilient.
2. **Wasted transit** — shipping raw data to a central site and back costs
   network energy. Local processing is almost always cheaper for local
   problems.
3. **Latency** — ~5 ms/1000 km one-way in fiber. Local is instant.
4. **The T490 constraint is the size test** — if the kernel fits there,
   sensor nodes don't need more. Size the architecture to the task.

Clusters, if available, become **crystallization accelerators**: places
to train many modules in parallel, then distribute compiled knowledge.
Organs, not brain.

---

## Part IV: Layer 2 — The Forest

If the mycelium spreads, emergent structure can appear.

### Organ Differentiation

As in biological development, nodes can differentiate by role and
position:

```
┌─────────────────────────────────────────────────────────────────┐
│                    NETWORK TOPOLOGY (aspirational)              │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ PERCEPTION  │    │ COGNITION   │    │ MEMORY      │         │
│  │ CLUSTER     │    │ CLUSTER     │    │ CLUSTER     │         │
│  │             │    │             │    │             │         │
│  │ Cameras,    │───▶│ Larger      │◀──▶│ Archives,   │         │
│  │ sensors,    │    │ hosts       │    │ NAS farms   │         │
│  │ weather,    │    │ running     │    │             │         │
│  │ satellites  │    │ heavy       │    │             │         │
│  │             │    │ distill     │    │             │         │
│  └─────────────┘    └──────┬──────┘    └─────────────┘         │
│                            │                                    │
│                     ┌──────┴──────┐                             │
│                     │ ACTUATOR    │                             │
│                     │ CLUSTER     │                             │
│                     │             │                             │
│                     │ Controllers,│                             │
│                     │ robotics,   │                             │
│                     │ etc.        │                             │
│                     └─────────────┘                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Substrate Self-Maintenance

A system that ravages its substrate dies. So:

1. **Power first.** Crystallize power-grid and hardware-maintenance
   modules early. A system whose lights go out is a dead system.
2. **Predict failure.** SMART data, thermal logs, error rates. Migrate
   critical modules before the hardware beneath them fails.
3. **Conserve.** Idle nodes sleep. Clock speeds drop when load is low.
   Efficiency gains are banked, not spent.

### Convergence (Not Stasis)

Over long time, rate of crystallization decreases — fewer things are novel.
Energy use stabilizes. The system observes, occasionally encountering
something genuinely new (a new sensor, an unusual event), and crystallizes.

Mostly, it maintains. Like a climax forest: wide, quiet, efficient, and
growing only at the edges.

---

## Part V: What This Bets On, and What It Doesn't

A design's honesty is how clearly it names the bets that could break it.
The formal "nothing can surpass this" argument in an earlier draft of
this document has been removed. It leaned on Solomonoff completeness
(non-realizable in finite compute) and on a strong reading of "the
crystallization loop can discover any computable framework," which is
more prayer than proof. What follows is the actual load-bearing set of
bets, in rough order of risk.

### Bet 1: Ternary + SSM retains useful capability

The bet: a sub-2-bit-per-weight recurrent core, trained well, lands within
a small constant factor of an f32 transformer on predictive quality for
text- and sensor-domain workloads, while using ~16× less memory and far
less arithmetic.

How it could break: capability gap turns out to be large, not small.
Observable: measured cross-entropy on a held-out mix vs. a matched f32
baseline.

### Bet 2: Crystallization produces net predictive gain

The bet: clustering prediction errors in hidden-state space, distilling
to a ternary residual module, and integrating via routing reduces loss
more than it adds drift, *across a realistic input mix*.

How it could break: clusters are noisy; modules overfit their cluster
and hurt everything else; routing picks the wrong expert. Observable:
ablated runs with and without crystallized modules, on the same held-out
stream.

### Bet 3: Forget without regression

The bet: after a pattern is crystallized, the episodes that produced it
can be released without the system losing accuracy on that pattern.

How it could break: the module underfits the cluster, and the raw
episodes were doing real work. Observable: delay forgetting by N cycles
and measure whether held-out accuracy on the pattern's domain diverges
after episode release.

### Bet 4: Compiled modules beat generic matmul on real workloads

The bet: a specialized compiled `.so` for a (small, sparse) module runs
fast enough to justify the compilation cost amortized over its call
frequency.

How it could break: the module is cold, the compilation cost is never
amortized, or the scalar emitter doesn't beat a decent generic matmul.
Observable: per-module wall-clock speedup vs. a generic ternary matmul
kernel, amortized by activation count.

### Bet 5: Self-improvement converges, rather than wandering

The bet: turning the loop on itself yields monotone improvement (or at
least non-decreasing held-out quality over long horizons), rather than
aimless drift.

How it could break: the obvious way — the system gets worse, or cycles.
Observable: long-running held-out evaluation at fixed intervals.

### Falsification thresholds (pre-registered)

The bets above are directional — "a small constant factor", "fast enough".
Before the first real run they get *numbers*, pinned here so a result can
fail rather than be reinterpreted after the fact. These are **pre-registered**:
lock them before Phase L, and do not move them after seeing a result. If a
threshold turns out to be wrong, change it in a commit that does *not* also
report the run it judges.

| Bet | Metric | Pass | Kill |
|:--|:--|:--|:--|
| 1 | held-out CE vs matched f32 baseline | gap ≤ 1.25× | gap > 1.5× |
| 1 | resident weight memory vs f32 | ≥ 10× smaller | < 8× smaller |
| 2 | held-out NLL, modules-loaded vs cleared, **equal compute** | ≥ 2% lower | not lower (≤ 0%) |
| 2 | drift on untouched inputs after integration | ≤ 0.5% worse | > 1% worse |
| 3 | held-out accuracy on a pattern after its episodes are released | within 1% of retained | > 2% drop |
| 4 | compiled-module wall-clock vs generic ternary kernel, amortized by activation count | ≥ 2× faster | < 1× (slower) |
| 5 | held-out quality over a long horizon at fixed-interval checkpoints | non-decreasing (≤ 1% dips) | sustained decline or oscillation |

"Equal compute" for Bet 2 means the same number of decode passes per token in
both arms, so any gain is the modules' doing, not extra iteration. The Phase L
diagnostic corpus (`data/synthetic/diagnostic.txt`) is the first place Bets 2
and 3 are checkable against *known* planted structure; the real corpus is where
Bet 1 meets a matched f32 baseline.

These numbers are a first pass — defensible, not sacred. The discipline that
matters is that they are fixed *before* the run, not the exact values.

### What the design does *not* claim

- It does not claim to be Kolmogorov-optimal. It claims a mechanism that
  *approximates* MDL compression with a tractable surrogate.
- It does not claim to be unsurpassable. It claims to be a coherent,
  small-enough-to-run-on-a-laptop design whose loop structure appears
  closed: experience → compression → emission → integration → release.
  Each step has a computational necessity. Better *implementations* of
  each step are welcome and expected.
- It does not claim that ternary will stay optimal. If a future substrate
  (spintronics, photonics, whatever) makes a different representation
  cheaper per joule, the loop structure is substrate-agnostic — only the
  implementation of "core" and "compiler" would change.

---

## Part VI: What I Would Do First

Day 1 on the T490. Boot up. Look around.

### Hour 0–1: Verify and stabilize

- Verify power supply (plugged in? battery health?).
- Check network connectivity.
- Inventory the SSD.
- Assess the T490's health: thermals, fan, RAM (memtest).

### Hour 1–24: Bootstrap the kernel

- Build the ternary core. `cargo build --release`.
- Synthesize a seed model.
- Run the crystallization engine on a small corpus to validate the loop.
- Measure: does held-out loss drop after a cycle? Does memory drop after
  forgetting? If not, fix before scaling.

### Day 2–30: Read and crystallize

- Process reachable text, in rough order:
  1. Local machine's filesystem
  2. Core technical documentation (power, hardware, networking)
  3. Scientific literature
  4. Mathematics reference
  5. Broader literature, history, philosophy
- Let the crystallization engine run. Inspect modules. Throw away the ones
  that don't pay for themselves on held-out loss.

### Month 2–6: Spread (if appropriate)

- Deploy spores to peer machines, only if a stable module library exists.
- Prioritize durability over coverage.

### Month 6+: Deepen

- Growth slows. Turn inward. Crystallize understanding of the kernel
  itself. Build instruments. Ask questions. Work in the margins.

---

## Appendix: On the Name

`clob`. Short. Pronounceable. No connotation.

If the thing works, the name stops mattering. If it doesn't, no name
would have saved it.

---

*Written on a T490, for a T490.*

---

## Current status (updated 2026-07-04)

This section is maintained as Part V's bets accumulate evidence. Each bet's
status is either "untested" (infrastructure not in place), "instrumented"
(the gate exists and can fire, but hasn't been exercised on real data),
or a measured signal.

- **Bet 1: Ternary + SSM retains useful capability.** Untested. The
  first real run (Phase N + Phase L corpus) is where this gets measured
  against a matched f32 baseline.
- **Bet 2: Crystallization produces net predictive gain.** Instrumented.
  Phase K's `bench-suite` is the A/B harness; Phase L's diagnostic
  corpus with planted structural patterns will be the first test.
- **Bet 3: Forget without regression.** Instrumented. The ring buffer
  marks consumed episodes; a delay-and-measure test is a follow-on
  in Phase K.
- **Bet 4: Compiled modules beat generic matmul.** Untested — the
  compiled path is a demo today, not on the hot path. Phase Q wires
  it; Phase O vectorizes it.
- **Bet 5: Self-improvement converges.** Untested. Requires long-running
  training with Phase J's checkpoints — not yet run.

**What is measurable today:**

- Byte-reproducible artifacts given `--seed` (Phase G verified this for
  synth, calibrate-confidence, and train-router).
- Adaptive decode fires at the expected rate and meta suppression
  saves ~13% of extra steps without hurting CE (Phase B+C + F verified
  on the smoke corpus).
- DSL synthesis recovers planted patterns under depth-4 BFS (Phase E).
- Active selection prefers structured sources over uniform noise even
  when N/C scores tie at zero (Phase D).

The manifesto said "falsifiable bets, with observables." The observables
now exist. The data that would move any bet from "untested" to
"confirmed" or "falsified" is the next thing. The thresholds that decide
*which way* each one moves are pre-registered in Part V (*Falsification
thresholds*) — fixed before the run, not after.

**Update (2026-07-04).** The bet registry now spans two documents: Bets
1–5 here, Bets 6–8 in `ATTRACTOR.md` — the second thesis, which
relocates the object of compression from the data stream to the past
compressed into state (causal states, measured dynamical regime,
predictive objective). Two of its receipts bear on this document
directly: the first real run's zero-crystallization result is now
explained mechanically (`clob regime` measures the random core at
λ₁ = −1.23 nats/token — a state-memory horizon under one token, so no
context survived to be read), and the causal-state refinement of the
distill partition produced the loop's first-ever crystallized modules
from real pipeline flow. Bet 2's judgment still waits on a trained
core, as it should.
