# Build plan: train the ternary recurrent core (Path B)

Companion to `docs/experiments/next-run-core-training.md` (the *run* design).
This is the *build* — the backward pass and trainer that the run is gated on.

## Where we start (what exists, what's missing)

Already built (`src/learn/`):
- `LatentWeights` — f32 shadow weights behind each ternary matrix, with
  `re_ternarize`.
- `GradBuffer::accumulate_ste(latent, input, output_grad)` — computes `dL/dW`
  for one ternary linear via the straight-through estimator (outer product +
  STE mask).
- `AdamW` — steps `LatentWeights` from a `GradBuffer`.
- A training-mode forward hook (`decode_step_training` / `Block::forward_training`)
  that today captures only the MoE router's state for REINFORCE.

Missing for end-to-end backprop (this is the build):
1. The **`dL/dx` path** for every layer — `accumulate_ste` gives `dL/dW` but not
   the gradient w.r.t. the layer *input*, without which gradient cannot flow
   between layers.
2. **Per-timestep activation capture** within a truncation window (the current
   forward is zero-allocation and overwrites its buffers).
3. The **recurrence backward** (backprop-through-time for the selective SSM).
4. Backward for the **non-ternary** pieces: RMSNorm, the GLU nonlinearity,
   embedding/unembedding, and the SSM's `a_log` / `dt` / `D` parameters.
5. The **orchestration**: a truncated-BPTT training loop and a `train` command.

## A note on what "gradient check" certifies

Ternary quantization is non-differentiable, so we cannot finite-difference the
*ternary* forward and expect a match. The discipline is: the differentiable
**surrogate** is the forward computed with the f32 *latent* weights; we
gradient-check the autograd of that surrogate, and the straight-through
estimator is the accepted bridge from the surrogate's gradient to the deployed
ternary forward. So every gradient check below finite-differences the
**f32-latent forward**, and STE correctness is verified separately (the `dL/dW`
the surrogate produces is what `accumulate_ste` must yield with mask = 1).

## Design decisions (locked before building)

- **Dense-first.** MoE top-k routing is non-differentiable; the project trains
  routers by REINFORCE separately. The first trainable core uses an all-Dense
  config (no MoE layers) so the whole forward is differentiable. MoE
  integration is the last phase.
- **Truncated BPTT**, window `T = 64`: unroll the recurrence over the window,
  backprop, detach SSM state at window boundaries. Bounds memory; standard.
- **Untie the embedding.** Embedding and unembedding become separate trainable
  layers (the unembedding reuses the `TrainedReadout` machinery already shipped).
- **Autograd-by-hand, leaf-to-root.** Each layer gets a `backward` that takes
  the upstream grad + cached input and returns the downstream grad while
  accumulating its own `dL/dW`. Build and gradient-check each in isolation
  before composing — a wrong layer backward is then a local, caught failure.

## Phases (each ships its own tests in the same commit)

### Phase 0 — Autograd harness + finite-difference checker
- A reusable gradient checker: given a scalar-loss closure over a param slice
  and an analytic-gradient closure, assert max relative error < ~1e-4.
- The backward API convention every layer implements.
- **Verify:** the checker itself on a hand-written scalar function.
- **Risk:** low. Foundation for everything else.

### Phase 1 — Leaf-layer backwards (each gradient-checked in isolation)
- **1a. TernaryLinear** — add the `dL/dx = Wᵀ·upstream` path (using the scaled
  effective matrix); reuse/verify `accumulate_ste` for `dL/dW`. Check `dL/dx`
  by finite-difference of the f32-latent forward; check `dL/dW` against the STE
  formula.
- **1b. RMSNorm** — `dL/dx` and `dL/dgain` for `y = x/rms(x) · gain`.
- **1c. GLU** — backward through the SwiGLU gate·up nonlinearity.
- **1d. Embedding + unembedding** — lookup-row grad for embed; the readout
  backward (`dL/dlogit = softmax − onehot`, then `dL/dh`, `dL/dW`, `dL/db`) is
  already derived from the probe's logistic fit — reuse it.
- **1e. MLGRU** — backward for the Dense channel-mixer token op.
- **Verify:** a finite-difference gradient check per layer on tiny dims.
- **Risk:** low–medium. Mechanical but must be exact.

### Phase 2 — SSM recurrence backward (BPTT) — the hard one
- Read `src/nn/ssm.rs` forward exactly (selective `dt`/`B`/`C`, `Ā = exp(dt·A)`,
  `D` skip), then derive the through-time backward over a window:
  `dL/dh_t = (y_t path) + Ā_{t+1} ⊙ dL/dh_{t+1}`, with grads to `a_log` (through
  `Ā`), `dt`/`dt_bias` (selective), `B`/`C` (through `x_proj`), `D`, and each
  projection (via Phase 1a).
- **Verify:** finite-difference gradient check on a tiny SSM (small `d_state`,
  short sequence) w.r.t. `a_log`, `dt_bias`, `D`, and every projection's latent
  weights. **This check is the linchpin** — the recurrence is where a subtle
  sign/indexing error hides.
- **Risk:** high. The single most error-prone piece; budget the most care here.

### Phase 3 — Compose the full-model backward + the Stage-0 gate
- A training-mode forward that captures all activations needed for backward per
  timestep (extend `forward_training`).
- Wire the per-layer backwards through the block structure: residual streams sum
  gradients, pre-norm placement, the SSM and channel-mixer sub-blocks, the
  final norm and unembedding.
- **Verify — this is Stage 0 from the run design:** an end-to-end
  finite-difference gradient check on a tiny *dense* model, sampling parameters
  of every type. Until this is green, no training curve is trustworthy.
- **Risk:** medium. Composition bugs surface here; the end-to-end check catches
  them.

### Phase 4 — Training loop + `train` subcommand
- Untie embedding (trainable embed + unembed).
- Truncated-BPTT loop: stream corpus → forward+capture over window `T` → backward
  → accumulate into per-layer `GradBuffer`s → global-norm grad clip → `AdamW`
  step over `LatentWeights` → `re_ternarize` every `N` steps → detach state at
  the window boundary.
- lr warmup + decay; `SeedTree` for all RNGs; a run manifest; checkpoint/resume
  (reuse Phase J).
- `Commands::Train` → `cmd_train`, dense config.
- **Verify:** an integration smoke — loss strictly decreases on a trivially
  learnable input (e.g. a single repeated token), and a re-ternarization
  flip-rate that settles rather than thrashes.
- **Risk:** medium. Mostly orchestration over verified pieces.

### Phase 5 — Stage 1 run: the planted-pattern corpus
- Train a small dense core on `data/synthetic/diagnostic.txt`.
- **Pass/kill (pre-registered in the run design):** next-token NLL on the
  deterministic sections falls below ~0.2 nats; kill if it cannot beat the
  unigram marginal there. The first real evidence that the core learns.
- **Risk:** this is a *run*, not code — it reports whether Phases 0–4 are right.

### Phase 6 — Stage 2: real corpus, f32 control, Bet 1
- Add an f32 (non-ternary) linear variant for the matched control.
- Train ternary + f32 on the real corpus; beat a **bigram** (not just the
  unigram); judge against Part V Bet 1 (≤ 1.25× f32) and Bet 5 (monotone
  held-out).
- Integrate MoE: combine the supervised core gradient with the existing
  REINFORCE router (or a differentiable top-k), then enable MoE layers.
- **Risk:** medium. Bet 1 is the real test of the ternary thesis.

## Verification ladder (the trust chain)

```
Phase 0 checker  →  per-layer checks (1a–1e)  →  SSM-BPTT check (2)
        →  end-to-end gate (3)  →  loss-decreases smoke (4)
        →  Stage-1 known-answer run (5)  →  Bet-1 real run (6)
```

Each rung is cheap and local; a failure is caught at the lowest rung that can
see it. Nothing claims "the core learns" until rung 5, and nothing claims the
ternary bet holds until rung 6.

## Conventions

Tests in the same commit as the code they cover; `SeedTree` for reproducibility;
a manifest per write; byte-identical artifacts from a fixed seed; one logical
change per commit; split-enforcement (`src/util/split_check.rs`) on any
train/holdout boundary.
