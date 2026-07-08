# Path B Core Training Build

This is the next engineering track for `clob`: train the ternary recurrent
core first, then rerun crystallization on a substrate with real signal.

The first real run and reservoir probe argue against treating the current
random core as a useful fixed reservoir. The probe artifact at
`experiments/2026-06-22_first_run/probe_readout.json` reports:

- tied random readout holdout NLL: `8.506331`
- trained bias-only unigram null: `4.916769`
- trained readout, floored against the null: `4.916769`
- contextual gain: `0.000000`
- verdict: `RESERVOIR DEAD`

That does not prove no reservoir variant can work. It is enough to stop
spending first effort on crystallizing corrections over the current frozen
core.

## Starting Point

Already present:

- `src/learn/check.rs`: reusable scalar finite-difference gradient checker.
- `src/learn/grad.rs`: latent f32 weights behind ternary matrices and STE
  gradient accumulation for `dL/dW`; checked linear input gradients; checked
  dense unembedding/cross-entropy/embedding local backwards.
- `src/learn/ssm.rs`: checked one-step and short-window selective SSM
  forward/backward with state-gradient carry.
- `src/learn/optimizer.rs`: AdamW over latent weights with re-ternarization.
- `src/learn/replay.rs`: prioritized replay helpers.
- `CoreModel::decode_step_training`: captures MoE router state for the current
  REINFORCE router update.

Missing:

- ~~`dL/dx` for every trainable layer beyond the checked
  linear/RMSNorm/GLU/MLGRU/readout leaves.~~ **Done** — the full-model
  composition in `src/learn/backprop.rs` produces `dL/dx` through the whole
  dense stack.
- ~~Per-timestep activation capture for truncated BPTT.~~ **Done** — the
  forward tape in `backprop.rs` records every activation the window backward
  needs.
- ~~Full-model training tape and end-to-end tiny-model gradient check.~~
  **Done** — `dense_model_end_to_end_gradient_matches_finite_difference`
  (T=3) and `single_token_window_also_checks` gate it, plus
  `every_trainable_family_has_nonzero_gradient` guards against a vacuous pass.
- ~~Untied trainable readout.~~ **Done in the training path** — `backprop.rs`
  keeps `embed` and `readout` as independent `[vocab × d_model]` tables with
  separate gradients. The deployed inference model still ties them; wiring the
  untied readout into the persisted model is Phase 4.
- ~~A supervised `train` subcommand with manifests and checkpoints.~~ **Done**
  — `clob train` (`src/learn/train.rs`), smoke-gated. **← Phase 5 (planted
  patterns) is next.**

## Locked Direction

Use a dense-first path. MoE top-k routing is non-differentiable and already has
a separate REINFORCE trainer; the first trainable core should use an all-dense
config so the full forward is differentiable.

Use truncated BPTT with a small initial window, for example `T=64`. Detach SSM
state at window boundaries until the gradient checks and planted-pattern run
are stable.

Untie embedding and unembedding. The current tied random table blocks even a
readout-only training path.

## Phase 0: Gradient Check Harness

Create a reusable finite-difference checker for scalar losses over tiny
parameter vectors.

Verify it against a simple hand-written function before using it on model
layers.

Important: finite-difference the differentiable f32-latent surrogate, not the
deployed ternary step function. STE is the bridge from surrogate gradients to
ternary deployment; it is not itself finite-difference smooth.

## Phase 1: Leaf Backwards

Implement and gradient-check each local backward in isolation:

- `TernaryLinear`: `dL/dx = W^T upstream`, plus checked `dL/dW` against the STE
  formula when the mask is one.
- `RMSNorm`: input gradient and gain gradient. **Done locally.**
- `GLU`: backward through gate/up/down projections and activation. **Done
  locally.**
- `Embedding`: row updates for consumed token IDs. **Done locally.**
- `Unembedding`: softmax-cross-entropy gradient, `dL/dh`, `dL/dW`, `dL/db`.
  **Done locally except bias, pending untied readout module.**
- `MLGRU`: dense channel mixer recurrence path. **Done locally for one-step
  recurrence with future state-gradient carry.**

Each layer test should run on tiny dimensions and assert relative error before
composition starts.

## Phase 2: Selective SSM BPTT

This is the hard phase. Read `src/nn/ssm.rs` forward exactly, then derive the
through-time backward for the selective recurrence over a short window.

The check must include:

- projection latent weights. **Done for one step and short window.**
- `a_log`. **Done for one step and short window.**
- `dt_bias`. **Done for one step and short window.**
- `d_param`. **Done for one step and short window.**
- state carry from `t + 1` back to `t`. **Done for one step and short
  window.**

Do not proceed to model training until a tiny SSM finite-difference check is
green. Most plausible silent failures live here.

## Phase 3: Full Dense Model Backward — **Done** (`src/learn/backprop.rs`)

Add a training forward that records every activation needed for a BPTT window.
Compose the block backward with residual-gradient splitting, pre-norm order,
SSM gradients, channel mixer gradients, final norm, and unembedding.

Gate this phase with an end-to-end finite-difference check on a tiny dense
model, sampling parameters from every trainable family.

Shipped: `DenseModel::{forward_window, backward_window}` run truncated BPTT
over a T-token window through the dense stack (`h_mid = x + SSM(norm1(x))`;
`x = h_mid + GLU(MLGRU(norm2(h_mid)))`), carrying a future-state gradient per
layer for both the SSM and MLGRU recurrences, with an untied readout. The
finite-difference gate passes at T=1 (~1%) and T=3 (~1.3%, f32-precision
limited on the smaller through-time gradients); a coverage test asserts every
trainable family is non-trivially exercised so the check can't pass vacuously.
The composition differentiates the f32 latent surrogate; the STE→ternary
bridge stays separate per the trust chain.

## Phase 4: Train Command — **Done** (`src/learn/train.rs`, `clob train`)

Add `clob train` for dense configs:

- stream corpus into BPTT windows
- compute next-token cross entropy
- backprop through the captured window
- clip global gradient norm
- step latent weights with AdamW
- re-ternarize on a fixed cadence
- detach recurrent state at window boundaries
- emit a manifest and checkpoint directories

Smoke gate: train on a trivial repeated-token corpus and require loss to fall
well below the unigram baseline.

Shipped: `train_dense` runs truncated-BPTT windows (state detached at each
boundary) through the Phase 3 backward, clips the global gradient norm, and
steps the f32 latents with a flat AdamW. `clob train` wraps it with `mini`/
`small` dense presets, a trained-dense artifact (latents + effective ternary
weights), a manifest sidecar, and numbered checkpoint dirs. The smoke gate
(`trivial_structured_corpus_beats_unigram`) uses the memory-requiring pattern
`[0,1,0,2]` and asserts the trained NLL falls below the *memoryless* floor
(~0.35), so the recurrence — not just the readout — is proven to train.

Honest finding from the re-ternarization cadence: on `tests/data/eval_corpus.txt`
(byte-level, `mini`), the latent core reaches ~41% of the unigram baseline
(NLL 1.25 vs 3.02), but the naive effective-ternary projection of those latents
sits near baseline (gap ~+1.5–1.8 nats). Pure latent training does not yield
deployable ternary weights; forward-through-ternary STE quantization-aware
training is the refinement that closes that gap, and the gap is exactly the
quantization cost Bet 1 will measure. Phase 5 trains on the planted-pattern
diagnostic corpus (known answers) before crystallization is judged again.

## Phase 5: Planted-Pattern Run

Train a small dense core on `data/synthetic/diagnostic.txt`.

Pass: deterministic sections fall to very low NLL and beat the unigram/bigram
baseline. Kill: the trained core cannot beat those baselines on planted
structure.

Only after this passes should crystallization be evaluated again.

## Phase 6: Real Corpus And Controls

Train ternary and matched f32 controls on the pinned real corpus. The first real
Bet 1 measurement is held-out CE gap versus the f32 control and memory savings
against the same architecture.

Then re-enable crystallization and judge Bet 2 with equal compute:
modules-loaded must reduce held-out NLL versus modules-cleared.

## Trust Chain

Gradient checker -> per-layer checks -> SSM BPTT check -> tiny full-model check
-> trivial-corpus loss drop -> planted-pattern run -> real-corpus f32 control
-> crystallization ablation.

No step should claim the core learns until the planted-pattern run passes.
