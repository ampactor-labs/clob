# The multi-head critic

Three linear probes over the model's hidden state plus a meta-critic
that keeps the training loop stable.

## The heads

### Head N — novelty (`src/nn/energy.rs:EnergyCritic`)

Predicts expected NLL of the next token from the current hidden
state. Absolute value consumed by the novelty detector for the
notice step. Trained via SGD against per-step NLL.

### Head C — confidence (`src/nn/confidence.rs:ConfidenceHead`)

Same shape as Head N — linear probe, SGD against NLL — but with an
**independent weight vector**. The separate parameters are load-
bearing: Head N's gradient ("I'm surprised") and Head C's gradient
("I'm about to be wrong") are nearly colinear on short horizons and
would cannibalize each other if they shared projection weights.

Consumer: the adaptive decode loop in `CoreModel::decode_step`. When
Head C's detector z-score crosses `adaptive_z` (default 1.0), the
block stack iterates up to `adaptive_max_extra` additional times on
the current buffer before unembedding. Each extra pass refines the
SSM state without re-embedding — the poor-man's pondering loop.

`predict()` clamps at zero since NLL is non-negative; an untrained
head can emit negatives which the detector would then misinterpret.

### Head S — symbolic suitability (planned, Phase P)

Predicts whether a cluster's crystallized matrix will admit a DSL
program under `try_synthesize`. Not shipped yet — today every
cluster is tried via BFS. Head S will gate the BFS cost, skipping
~80% of unproductive searches once trained.

## The meta-critic (`src/nn/confidence.rs:MetaCritic`)

Predicts the absolute error of Head C itself:
`|head.predict(h) - actual_nll|`. Trained via a **target-network
trick**: the label is produced by a frozen snapshot of Head C that
refreshes every `meta_refresh` steps. Without the snapshot, meta
and head parameters couple and their MSEs co-drift.

Consumer: gates adaptive iteration. When meta's prediction at this
hidden state exceeds `meta_unreliable_threshold` (default 2.0),
adaptive iteration is *suppressed* for this token — Head C's
uncertainty signal is itself unreliable here, so acting on it
would waste joules. The suppression count flows into the metrics
window as `meta_suppressed`.

## Training primitives reused

All heads reuse `train_step(hidden, target, lr)` — plain SGD on
MSE. They plug into `src/learn/replay.rs::sample_batch` with
head-specific priority keys (NLL for N, |C_pred - actual| for C,
cluster determinism for S once it lands).

## Novelty detectors as calibration wrappers

Each head wraps its output in a `NoveltyDetector` for adaptive
thresholding. This single abstraction (`src/crystal/detector.rs`)
is the project's standard answer to "should we hand-tune this
threshold?" — no, let the detector calibrate from the signal's
own distribution.

## Where the heads earn their keep

Phase K's `bench-suite` will A/B test every head's consumption
path. Until that lands (and a real corpus is in place), the
signals are uninformative because the underlying model is random
weights. On a trained model:

- **Head N ablation:** novelty gating off → storage explodes, buffer
  fills with uninteresting episodes.
- **Head C ablation:** adaptive-compute off → same J/nat but lower
  accuracy on uncertain tokens.
- **Meta ablation:** no suppression → J/nat rises without accuracy
  gain when Head C miscalibrates.

## Current Pearson numbers

On a synth model + smoke corpus, Pearson(Head C, NLL) ≈ 0. This is
the honest signal that random weights carry no predictive
information. A trained base model should push Pearson > 0.3; if it
doesn't, the hidden-state representation itself is the bottleneck,
not the head.
