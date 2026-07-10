# Phase 6: real-corpus QAT vs matched f32 controls

## Question

The QAT planted-projection diagnostic proved `clob train --qat` can make the
deployable effective-ternary weights learn planted structure. Phase 6 asks the
question Bet 1 actually stakes: on the pinned real corpus, with held-out
evaluation, how much capability does the ternary deployment cost against a
matched f32 control of the same architecture, and what does it buy in resident
weight memory?

This is the first real Bet 1 measurement. It is judged by the pre-registered
Bet 1 thresholds in `ARCHITECTURE.md` Part V, at this configuration. A pass
here is a first data point at `small` scale (d=64, one corpus), not a
validation of Bet 1 at target scale. This run does not judge crystallization
(that is the second half of Phase 6, registered separately once a trained
substrate exists).

## Pre-registered setup

Corpus (pinned 2026-06-22, unchanged):

- Train: `data/corpus/train.tokens` — 486,856 tokens, vocab 3260
  (`train.txt` sha256 `fefc842c...0815975a`).
- Holdout: `data/corpus/holdout.tokens` — 52,273 tokens
  (`holdout.txt` sha256 `5656b852...01c9e37b`).
- Tokenizer: `data/corpus/tokenizer.bin`.
- Split discipline: the existing Phase L contamination gate.

Model and optimizer, identical across all four runs except where stated:

- Dense config: `small` (`d_model=64`, `n_layers=3`, `d_inner=128`,
  `n_heads=4`, `d_state=16`)
- Seed: `1`
- Window: `32`
- Clip norm: `1.0`
- Steps: `20000` (~1.3 epochs of 15,214 windows)
- Checkpoints: every `2500` steps

The grid is two arms × two recipes. Prior registered runs used different
optimizer recipes per mode (the Phase 5 / Bet 7 latent runs used
`lr=5e-3, wd=0.01`; the QAT planted run used `lr=2e-3, wd=0.0`), so a
single-recipe comparison would confound representation with recipe. Both
recipes run on both arms, and each arm is judged at whichever of its two
recipes evaluates better:

| run | mode | lr | weight decay |
|:--|:--|--:|--:|
| latent_A | latent | 5e-3 | 0.01 |
| latent_B | latent | 2e-3 | 0.0 |
| qat_A | `--qat` | 5e-3 | 0.01 |
| qat_B | `--qat` | 2e-3 | 0.0 |

`latent_A` exactly replicates the Bet 7 training run (same seed, corpus,
config, recipe), so its held-out curve doubles as an internal replication
check against `experiments/2026-07-08_bet7/sweep.tsv`.

Evaluation protocol:

- Curve: each run evaluates init (step 0), every checkpoint, and the final
  artifact on the holdout with `--max-tokens 4000`, `--window 32`, in the
  arm's **deployed view**: latent view for latent runs, `--ternary` for QAT
  runs (a QAT artifact's latent view is not the learned object).
- Selection: per run, the artifact with the lowest curve NLL; ties go to the
  lower step.
- Judgment: each run's selected artifact is evaluated on the **full holdout**
  (`--max-tokens 0`). Each arm's representative is its lower full-holdout
  NLL across the two recipes.
- The Bet 1 capability gap is
  `gap = (QAT arm representative ternary NLL) / (latent arm representative latent NLL)`.

Context measurements (reported, not part of the verdict):

- λ₁ regime of each arm's representative artifact
  (`clob regime --dense`, `--ternary` for the QAT arm; 2000 tokens,
  warmup 256, seed 1) — extends the Bet 7 record.
- The latent arm representative's effective-ternary projection on the full
  holdout — the no-QAT deployment gap, for comparison with the planted-corpus
  finding.

## Memory accounting (deterministic, computed before the run)

Bet 1's memory line is judged on the **ternary-eligible recurrent-core
matrix families** — the nine per-layer matrices `ternarize_matrix_families`
projects (SSM in/x/out projections, MLGRU w_f/w_c/w_o, GLU gate/up/down) —
encoded as the deployed `TernaryMatrix` representation actually implemented
in `src/tensor/ternary.rs`: 2 bits per trit, 32-row tiling (padding counted),
plus one f32 scale per row. This scope is fixed now, before the run, on Bet
1's own wording: the bet claims a "sub-2-bit-per-weight recurrent core." The
embedding and readout tables are not part of the recurrent core and are not
ternarized by the current architecture.

At `small`/vocab-3260 the numbers are fully determined by the shapes, so they
are computed here rather than measured later:

- Core families: 160,512 params (53,504 × 3 layers). f32: 642,048 bytes.
  Deployed ternary: 41,472 packed + 9,264 scale bytes = 50,736 bytes.
  **Ratio: 12.66× smaller.**
- Whole model (embed + readout + norms + biases + SSM scalars kept f32):
  579,032 params, f32 2,316,128 bytes; deployed 1,724,816 bytes.
  **Ratio: 1.34×** — reported as context.

Stated plainly before the result: the whole-model ratio at this config fails
the ≥10× line arithmetically, because the un-ternarized vocab tables are
417,280 of 579,032 params at d=64. That is a real, open engineering limit of
the current design at small width (embed/readout ternarization, or larger d,
changes it); the registered line judges the representation bet on the core it
claims. Both numbers are reported so the record can't hide the distinction.

## Pre-registered verdict rule

Validity precondition (checked first):

- The latent arm's representative full-holdout NLL must be below the holdout
  unigram baseline that `eval-dense` itself reports (`4.776462` on the full
  holdout). If not, the run is **INVALID** — the f32 control failed to train,
  and Bet 1's gap is not judged against a broken control.

Bet 1 capability line (from the ARCHITECTURE.md Part V table):

- **Pass** if `gap <= 1.25`.
- **Kill** if `gap > 1.5`.
- Otherwise **indeterminate**.

Bet 1 memory line (same table, scope fixed above):

- **Pass** if the core-family ratio is `>= 10×` (it is 12.66× by
  construction at this config).
- **Kill** if `< 8×`.

The recorded verdict is the pair (capability line, memory line), plus the
validity precondition. No other comparison in this run carries a verdict.

## Pre-registered caveats

Fixed before results, applying regardless of outcome:

- One corpus (Moby-Dick BPE), one architecture, one seed, batch-1 AdamW,
  ~1.3 epochs. A pass or kill here is evidence at this configuration, not a
  scale claim in either direction.
- Selection uses a 4000-token holdout curve; judgment uses the full holdout.
  A noisy curve could select a slightly sub-optimal checkpoint, symmetrically
  for both arms.
- The QAT artifact's latent view is known-bad (schema does not yet record
  training mode); every QAT evaluation in this run goes through `--ternary`.
- The dense trainer's readout is untied; the deployed CoreModel ties
  embed/readout. The dense artifact is judged as its own deployment object;
  bridging to CoreModel is out of scope here.

## Results

(To be filled in by the run. The verdict rule above is frozen as of this
pre-registration; if a threshold or scope turns out to be wrongly chosen, it
changes in a separate commit that does not also report this run.)
