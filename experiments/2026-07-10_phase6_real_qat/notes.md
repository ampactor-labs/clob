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

**Verdict: PASS on both registered Bet 1 lines.** The validity precondition
held, the capability gap fired the pass condition, no kill condition fired.
Machine-readable verdict: `verdict.txt`; per-run receipts: `results.tsv`,
`curves.tsv`; provenance: `runinfo.txt` (run at the pre-registration commit
`92b1af6`).

All four runs selected their step-15000 checkpoint by the registered curve
rule — the same best-step the Bet 7 sweep found:

| run | mode | lr | wd | selected | curve nll (4k) | full holdout NLL |
|:--|:--|--:|--:|--:|--:|--:|
| latent_A | latent | 5e-3 | 0.01 | 15000 | 3.9197 | 3.8838 |
| latent_B | latent | 2e-3 | 0.0 | 15000 | 3.8441 | **3.8016** |
| qat_A | qat | 5e-3 | 0.01 | 15000 | 3.8831 | 3.8420 |
| qat_B | qat | 2e-3 | 0.0 | 15000 | 3.8501 | **3.7972** |

Registered checks:

- Validity: the f32 representative (`latent_B`, `3.801559`) beats the
  holdout unigram baseline (`4.776462`). OK.
- Capability gap: `3.797228 / 3.801559 = 0.9989×`, against pass `≤ 1.25×`
  and kill `> 1.5×`. **PASS.** Read this as capability parity: a 0.004-nat
  difference on one seed is noise, not "ternary beats f32."
- Memory, core families: `642,048 / 50,736 = 12.65×` smaller, against pass
  `≥ 10×` and kill `< 8×`. **PASS.** The pre-registration's hand arithmetic
  above says `12.66×`; the scripted derivation in `run.sh` gives `12.65×`
  (12.6547) — a rounding slip in the hand calculation, same conclusion. The
  whole-model context ratio is `1.34×` as pre-computed.

The context rows sharpen what QAT bought. The f32 representative's naive
effective-ternary projection measures `6.166165` on the full holdout — worse
than the unigram baseline. The QAT arm's deployed ternary measures
`3.797228`. On this corpus, forward-through-ternary training does not narrow
the deployment gap; it removes it.

Replication: `latent_A` (the Bet 7 recipe) reproduces the Bet 7 sweep's
step-15000 curve point (`3.9197` here vs `3.92` recorded 2026-07-08).
Recipe B (`lr 2e-3, wd 0`) beat recipe A for **both** arms, which is why the
grid ran both recipes on both arms.

Regime context (no verdict attached): λ₁ of the latent representative is
`−0.0241` (state-memory horizon ≈ 41 tokens); the QAT representative's
ternary dynamics measure `−0.0675` (≈ 15 tokens). Both sit far nearer the
edge than the `−0.28` the Bet 7 sweep recorded for its best checkpoint under
recipe A — a recipe→regime effect this run was not designed to judge. It
goes on the Bet 7 open-thread pile.

## Conclusion

Bet 1's first real measurement passes at this configuration: on the pinned
real corpus, with held-out evaluation, the deployable effective-ternary core
trained by QAT costs no measurable capability against its matched f32
control, and the ternary core families are 12.65× smaller resident than
f32.

What this does **not** say, kept in one place: one corpus, one seed, one
small architecture (~1.3 epochs, batch-1 AdamW); both arms keep f32
embedding/readout tables (72% of parameters), so this is parity for the
recurrent core's ternarization, not for a fully-ternary model; the
whole-model memory ratio at this width is 1.34×. Bet 1 moves from untested
to first-pass-at-small-scale — validation at target scale still requires
larger width, more seeds, and a second corpus.

Next honest gates: the second half of Phase 6 — re-enable crystallization on
this trained substrate and judge Bet 2's equal-compute line — plus the named
follow-ups: stamp training mode into the dense artifact schema (the latent
view of a QAT artifact is still a footgun), and the long-range-dependency
corpus test for Bet 7.
