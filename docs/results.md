# Results

Every number on this page is copied from the files each experiment committed
under `experiments/` (`verdict.txt`, `results.tsv`, `sweep.tsv`, `notes.md`).
Each run was pre-registered: its pass and kill thresholds were fixed before it
ran, in `ARCHITECTURE.md` Part V (Bets 1 to 5) or `ATTRACTOR.md` Part V
(Bets 6 to 8). The README's Benchmarks section is a summary of this page.

## Setup

- **Loss** is mean negative log-likelihood per token in nats (lower is
  better). On the real corpus it is measured on the held-out split: the last
  10% of *Moby-Dick* by line, 52,273 tokens, vocabulary 3,260. A unigram model
  that knows only token frequencies scores 4.7765 there.
- **Model.** The dense `small` preset of `clob train`: d_model 64, 3 layers,
  d_inner 128, 4 heads, d_state 16, 579,032 parameters, seed 1, window 32,
  clip norm 1.0, 20,000 steps (about 1.3 epochs) with checkpoints every 2,500.
  Each run's checkpoint is selected on a 4,000-token held-out curve and then
  judged on the full held-out split. All Phase 6 runs selected step 15,000.
- **Two views of one model.** Training keeps f32 "latent" weights. The
  deployed view rounds the recurrent-core matrices to ternary. `--qat` trains
  through the ternary view, so its ternary view is the learned object.
- **Hardware** is not recorded in the run files.
- **Reproducing.** Each directory's `run.sh` repeats its run after you set
  `REPO` (and for Phase 6 the work directory) to your own paths and build the
  corpus with `scripts/acquire_corpus.sh` and
  `N_MERGES=3000 scripts/tokenize_corpus.sh`. Exact numbers need the original
  corpus bytes: a download on 2026-09-26 no longer matched the pinned
  checksums and gave 475,133 train and 50,395 holdout tokens.

## Bet 1: a ternary core keeps its capability

Source: `experiments/2026-07-10_phase6_real_qat/`. Four runs: latent and QAT,
each with two optimizer recipes (A: lr 5e-3, weight decay 0.01; B: lr 2e-3,
weight decay 0). Each arm is judged by its better recipe, which was B for both.

| Run | Mode | Recipe | Full held-out loss |
|:--|:--|:--|--:|
| latent_A | f32 latent | A | 3.8838 |
| latent_B | f32 latent | B | 3.8016 |
| qat_A | QAT, ternary view | A | 3.8420 |
| qat_B | QAT, ternary view | B | 3.7972 |

| Measurement | Result | Pass | Kill | Verdict |
|:--|:--|:--|:--|:--|
| Loss ratio, ternary QAT vs matched f32 | 3.7972 / 3.8016 = 0.9989 | ≤ 1.25 | > 1.5 | pass |
| Recurrent-core weight memory, f32 vs ternary | 642,048 / 50,736 bytes = 12.65× | ≥ 10× | < 8× | pass |
| Whole-model memory, f32 vs deployed (context) | 2,316,128 / 1,724,816 bytes = 1.34× | | | not judged |
| f32 control rounded to ternary without QAT (context) | 6.1662, worse than unigram | | | not judged |

The memory line covers the nine ternary matrix families of the recurrent core,
stored as 2 bits per weight with 32-row tiling and one f32 scale per row. The
embedding and output tables stay f32 in both arms and hold 417,280 of the
579,032 parameters, which is why the whole model is only 1.34× smaller. A
0.004-nat difference on one seed is parity; it does not show ternary beating
f32.

## Bet 2: crystallization gives a net gain

Source: `experiments/2026-07-10_phase6_bet2/`. `clob crystal-dense` captures
the trained QAT core's high-error episodes (error above 0.5, up to 4,000) on
the train split, crystallizes them, and scores the held-out split with modules
loaded and cleared in one forward pass, so compute is equal. The primary
configuration is 16 clusters, causal refinement on, activation threshold 0.3.

| Cell | View | Clusters | Causal | Threshold | Modules | Loss change |
|:--|:--|--:|:--|--:|--:|--:|
| primary | ternary | 16 | on | 0.3 | 0 | 0.000% |
| nclusters_8 | ternary | 8 | on | 0.3 | 0 | 0.000% |
| nclusters_32 | ternary | 32 | on | 0.3 | 0 | 0.000% |
| causal_off | ternary | 16 | off | 0.3 | 0 | 0.000% |
| act_0p5 | ternary | 16 | on | 0.5 | 0 | 0.000% |
| act_0p7 | ternary | 16 | on | 0.7 | 0 | 0.000% |
| latent_ctx | f32 latent | 16 | on | 0.3 | 2 | 3.611% worse |

Pass needed at least 2% lower loss; kill was no improvement. **Verdict: kill.**
On the ternary core every cluster failed the coherence or description-length
gate, so no modules formed. On the f32 core two modules formed, routed on
50.6% of tokens and made held-out loss worse. The drift line (loss on tokens no
module touches) passes by construction, since those tokens are scored from
the same hidden state in both arms.

## Bet 7: the core's memory tracks its capability

Source: `experiments/2026-07-08_bet7/`, one f32 latent `small` run with recipe
A, each checkpoint measured on the held-out split. λ₁ is the largest Lyapunov
exponent from `clob regime`: how fast the state forgets a small perturbation.
Values nearer 0 mean longer memory; the memory horizon is about 1/|λ₁| tokens.

| Step | Held-out loss | λ₁ | Horizon (tokens) |
|--:|--:|--:|--:|
| 0 (random) | 8.2907 | −0.3467 | 2.9 |
| 2,500 | 4.3264 | −0.1116 | 9.0 |
| 5,000 | 4.2188 | −0.1937 | 5.2 |
| 7,500 | 4.1866 | −0.1217 | 8.2 |
| 10,000 | 4.0549 | −0.1122 | 8.9 |
| 12,500 | 4.0509 | −0.1879 | 5.3 |
| 15,000 | 3.9197 | −0.2810 | 3.6 |
| 17,500 | 4.0501 | −0.2533 | 3.9 |
| 20,000 | 3.9814 | −0.3321 | 3.0 |

Spearman ρ between λ₁ and capability is −0.27 over all nine points and −0.81
over the eight trained checkpoints. Pass needed ρ ≥ 0.6 and a best checkpoint
with λ₁ between −0.5 and 0.05; kill was |ρ| < 0.2 or a best checkpoint below
λ₁ = −1. The literal verdict is indeterminate, but the
hypothesis that capability rises as the state nears λ₁ = 0 is falsified: the
best checkpoints forget fastest.

## Earlier runs

| Run | Result | Source |
|:--|:--|:--|
| Reservoir probe on the random core | Trained readout gains 0.000 nats over a unigram null | `docs/experiments/reservoir-probe.md` |
| Phase 5, planted patterns, f32 latent | Loss 0.2798 vs unigram 3.1530 and bigram 3.2121; its ternary rounding scores 5.6868 | `experiments/2026-07-08_phase5_planted_patterns/` |
| QAT on the same planted corpus | Ternary view loss 0.0938; all 4 structured sections beat both baselines | `experiments/2026-07-08_qat_planted_projection/` |
| Causal refinement on a random core | 7 modules from 318 episodes where state-only clustering found 0; loss 7.0494 to 7.0273 | `ATTRACTOR.md`, Receipts |
