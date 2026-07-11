# Phase 6 second half: Bet 2 on the trained dense substrate

## Question

Bet 2 (`ARCHITECTURE.md` Part V): does crystallizing prediction errors into
ternary residual modules reduce held-out loss more than it adds drift, on a
substrate that can actually learn? Every prior crystallization run was on a
random or untrained core (the first real run crystallized zero modules; the
causal track got 7 modules with a −0.31% equal-compute whisper — both on
untrained cores). Phase 6 produced the first trained core. This run points the
crystallization loop at it and applies Bet 2's pre-registered thresholds.

The test is the equal-compute A/B built in `src/eval/bet2_dense.rs` and driven
by `clob crystal-dense`: capture the trained core's own high-error episodes on
the train corpus, crystallize them with the trained readout as the
token-direction table, then measure the holdout with modules loaded vs cleared
in a single forward pass (one decode, two readouts). Modules apply additively
post-final-norm and never feed back into recurrent state, so equal compute is
exact and tokens no module routes to are identical between arms.

## Substrate (pinned)

The Phase 6 QAT representative (`experiments/2026-07-10_phase6_real_qat/`):

- `small` dense, `seed=1`, `window=32`, `lr=2e-3`, `weight_decay=0.0`,
  forward-through-ternary `--qat`, the **step-15000** checkpoint (Phase 6's
  selected representative). Training is deterministic, so training 15000 steps
  reproduces that checkpoint's parameters.
- Evaluated in its deployed **effective-ternary** view (the QAT artifact's
  learned object). The dense trainer now stamps mode into the artifact, so the
  view is auto-selected; the run passes `--ternary` explicitly as belt-and-braces.
- Corpus: pinned Moby-Dick BPE — `data/corpus/{train,holdout}.tokens`, vocab
  3260, holdout unigram baseline `4.776462` (Phase 6).

## Pre-registered primary configuration (the verdict rests on this one)

To avoid multiple-comparisons inflation, exactly one configuration carries the
Bet 2 verdict, fixed here before the run:

- Capture: `error_threshold=0.5`, `max_episodes=4000`, future horizon 8.
- Crystallize: `n_clusters=16`, **causal refinement ON**
  (`causal_horizon=8`) — the project's most developed crystallization method
  (the attractor track's causal-state refinement), so the primary gives Bet 2
  its principled best method rather than the plain k-means baseline.
- Route: `activation_threshold=0.3` (the `CrystalConfig` default used
  everywhere else).
- Evaluate: full holdout (`max_eval_tokens=0`), `window=32`.

## Pre-registered verdict rule

From the `ARCHITECTURE.md` Part V table (unchanged):

Validity precondition (checked first):

- The substrate's cleared holdout NLL must beat the holdout unigram baseline
  `4.776462`. If not, there is nothing for modules to improve and the run is
  **INVALID**.

Bet 2 capability line — `delta_pct = (nll_cleared − nll_loaded)/nll_cleared ×
100` on the primary config, full holdout:

- **Pass** if `delta_pct ≥ 2.0` (modules-loaded held-out NLL ≥2% lower).
- **Kill** if `delta_pct ≤ 0.0` (not lower).
- Otherwise **indeterminate**.

Bet 2 drift line — held-out NLL on inputs no module routes to, loaded vs
cleared:

- Structurally **zero** in this harness: a token no module routes to is scored
  from the identical hidden in both arms (single forward pass), so its two
  NLLs are bit-equal. The drift line (`≤0.5%` worse) therefore passes by
  construction, not by measurement. Reported as PASS-by-construction, and the
  fraction of tokens routed is reported so "zero drift" is not mistaken for
  "modules did nothing everywhere."

The recorded verdict is the pair (capability line, drift line) plus validity.

## Sensitivity grid (reported, NOT verdict-bearing)

To show whether the primary's result is a knife-edge or robust, vary one knob
at a time from the primary and report `delta_pct` for each. These do not change
the verdict; a passing grid cell with a failing primary is an exploratory
signal for a future pre-registered confirmatory run, not a pass.

- `n_clusters ∈ {8, 32}` (primary 16)
- causal refinement OFF (primary ON)
- `activation_threshold ∈ {0.5, 0.7}` (primary 0.3)

Plus one context row on the **f32-latent** substrate (`latent_B` step-15000,
latent view) at the primary crystallization settings — does crystallization
help the f32 core any differently than the ternary one? Context only.

## Pre-registered caveats

Fixed before results:

- One substrate (small, seed 1), one corpus, one crystallization pass. A pass
  or kill is evidence at this configuration, not a scale claim.
- The 2% bar is hard on an already-trained core (NLL ~3.85): the residual
  structure left for modules to capture is small. A kill here means
  "crystallization as implemented adds no net held-out gain on this trained
  ternary core," which is a finding about the loop, not proof no variant can
  work.
- Modules are crystallized from train-corpus errors and applied to holdout;
  the failure mode Bet 2 names (modules overfit their cluster and hurt
  elsewhere) is exactly what the held-out A/B is built to expose.
- Distillation uses the trained readout as the token-direction table (the
  dense analog of the deployed core's tied embed/unembed).

## Results

**Verdict: Bet 2 capability line KILL; drift line PASS-by-construction.**
Machine-readable: `verdict.txt`; receipts: `results.tsv`, `grid.tsv`;
provenance: `runinfo.txt` (run at the pre-registration commit `8cfd7d8`).

The substrate reproduced Phase 6 exactly: cleared holdout NLL `3.797228` is
bit-identical to `experiments/2026-07-10_phase6_real_qat/` qat_B step-15000
(`3.797228`), so the substrate is the validated one and `crystal-dense`'s
cleared arm agrees with `eval-dense --ternary`. Validity precondition holds
(`3.797228 < 4.776462`).

Primary config (n_clusters 16, causal on, activation threshold 0.3): capture
recorded 4000 high-error episodes, clustering produced 20 causal-refined
clusters — and distillation crystallized **zero modules**. With no modules,
loaded == cleared, `delta_pct = 0.0000`, which is `≤ 0` → **KILL**.

The kill is robust, and it has two distinct mechanisms across the grid:

| cell | substrate/view | n_clusters | causal | act | modules | delta_pct |
|:--|:--|--:|:--|--:|--:|--:|
| primary | qat / ternary | 16 | on | 0.3 | 0 | +0.000 |
| nclusters_8 | qat / ternary | 8 | on | 0.3 | 0 | +0.000 |
| nclusters_32 | qat / ternary | 32 | on | 0.3 | 0 | +0.000 |
| causal_off | qat / ternary | 16 | off | 0.3 | 0 | +0.000 |
| act_0p5 | qat / ternary | 16 | on | 0.5 | 0 | +0.000 |
| act_0p7 | qat / ternary | 16 | on | 0.7 | 0 | +0.000 |
| latent_ctx | latent / f32 | 16 | on | 0.3 | 2 | **−3.611** |

Mechanism 1 — the deployed ternary core yields **no distillable structure.**
Every QAT cell (n_clusters 8/16/32, causal on/off, threshold 0.3/0.5/0.7)
crystallized zero modules. Capture and clustering ran (4000 episodes, 16–20
clusters); distillation rejected every cluster at its coherence or MDL gate.
On a core that has actually learned, the tokens it is still surprised by
(error > 0.5) are spread across too many distinct next-tokens to form a
coherent correction direction — the distiller correctly declines to build a
module from incoherent evidence, so there is nothing to integrate.

Mechanism 2 — where modules *do* form, they **hurt.** The f32-latent context
row crystallized 2 modules (from clusters of 57 and 172 episodes, both
avg_error ≈ 1.0 — tokens the core gets flatly wrong) and they lowered held-out
capability by 3.611%, routing on 50.6% of tokens. This is the exact failure
Bet 2 names: modules overfit their train-error cluster and hurt on held-out
data. Corrections learned as "push the hidden toward the cluster's majority
next-token" do not generalize.

Neither branch reaches Bet 2's ≥2% gain. The primary (ternary, verdict-bearing)
kills on zero modules; the latent branch (context) would kill on a −3.6%
regression. The drift line passes by construction: on the ternary substrate no
token is routed at all (0.0% routed, drift trivially zero); the harness scores
untouched tokens from the identical hidden in both arms regardless.

## Conclusion

Bet 2 is falsified at this configuration — the first time it has been testable,
because it is the first trained substrate. On the deployed ternary core,
crystallization of the core's high-error episodes produces nothing distillable
across the whole registered grid; on the f32-latent core it produces modules
that reduce held-out capability by 3.6%. The crystallization loop as
implemented adds no net held-out predictive gain on a trained core.

This is a finding about the loop as built, not a proof that no variant can
work. Two things it directly implicates, as named follow-ups (each needs its
own pre-registered run — do not retrofit this one):

- **Capture starves the clusterer.** Recording only error > 0.5 episodes feeds
  the distiller the hardest, most incoherent tokens — exactly the ones the
  coherence gate rejects. A registered run capturing at a lower error
  threshold, or all tokens, tests whether coherent mid-error structure exists
  that this run never sampled.
- **The distill target overfits.** avg_error ≈ 1.0 clusters crystallized into
  actively harmful corrections. The "push toward readout[actual]" target, or
  the absence of a held-out gate inside distillation, is the suspect.

Both are crystallization-design questions. Bet 2's honest status is now a
measured KILL on a trained core, superseding every prior "instrumented" or
whisper result recorded on random cores.
