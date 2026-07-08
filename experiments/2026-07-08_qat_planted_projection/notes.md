# QAT planted-projection diagnostic

## Question

Phase 5 proved the `small` dense latent core can learn the planted diagnostic
corpus, but its effective ternary projection failed badly (`5.686825`
whole-corpus NLL). This run asks the next narrower question: does
forward-through-ternary STE-QAT close that projection gap on the same planted
corpus?

This run does **not** judge Bet 1 on the real corpus, and it does not judge
crystallization. It only tests whether the new `clob train --qat` path can
make the deployable effective-ternary weights learn the planted structure.

## Pre-registered setup

- Corpus: `data/synthetic/diagnostic.txt`
- Tokenizer: built-in byte-level tokenizer (`vocab=260`)
- Dense config: `small` (`d_model=64`, `n_layers=3`)
- Seed: `1`
- Window: `32`
- Learning rate: `2e-3`
- Weight decay: `0.0`
- Clip norm: `1.0`
- Steps: `8000`
- Training mode: `--qat`
- Ternary projection cadence: every `1000` steps

## Pre-registered verdict rule

**Pass** if all are true:

- whole-corpus effective-ternary QAT NLL is at most `0.75 * unigram NLL`;
- effective-ternary QAT NLL beats both byte unigram and Laplace-smoothed byte
  bigram baselines on structured Sections 1, 2, 3, and 5;
- whole-corpus effective-ternary QAT NLL is at most half the Phase 5
  pure-latent effective-ternary NLL (`5.686825`);
- effective-ternary QAT NLL is at most `1.0` on Section 1;
- effective-ternary QAT NLL is at most `1.5` on Section 2;
- effective-ternary QAT NLL is at most `1.2` on Section 3.

**Kill** if any are true:

- whole-corpus effective-ternary QAT NLL is not below whole-corpus unigram NLL;
- fewer than three of the four structured sections beat both baselines;
- whole-corpus effective-ternary QAT NLL is not below the Phase 5 pure-latent
  effective-ternary NLL.

Anything between those lines is **indeterminate**.

## Results

**Verdict: PASS.** All six pass conditions fired, and none of the kill
conditions fired. The full table is `results.tsv`; the machine-readable
verdict is `verdict.txt`.

Summary:

| scope | unigram | bigram | Phase 5 projection | QAT ternary | QAT latent | result |
|:--|--:|--:|--:|--:|--:|:--|
| whole corpus | 3.1530 | 3.2121 | 5.6868 | **0.0938** | 6.6810 | beats both |
| section 1 periodic phrase | 2.6526 | 3.0119 | 5.8298 | **0.1473** | 6.5564 | beats both |
| section 2 shift rule | 2.2886 | 3.9906 | 3.2969 | **0.3539** | 6.3994 | beats both |
| section 3 period two | 2.4089 | 3.1175 | 6.3778 | **0.4788** | 6.6397 | beats both |
| section 4 high-entropy noise | 3.0748 | 4.5182 | 6.3328 | **0.9699** | 7.2986 | diagnostic only |
| section 5 interleaved structure | 2.8413 | 3.5641 | 5.7953 | **0.2968** | 6.7920 | beats both |

Registered checks:

- whole-corpus effective-ternary QAT NLL was `0.093766`, or `3.0%` of unigram
  baseline (`3.152967`), passing the `<= 75%` line.
- all four structured sections beat both byte unigram and Laplace-smoothed
  byte bigram baselines.
- whole-corpus effective-ternary QAT NLL was `1.6%` of the Phase 5 pure-latent
  projection NLL (`5.686825`), passing the `<= 50%` gap line by a wide margin.
- Section 1, 2, and 3 absolute low-NLL thresholds all passed:
  `0.1473 <= 1.0`, `0.3539 <= 1.5`, `0.4788 <= 1.2`.
- kill conditions were false.

The important caveat is operational: the QAT artifact's latent path is bad
(`6.681042` whole-corpus NLL). The learned object is the effective ternary
projection, so QAT artifacts must be judged with `eval-dense --ternary` unless
the artifact schema later records training mode and selects the correct path
automatically.

The negative-control Section 4 also improves (`0.969880` vs unigram `3.074773`).
As in Phase 5, this run trains and evaluates on the same diagnostic corpus, so
that is a memorization/overfit warning rather than evidence that noise contains
meaningful structure.

## Conclusion

The forward-through-ternary QAT path closes the planted diagnostic projection
gap. On this corpus, the deployable effective-ternary weights learn the planted
structure better than the dense latent artifact did in its own latent path.

This does **not** validate Bet 1. The next honest gate is a real-corpus QAT
control against a matched f32/latent baseline, with held-out evaluation and
memory accounting.
