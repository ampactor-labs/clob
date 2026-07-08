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

Pending.

