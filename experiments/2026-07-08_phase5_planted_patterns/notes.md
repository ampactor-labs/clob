# Phase 5 planted-pattern dense training

## Question

Path B Phase 4 proved the dense recurrent trainer on a tiny structured unit
test. Phase 5 asks the next larger question: can a `small` dense latent core
learn the known planted structure in `data/synthetic/diagnostic.txt` before
crystallization is judged again?

This run does **not** judge Bet 1, Bet 2, Bet 6, or Bet 7. It is a training
gate: if the dense latent core cannot beat simple local baselines on planted
structure, crystallization should not be re-evaluated on top of it.

## Pre-registered setup

- Corpus: `data/synthetic/diagnostic.txt`
- Tokenizer: built-in byte-level tokenizer (`vocab=260`)
- Dense config: `small` (`d_model=64`, `n_layers=3`)
- Seed: `1`
- Window: `32`
- Learning rate: `5e-3`
- Weight decay: `0.01`
- Clip norm: `1.0`
- Steps: `8000`
- Ternary projection cadence: every `2000` steps, reported only as a
  diagnostic.

## Pre-registered verdict rule

**Pass** if all are true:

- whole-corpus final latent NLL is at most `0.75 * unigram NLL`;
- final latent NLL beats both byte unigram and Laplace-smoothed byte bigram
  baselines on structured Sections 1, 2, 3, and 5;
- final latent NLL is at most `1.0` on Section 1;
- final latent NLL is at most `1.5` on Section 2;
- final latent NLL is at most `1.2` on Section 3.

**Kill** if either is true:

- whole-corpus final latent NLL is not below the whole-corpus unigram NLL;
- fewer than three of the four structured sections beat both baselines.

Anything between those lines is **indeterminate**.

The bigram baseline is byte-level Laplace smoothing with `alpha=1.0`, trained
and evaluated on the same scope. This is intentionally a local-pattern
baseline; Phase 5 should beat it on planted structure before the core is used
as a substrate for crystallization.

## Results

Pending.

