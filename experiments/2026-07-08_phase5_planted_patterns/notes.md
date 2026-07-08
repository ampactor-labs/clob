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

**Verdict: PASS.** All five pass conditions fired, and neither kill condition
fired. The full table is `results.tsv`; the machine-readable verdict is
`verdict.txt`.

Summary:

| scope | unigram | bigram | final latent | final ternary | result |
|:--|--:|--:|--:|--:|:--|
| whole corpus | 3.1530 | 3.2121 | **0.2798** | 5.6868 | beats both |
| section 1 periodic phrase | 2.6526 | 3.0119 | **0.2811** | 5.8298 | beats both |
| section 2 shift rule | 2.2886 | 3.9906 | **0.7045** | 3.2969 | beats both |
| section 3 period two | 2.4089 | 3.1175 | **0.7041** | 6.3778 | beats both |
| section 4 high-entropy noise | 3.0748 | 4.5182 | **2.2049** | 6.3328 | diagnostic only |
| section 5 interleaved structure | 2.8413 | 3.5641 | **0.6916** | 5.7953 | beats both |

Registered checks:

- whole-corpus latent NLL was `0.279761`, or `8.9%` of unigram baseline
  (`3.152967`), passing the `<= 75%` line.
- all four structured sections beat both byte unigram and Laplace-smoothed
  byte bigram baselines.
- Section 1, 2, and 3 absolute low-NLL thresholds all passed:
  `0.2811 <= 1.0`, `0.7045 <= 1.5`, `0.7041 <= 1.2`.
- kill conditions were false.

The negative-control Section 4 also improved (`2.2049` vs unigram `3.0748`),
but that was not a pass condition. This run trains and evaluates on the same
diagnostic corpus; Section 4 is therefore a memorization/overfit warning, not
evidence that noise contains meaningful structure.

The ternary projection failed badly (`5.6868` whole-corpus NLL vs latent
`0.2798`). That is not part of the Phase 5 verdict; it repeats the known
finding that pure latent training does not produce a deployable ternary core.
Bet 1 still waits on forward-through-ternary QAT or a matched f32/ternary
control run.

## Conclusion

Path B Phase 5 passes: the dense latent core can learn the planted diagnostic
structure, including the clean periodic, shift, alternation, and interleaved
sections, before crystallization is judged again.

This does **not** validate crystallization. The next honest gate is Phase 6:
train on the real corpus with controls, then rerun causal crystallization
against a trained substrate and judge equal-compute module gain separately.
