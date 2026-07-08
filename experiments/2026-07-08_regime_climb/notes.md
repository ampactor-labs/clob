# λ₁ climbs as the core learns (2026-07-08)

The first look at Bet 7 (`ATTRACTOR.md`): does the dynamical regime track
capability? A `mini` dense core (d=16, L=2) trained on `tests/data/eval_corpus.txt`
(byte-level, 1589 tokens), measured before, during, and after training on the
same drive corpus with `clob regime --dense`.

```
clob train  --corpus eval_corpus.txt --config mini --window 16 --steps N --lr 3e-3
clob regime --dense <artifact> --corpus eval_corpus.txt --tokens 2000 --warmup 256 --seed 1
```

| core          | train NLL (/unigram 3.02) | λ₁ (nats/token) | memory horizon | report |
|:--------------|:--------------------------|:----------------|:---------------|:-------|
| random init   | —                         | −0.398          | 2.5 tokens     | `regime_init.toml` |
| 2500 steps    | 1.36 (45%)                | −0.163          | 6.1 tokens     | `regime_mid.toml` |
| 6000 steps    | 0.73 (24%)                | −0.112          | 8.9 tokens     | `regime_trained.toml` |

As the core gains predictive capability (NLL falls 3.02 → 0.73), λ₁ rises
monotonically toward zero and the state-memory horizon grows 3.6×. The core
leaves deep contraction as it learns to hold context — exactly the mechanism
the attractor thesis predicts.

## Read this carefully

- **Same architecture throughout.** All three points are the *mini dense* core
  (d=16, f32 latent). The −0.398 init is **not** comparable to the −1.23 the
  random d=128 *ternary* CoreModel measured in `2026-07-04_attractor/`;
  different model. The valid claim is the *within-family* before→after climb.
- **This is a first data point, not Bet 7.** Bet 7 pre-registers a Spearman ρ
  between λ₁ and held-out capability across ≥5 checkpoints on a real corpus
  (`ATTRACTOR.md` Part V). Three points on a 1.5 KB byte corpus with batch-1
  window SGD are a preview: consistent with Bet 7, not a test of it. The
  instrument (`clob regime --dense`) is what makes the full test runnable.
- **Ternary preserves the regime, not the readout.** The effective-ternary
  projection of the trained latents (`regime_trained_ternary.toml`) measures
  λ₁ = −0.128, horizon 7.8 tokens — almost the same regime as the latent core
  (−0.112, 8.9), even though its *loss* sits ~1.5 nats worse. Quantization
  costs accuracy at the readout far more than it moves the recurrent dynamics.

The `.dense` artifacts are training byproducts (regenerable from the commands
above) and are not committed; the regime reports + manifests are.
