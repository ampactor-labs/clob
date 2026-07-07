# Attractor-track evidence (2026-07-04)

The receipts behind `ATTRACTOR.md`. Every number in that document's Receipts
block traces to a file here, the same way the 2026-06-22 reservoir probe is
backed by `probe_readout.json`. Regenerate with the commands below against a
release build; the model is the random small core at `--seed 42` (no training
— these measure the frozen substrate and the causal partition, not learning).

## Regime: the core's dynamical regime

`regime.toml` + `regime.toml.manifest.toml` — largest Lyapunov exponent of
the token-driven state dynamics, random small core, `eval_corpus.txt` drive.

```
clob regime --model seed.clob --corpus tests/data/eval_corpus.txt \
    --tokens 2000 --warmup 256 --seed 1 --out regime.toml
```

λ₁ = −1.2269 nats/token at the default eps (1e-4); memory horizon 0.815
tokens. The state erases a perturbation in under one token — the mechanism
behind the reservoir probe's zero contextual gain.

`regime_eps_sweep.txt` — λ₁ across probe scales eps ∈ {1e-3, 1e-4, 1e-5}:
−1.4880 / −1.2269 / −1.5464. The *regime* is stable (contractive, horizon
< 1 token at every scale); the *magnitude* is not (it swings ~26% and
non-monotonically). Read λ₁ as "deeply contractive, sub-one-token horizon,"
not as a precise constant — the sign is the load-bearing fact.

## Causal distill: futures vs geometry

Same 318 episodes ingested from `data/synthetic/diagnostic.txt`, crystallized
two ways.

```
clob crystal --memory-dir episodes --modules-dir mods_state  --model seed.clob
clob crystal --memory-dir episodes --modules-dir mods_causal --model seed.clob --causal
```

- `crystal_state_only.txt` — 8 clusters, **0 modules**. Hidden-state k-means
  blends the planted futures; the coherence gate rejects the blend. As in
  every prior run of this project.
- `crystal_causal.txt` — 8 state clusters refined to **25**, **7 modules**
  crystallized, **121 episodes consumed**. The loop's Forget step firing on
  real pipeline flow.

## A/B: do the modules earn their keep

Equal compute, 1945 held-out predictions on the diagnostic corpus.

- `eval_modules_cleared.json` — mean NLL 7.04935 (0 modules).
- `eval_modules_causal.json` — mean NLL 7.02733 (7 modules loaded).

Δ = −0.31%. A whisper, far under Bet 2's 2% pass line, on a core that cannot
represent context. Reported because it is the loop's first non-zero signal
from real flow, not because it validates anything. Bet 6 (`ATTRACTOR.md`)
judges the causal partition for real on a *trained* core.

## Provenance

Measured at the commit that added the horizon-aware causal merge. An earlier
merge compared only the one-step next-token distribution and silently
recombined multi-token splits, suppressing modules (it reported 4, not 7);
the fix made merge honor the split horizon. If you reproduce and see 4, you
are on the pre-fix commit.
