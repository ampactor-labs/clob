# The crystallization loop

The heart of the architecture. Turns prediction errors into compiled
ternary modules, one cluster at a time, and discards the raw episodes
once the pattern is captured.

## The eight steps

```
 experience → notice → buffer → distill →
 crystallize → compile → integrate → forget
```

### 1. Experience — `src/model/stack.rs:decode_step`

Each token yields a `hidden_state: [f32; d_model]`, logits over the
vocabulary, and an energy scalar from Head N. Zero allocations on
the hot path.

### 2. Notice — `src/crystal/detector.rs:NoveltyDetector`

Adaptive z-score threshold over the energy distribution. A running
EMA of mean and variance; `z = (energy - mean) / std`. Tokens with
`z > z_threshold` (default 1.0) get marked novel. Warmup suppresses
firing for the first 10 observations while statistics bootstrap.

### 3. Buffer — `src/memory/ring.rs:EpisodicMemory`

Novel episodes (`Episode { timestamp, context, hidden_state, energy,
predictions, actual_token, prediction_error, consumed }`) go into a
memory-mapped ring buffer. Consumed flag lets crystallization mark
episodes as "digested" without deleting them until the ring rolls
over.

### 4. Distill — `src/crystal/distill.rs`

Cluster coherence + MDL gate. Clusters whose `actual_token`
distribution has Shannon entropy > 2.0 bits are rejected (mixed
targets average corrections to noise). Passing clusters produce a
`DistilledPattern` — avg_input, avg_correction, domain signature.
The MDL gate uses Gaussian likelihood-ratio savings against batch
variance, not the unit-mismatched math the early version had
(fixed in commit `827c890`).

### 5. Crystallize — `src/crystal/crystallize.rs`

Rank-1 outer product `avg_correction × avg_input / ||avg_input||²`,
ternarized row-wise via per-row absmean scale. Returns a
`CrystalModule { weight: TernaryMatrix, domain_signature,
symbolic_hint: Option<Program>, ... }`. Phase E's `try_synthesize`
attempts to reduce the matrix to a short DSL program; on success,
`symbolic_hint` is populated.

### 6. Compile — `src/compile/*`

IR → x86-64 → ELF `.so` → `dlopen`. Scalar emitter today; AVX2
planned (Phase O). A 5-op IR — `Accumulate`, `Scale`, `Store` — is
enough for the current scalar path; richer vocabulary arrives with
the DSL expansion (Phase V).

### 7. Integrate — `src/model/stack.rs:apply_crystal_modules`

After the block stack and final norm, each installed module is
evaluated: if `cosine(hidden, domain_signature) > activation_threshold`,
the module's output is added to the hidden, scaled by the similarity.
Soft mixture-of-experts, but experts are *grown*, not trained jointly.

### 8. Forget

Episodes consumed by a successful crystallization get
`mark_consumed`'d in the ring buffer. The ring advances. RAM is
freed. If the bet holds, the system is simultaneously more
knowledgeable *and* lighter. If it doesn't, that's visible as a
rise in held-out CE after forgetting — one of Part V's
falsifiable bets.

## Reusable utilities

- `NoveltyDetector` (`src/crystal/detector.rs:30-99`) — used beyond
  Head N. Wrap around any scalar signal where an adaptive threshold
  beats a hand-tuned constant.
- `sample_batch` (`src/learn/replay.rs:8-45`) — prioritized replay
  over episodes, parameterizable over the priority key.
- `compute_correction` (`src/learn/replay.rs:51-62`) — centroid-delta
  helper.

## What's aspirational, what's real

- **Real:** the full loop runs end-to-end on the smoke corpus; 50+
  tests cover each stage.
- **Aspirational until Phase O lands:** the "32 ops/clock" claim in
  ARCHITECTURE.md. The scalar emitter delivers a fraction of it.
- **Aspirational until a real corpus runs (Phase L + first-run):**
  that the loop *net-reduces* J/nat over real training, not just
  on synthetic data.
