# Symbolic crystallization

Phase E added the ability to recognize when a crystallized ternary
matrix is reducible to a short DSL program — and to attach the
program as a `symbolic_hint` on the module.

## Why it matters

A crystallized d×d ternary matrix serializes to roughly d²/4 bytes
(~4 KB for d=128). A 6-op DSL program serializes to ~36 bytes. When
the matrix happens to encode a simple permutation-like
transformation, the storage compression ratio is two orders of
magnitude — and the MDL gate ensures we only keep programs that
actually shorten the encoding.

This is the literal `intelligence = compression` thesis playing out
at the module level.

## The DSL

Five primitives (`src/crystal/synth.rs:Op`):

- `Identity` — the identity matrix.
- `Shift(k: u16)` — cyclic permutation: `output[i] = input[(i+k) % d]`.
- `Negate` — flip all signs.
- `Reverse` — `output[i] = input[d-1-i]`.
- `Mask(bits: u64)` — diagonal with bit `i` masked on/off (low 64 bits
  meaningful).

Every primitive's matrix lives in {−1, 0, +1}, and their product is
also in {−1, 0, +1}. **This is the load-bearing trick:** the DSL is
*ternary-reducible*, so program equivalence reduces to matrix
equality and the BFS search doesn't need any program-equivalence
theory.

## Synthesis (`src/crystal/synth.rs:find_program`)

BFS over primitive sequences up to `max_depth` (default 4). For each
candidate sequence, materialize the product matrix and compare to
the target. First match wins.

At depth 4 with ~30 primitives (identity, negate, reverse, Shift(1)
through Shift(16), a few masks), the candidate count is ~800k —
tractable in seconds, parallelized via `rayon` when needed.

## The three-gate acceptance

`try_synthesize(weight, max_depth)`:

1. BFS finds a program whose materialization equals the target. (If
   not, return None.)
2. The program's encoded size must be strictly less than the
   ternary matrix's byte size. (Storage MDL gate.)
3. The program's encoded size is the storage cost; the matrix's
   `mdl_ratio` from distillation is preserved. (No separate gate
   — the distill stage already enforced it.)

On success, `CrystalModule.symbolic_hint = Some(Program)`. Today
this is **metadata only**: runtime still uses the ternary matrix.
Phase Q wires the hint into execution; Phase Q's storage
optimization serializes only the program and reconstructs the
matrix on load.

## What's planted, what's found

A synthetic cluster whose correction is a known Shift(3) produces a
ternary matrix that BFS recovers exactly. Tests cover:

- Identity, Shift, Reverse, Negate, and composed Shift+Negate
  round-trip through materialization.
- `find_program` recovers planted patterns at depth ≤ 3.
- Non-reducible dense targets correctly return None at depth 4
  (exhausts the search space).

On real data, the fraction of modules that reduce to programs is
an empirical question for Phase L's diagnostic corpus and the first
real run. Hypothesis: surprisingly high on text with repeating
structural patterns (byte-shift cycles, repeated punctuation),
surprisingly low on unstructured natural language.

## Phase P and Phase V — planned extensions

- **Head S** (`src/nn/symbolic_head.rs`, Phase P) — learns to predict
  from a cluster's hidden-state centroid whether synthesis will
  succeed. Skips BFS on clusters that won't reduce. Cost saver.
- **DSL expansion** (Phase V) — Sign, Where(cond, A, B), Accumulate,
  Delay(k). Breaks strict linearity but preserves ternary-reducibility
  via two-matrix decompositions. Enables discovering programs that
  today's DSL can't see (bit-shift + conditional zeroing, for
  instance).
