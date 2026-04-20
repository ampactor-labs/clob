# The compile pipeline

Crystallized ternary matrices → native x86-64 `.so` → `dlopen` →
hot-swapped into the running inference path.

## Current state: scalar

`src/compile/x86.rs:emit_x86` emits one `movss` + `addss`/`subss`
per nonzero trit. The dimensions are baked in as immediates; zero
trits are elided (the lower pass at `src/compile/lower.rs:30` skips
them — this is where sparsity becomes speed).

Roughly:

```
push rbp; mov rbp, rsp
memset output[] = 0  ; rep stosb
for each nonzero trit (row, col, ±1):
    movss xmm1, [rdi + col*4]
    addss xmm0, xmm1   ; or subss
movss [rsi + row*4], xmm0
xorps xmm0, xmm0
pop rbp; ret
```

This runs correctly but at a fraction of the silicon's capability.
The "32 ops/clock" target in ARCHITECTURE.md is aspirational; the
scalar path delivers closer to 2-3.

## The IR (`src/compile/ir.rs`)

Three ops, row-major sequenced:

- `Accumulate { row, col, trit: Plus | Minus }` — one `addss`/`subss`.
- `Scale { row, factor: f32 }` — one `mulss` with the scale factor.
- `Store { row }` — `movss` to the output slot.

Zero trits never reach the emitter; the lower pass filters them.

## The dispatcher (planned, Phase O)

Phase O will add `src/compile/x86_avx2.rs` and `x86_avx512.rs`
emitters that group trits into 8- (or 16-) lane batches and use
packed sign ops (`vpsignd`-equivalent or branchless masks) for the
accumulate phase. `src/compile/dispatch.rs` selects the emitter
based on CPUID — already detected in `src/simd/mod.rs` for the
runtime ternary matmul.

Bit-exact cross-check between scalar and AVX2 outputs is the
correctness gate; the same ternary arithmetic reordered into
wider lanes must produce identical results, not just
approximately-equal.

## ELF emission (`src/compile/elf.rs`)

Minimal ELF64 shared-object: text section with the emitted code,
symbol table with `module_forward` exported, dynamic section with
the init/fini structure `dlopen` requires. No relocations (the
emitter uses absolute offsets in `rdi`/`rsi`), no debug info, no
unwind tables — just enough ELF to survive the loader.

## JIT loader (`src/compile/jit.rs`)

`CompiledModule::load(path, module_id)` — `dlopen` + `dlsym
"module_forward"` + `transmute` to
`extern "C" fn(*const f32, *mut f32, usize)`. The raw function
pointer lives for the module's lifetime. `forward(input, output)`
is `unsafe` and assumes caller-checked lengths.

`Drop` calls `dlclose`. No attempt to unload mid-execution —
hot-swapping would need a generation counter and a grace period;
today modules live until process exit or explicit removal.

## Ternary runtime vs compile

There are two paths to executing a ternary module:

1. **Runtime path** — `CrystalModule::apply` through the
   `KernelDispatch::ternary_accumulate` SIMD kernels. AVX2-enabled.
   This is the default today; the emitter is not yet used in the
   hot inference path.
2. **Compile path** — `dlopen`'d `.so` from the emitter. Currently
   demoed by `clob compile` but not wired into inference.

Phase Q wires the compile path for modules with a `symbolic_hint`
and tests bit-exact equivalence with the runtime path. The
short-term win is faster specialized modules; the long-term win is
symbolic interpretation at the IR level (no native code needed for
the simplest DSL programs).
