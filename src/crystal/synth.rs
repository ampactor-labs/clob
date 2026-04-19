//! Symbolic crystallization — discover a short DSL program that realizes a
//! crystallized ternary matrix.
//!
//! The key insight (from the Phase E plan): **the DSL is ternary-reducible**.
//! Every primitive produces a matrix whose entries are in {-1, 0, +1}, and
//! composition of primitives produces another ternary matrix. This collapses
//! the program-equivalence problem to matrix equality and avoids any PhD-
//! level synthesis machinery.
//!
//! The storage win is the point: a 6-op program serializes in ~24 bytes;
//! the equivalent d×d ternary matrix serializes in ~d²/4 bytes (4 KB for
//! d=128). Attaching a `symbolic_hint` to a `CrystalModule` does not change
//! runtime behavior — the crystallized ternary matrix is still executed —
//! but marks the module as compressible for future storage optimization.
//!
//! ```text
//! Primitives (all produce d×d matrices with entries in {-1, 0, +1}):
//!   Identity       : I
//!   Shift(k)       : cyclic permutation by k positions
//!   Negate         : -I
//!   Reverse        : anti-diagonal I
//!   Mask(bits)     : diagonal with bit i set to 1, else 0
//!
//! A Program is a sequence of primitives composed as matrix products.
//! ```

use crate::tensor::ternary::TernaryMatrix;
use serde::{Deserialize, Serialize};

/// Maximum depth of enumerative search. At depth d, we try up to
/// |primitives|^d = 5^d programs. d=6 gives ≈15k candidates per target.
pub const DEFAULT_MAX_DEPTH: usize = 4;

/// One primitive operation in the DSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Identity,
    /// Cyclic shift by k positions (output[i] = input[(i+k) mod d]).
    Shift(u16),
    /// Flip all signs.
    Negate,
    /// Reverse the vector.
    Reverse,
    /// Zero-mask: keep dimension i if bit i is set. `bits` is a 64-dim
    /// approximation — only the low `min(d, 64)` bits are meaningful.
    Mask(u64),
}

/// A sequence of ops composed as matrix products (leftmost applied first).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub ops: Vec<Op>,
}

impl Program {
    pub fn identity() -> Self {
        Self { ops: vec![Op::Identity] }
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty() || self.ops.iter().all(|o| matches!(o, Op::Identity))
    }

    pub fn encoded_size(&self) -> usize {
        // Op enum serializes to roughly 9 bytes each under bincode (1-byte
        // discriminant + up to 8-byte payload). Caller comparing against
        // weight_bytes() of the ternary matrix wants a rough but
        // comparable figure.
        bincode::serialized_size(self).map(|n| n as usize).unwrap_or(self.ops.len() * 9)
    }

    /// Materialize the program into a d×d matrix of trits. Row-major.
    /// The matrix has entries in {-1, 0, +1}.
    pub fn materialize_trits(&self, d: usize) -> Vec<i8> {
        // Start with identity.
        let mut current = identity_matrix(d);
        for op in &self.ops {
            let prim = primitive_matrix(op, d);
            current = matmul_ternary(&current, &prim, d);
        }
        current
    }

    /// Apply a single ternary primitive to an input vector directly, useful
    /// for runtime interpretation later. Unused at the moment but keeps the
    /// runtime-execution path open.
    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        let d = input.len();
        let mut current = input.to_vec();
        let mut next = vec![0.0f32; d];
        for op in &self.ops {
            apply_op(op, &current, &mut next, d);
            std::mem::swap(&mut current, &mut next);
        }
        current
    }
}

fn identity_matrix(d: usize) -> Vec<i8> {
    let mut m = vec![0i8; d * d];
    for i in 0..d {
        m[i * d + i] = 1;
    }
    m
}

fn primitive_matrix(op: &Op, d: usize) -> Vec<i8> {
    let mut m = vec![0i8; d * d];
    match op {
        Op::Identity => {
            for i in 0..d { m[i * d + i] = 1; }
        }
        Op::Shift(k) => {
            let k = (*k as usize) % d;
            for i in 0..d {
                let j = (i + k) % d;
                m[i * d + j] = 1;
            }
        }
        Op::Negate => {
            for i in 0..d { m[i * d + i] = -1; }
        }
        Op::Reverse => {
            for i in 0..d { m[i * d + (d - 1 - i)] = 1; }
        }
        Op::Mask(bits) => {
            let w = d.min(64);
            for i in 0..d {
                if i < w && (bits >> i) & 1 == 1 {
                    m[i * d + i] = 1;
                }
            }
        }
    }
    m
}

/// Ternary matrix multiplication (values in {-1, 0, 1}; product may saturate
/// outside that range). Since we compose only permutation-like primitives
/// plus diagonal masks, the result is guaranteed ternary in our DSL —
/// we clamp defensively so a broader future DSL doesn't silently overflow.
fn matmul_ternary(a: &[i8], b: &[i8], d: usize) -> Vec<i8> {
    let mut out = vec![0i8; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut acc: i32 = 0;
            for k in 0..d {
                acc += (a[i * d + k] as i32) * (b[k * d + j] as i32);
            }
            out[i * d + j] = acc.clamp(-1, 1) as i8;
        }
    }
    out
}

fn apply_op(op: &Op, input: &[f32], output: &mut [f32], d: usize) {
    for v in output.iter_mut() { *v = 0.0; }
    match op {
        Op::Identity => output.copy_from_slice(input),
        Op::Shift(k) => {
            let k = (*k as usize) % d;
            for i in 0..d {
                output[i] = input[(i + k) % d];
            }
        }
        Op::Negate => {
            for i in 0..d {
                output[i] = -input[i];
            }
        }
        Op::Reverse => {
            for i in 0..d {
                output[i] = input[d - 1 - i];
            }
        }
        Op::Mask(bits) => {
            let w = d.min(64);
            for i in 0..d {
                if i < w && (bits >> i) & 1 == 1 {
                    output[i] = input[i];
                }
            }
        }
    }
}

/// Attempt to find a short program whose materialization exactly matches
/// the given target trit pattern (row-major d×d). Searches depth 1 through
/// `max_depth` in BFS order; returns the shortest match.
///
/// "Exact match" is reasonable because our DSL is closed under composition
/// and the target comes from ternarization — any "close" match is already
/// a perfect match up to the quantization granularity.
pub fn find_program(target: &[i8], d: usize, max_depth: usize) -> Option<Program> {
    assert_eq!(target.len(), d * d);

    // Primitives to try. Mask is parameterized by up to 2^d patterns, which
    // explodes quickly — we enumerate only a small set of "useful" masks
    // (full, alternating, half-masks). A richer mask search is a follow-up.
    let mut primitives = vec![
        Op::Identity,
        Op::Negate,
        Op::Reverse,
    ];
    for k in 1..d.min(16) {
        primitives.push(Op::Shift(k as u16));
    }
    // A few canonical masks.
    let mask_all: u64 = if d >= 64 { !0u64 } else { (1u64 << d) - 1 };
    primitives.push(Op::Mask(mask_all));
    let alt_even: u64 = 0x5555_5555_5555_5555;
    primitives.push(Op::Mask(alt_even));
    let alt_odd: u64 = 0xAAAA_AAAA_AAAA_AAAA;
    primitives.push(Op::Mask(alt_odd));

    // BFS by program length.
    for depth in 1..=max_depth {
        if let Some(prog) = search_depth(target, d, &primitives, depth) {
            return Some(prog);
        }
    }
    None
}

fn search_depth(target: &[i8], d: usize, primitives: &[Op], depth: usize) -> Option<Program> {
    // Iterate over all index tuples of length `depth` over the primitive set.
    let n = primitives.len();
    let mut idx = vec![0usize; depth];
    loop {
        let ops: Vec<Op> = idx.iter().map(|&i| primitives[i]).collect();
        let prog = Program { ops };
        let candidate = prog.materialize_trits(d);
        if candidate == target {
            return Some(prog);
        }
        // Increment idx in base n (least-significant position moves fastest).
        let mut pos = 0;
        while pos < depth {
            idx[pos] += 1;
            if idx[pos] < n {
                break;
            }
            idx[pos] = 0;
            pos += 1;
        }
        if pos == depth {
            return None;
        }
    }
}

/// Try to synthesize a program from an existing CrystalModule's ternary
/// matrix. If a program is found that produces an equivalent trit pattern
/// AND it encodes to fewer bytes than the ternary matrix, returns it.
/// This is the three-gate acceptance check from the plan, simplified to
/// the practical two gates (ternary equivalence + storage improvement).
pub fn try_synthesize(weight: &TernaryMatrix, max_depth: usize) -> Option<Program> {
    let d = weight.rows();
    if weight.cols() != d {
        return None; // Only square weight matrices supported.
    }
    let target = weight.unpack();
    let prog = find_program(&target, d, max_depth)?;
    if prog.encoded_size() < weight.byte_size() {
        Some(prog)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_materializes_correctly() {
        let p = Program::identity();
        let m = p.materialize_trits(8);
        for i in 0..8 {
            for j in 0..8 {
                let expect = if i == j { 1 } else { 0 };
                assert_eq!(m[i * 8 + j], expect);
            }
        }
    }

    #[test]
    fn shift_materializes_correctly() {
        let p = Program { ops: vec![Op::Shift(2)] };
        let m = p.materialize_trits(6);
        // Expect a permutation: m[i][(i+2)%6] = 1.
        for i in 0..6 {
            for j in 0..6 {
                let expect = if j == (i + 2) % 6 { 1 } else { 0 };
                assert_eq!(m[i * 6 + j], expect,
                    "mismatch at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn reverse_materializes_correctly() {
        let p = Program { ops: vec![Op::Reverse] };
        let m = p.materialize_trits(5);
        // m[i][4-i] = 1, else 0.
        for i in 0..5 {
            for j in 0..5 {
                let expect = if j == 4 - i { 1 } else { 0 };
                assert_eq!(m[i * 5 + j], expect);
            }
        }
    }

    #[test]
    fn find_program_recovers_planted_shift() {
        let planted = Program { ops: vec![Op::Shift(3)] };
        let target = planted.materialize_trits(8);
        let found = find_program(&target, 8, 3).expect("should find shift");
        // The recovered program should materialize to the same pattern.
        assert_eq!(found.materialize_trits(8), target);
    }

    #[test]
    fn find_program_recovers_planted_negate() {
        let planted = Program { ops: vec![Op::Negate] };
        let target = planted.materialize_trits(6);
        let found = find_program(&target, 6, 2).expect("should find negate");
        assert_eq!(found.materialize_trits(6), target);
    }

    #[test]
    fn find_program_recovers_planted_shift_then_negate() {
        let planted = Program { ops: vec![Op::Shift(1), Op::Negate] };
        let target = planted.materialize_trits(5);
        let found = find_program(&target, 5, 3).expect("should find composed program");
        assert_eq!(found.materialize_trits(5), target);
    }

    #[test]
    fn find_program_returns_none_for_non_reducible() {
        // A random-looking but valid ternary matrix that isn't a composition
        // of our primitives. (Row 0: [1, -1, 0, 1]; other rows zeroed.) Under
        // depth-4 search, no composition of the listed primitives should
        // produce exactly this matrix.
        let target: Vec<i8> = {
            let mut t = vec![0i8; 16];
            t[0] = 1; t[1] = -1; t[2] = 0; t[3] = 1;
            t
        };
        let found = find_program(&target, 4, 3);
        assert!(found.is_none(),
            "unexpectedly found program for non-reducible target: {:?}", found);
    }

    #[test]
    fn try_synthesize_returns_program_when_storage_improves() {
        // A small permutation ternary matrix ought to compress.
        let planted = Program { ops: vec![Op::Shift(1)] };
        let trits = planted.materialize_trits(64);
        let scales = vec![1.0f32; 64];
        let tmat = TernaryMatrix::pack(&trits, &scales, 64, 64);
        let got = try_synthesize(&tmat, 3).expect("should synthesize");
        assert_eq!(got.materialize_trits(64), trits);
        // Check the storage win: program size << matrix size.
        assert!(got.encoded_size() < tmat.byte_size() / 4,
            "program size {} vs matrix size {} — not a clear win",
            got.encoded_size(), tmat.byte_size());
    }
}
