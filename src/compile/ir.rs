//! Intermediate representation for ternary operations.

/// A trit value in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trit {
    Zero,
    Plus,
    Minus,
}

/// A single IR operation.
#[derive(Debug, Clone)]
pub enum Op {
    /// Accumulate: output[row] += trit * input[col]
    Accumulate { row: usize, col: usize, trit: Trit },
    /// Scale: output[row] *= factor
    Scale { row: usize, factor: f32 },
    /// Store result to output buffer.
    Store { row: usize },
}

/// A compiled function IR — sequence of ops for a ternary module forward pass.
#[derive(Debug, Clone)]
pub struct ModuleIR {
    pub ops: Vec<Op>,
    pub n_inputs: usize,
    pub n_outputs: usize,
    /// Sparsity: fraction of zero trits (skipped in codegen).
    pub sparsity: f32,
    /// Non-zero operations count.
    pub n_nonzero_ops: usize,
}

impl ModuleIR {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            ops: Vec::new(),
            n_inputs,
            n_outputs,
            sparsity: 0.0,
            n_nonzero_ops: 0,
        }
    }

    /// Stats.
    pub fn stats(&self) -> String {
        format!(
            "IR: {} ops ({} accumulate, {} scale), {}×{}, sparsity={:.1}%",
            self.ops.len(), self.n_nonzero_ops, self.n_outputs,
            self.n_outputs, self.n_inputs, self.sparsity * 100.0,
        )
    }
}
