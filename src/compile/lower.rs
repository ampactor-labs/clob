//! Lower a CrystalModule's ternary weight matrix to IR.

use crate::compile::ir::{ModuleIR, Op, Trit};
use crate::tensor::ternary::TernaryMatrix;

/// Lower a ternary matrix to IR.
/// Skips zero entries — this is where sparsity becomes speed.
pub fn lower_ternary(mat: &TernaryMatrix) -> ModuleIR {
    let rows = mat.rows();
    let cols = mat.cols();
    let trits = mat.unpack();
    let scales = mat.scales();

    let mut ir = ModuleIR::new(cols, rows);
    let mut total_entries = 0usize;
    let mut nonzero_entries = 0usize;

    for r in 0..rows {
        // Emit accumulate ops for non-zero trits
        for c in 0..cols {
            total_entries += 1;
            let t = trits[r * cols + c];
            match t {
                1 => {
                    ir.ops.push(Op::Accumulate { row: r, col: c, trit: Trit::Plus });
                    nonzero_entries += 1;
                }
                -1 => {
                    ir.ops.push(Op::Accumulate { row: r, col: c, trit: Trit::Minus });
                    nonzero_entries += 1;
                }
                _ => {} // Skip zeros — this is the sparsity win
            }
        }

        // Emit scale
        if scales[r].abs() > 1e-10 {
            ir.ops.push(Op::Scale { row: r, factor: scales[r] });
        }
        ir.ops.push(Op::Store { row: r });
    }

    ir.sparsity = 1.0 - (nonzero_entries as f32 / total_entries.max(1) as f32);
    ir.n_nonzero_ops = nonzero_entries;

    ir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_small() {
        let trits: Vec<i8> = vec![1, 0, -1, 0, 1, -1, 0, 0, 1];
        let scales = vec![0.5, 0.5, 0.5];
        let mat = TernaryMatrix::pack(&trits, &scales, 3, 3);
        let ir = lower_ternary(&mat);

        // 5 non-zero trits out of 9 total
        assert_eq!(ir.n_nonzero_ops, 5);
        assert!((ir.sparsity - (4.0 / 9.0)).abs() < 0.01);
    }
}
