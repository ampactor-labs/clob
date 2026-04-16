//! Bit-exact scalar reference implementations of all kernels.
//!
//! These produce identical results to the AVX2 variants and serve as
//! ground truth for correctness testing.

use crate::simd::TernaryKernel;
use crate::tensor::ternary::{TernaryMatrix, TRIT_DECODE};

/// Scalar (non-SIMD) kernel backend.
pub struct ScalarKernel;

impl TernaryKernel for ScalarKernel {
    fn ternary_accumulate(
        activations: &[f32],
        weights: &TernaryMatrix,
        output: &mut [f32],
    ) {
        let rows = weights.rows();
        let cols = weights.cols();
        assert_eq!(activations.len(), cols);
        assert!(output.len() >= rows);

        // Quantize activations to i8 via absmax
        let abs_max = activations.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let act_scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        let inv_act_scale = if abs_max > 0.0 { 127.0 / abs_max } else { 0.0 };

        let quantized: Vec<i8> = activations.iter().map(|&v| {
            (v * inv_act_scale).round().clamp(-127.0, 127.0) as i8
        }).collect();

        let c_chunks = (cols + 3) / 4;
        let packed = weights.packed_data();
        let scales = weights.scales();

        for r in 0..rows {
            let mut acc: i32 = 0;
            let r_chunk = r / 32;
            let r_rem = r % 32;

            for c in 0..cols {
                let c_chunk = c / 4;
                let c_rem = c % 4;

                let block_idx = r_chunk * c_chunks + c_chunk;
                let byte_idx = block_idx * 32 + r_rem;
                let bit_offset = c_rem * 2;

                let encoded = (packed[byte_idx] >> bit_offset) & 0x03;
                let trit = TRIT_DECODE[encoded as usize];

                acc += trit as i32 * quantized[c] as i32;
            }

            output[r] = acc as f32 * act_scale * scales[r];
        }
    }

    fn sigmoid_inplace(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }

    fn silu_inplace(x: &mut [f32]) {
        for v in x.iter_mut() {
            let sig = 1.0 / (1.0 + (-*v).exp());
            *v *= sig;
        }
    }

    fn hadamard(a: &[f32], b: &[f32], out: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
    }

    fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
        assert_eq!(x.len(), weight.len());
        let n = x.len() as f32;
        let ss: f32 = x.iter().map(|v| v * v).sum();
        let rms = (ss / n + eps).sqrt();
        let inv_rms = 1.0 / rms;
        for (v, w) in x.iter_mut().zip(weight.iter()) {
            *v = *v * inv_rms * w;
        }
    }

    fn softplus_inplace(x: &mut [f32]) {
        for v in x.iter_mut() {
            if *v > 20.0 {
                // softplus(x) ≈ x for large x
            } else if *v < -20.0 {
                *v = 0.0;
            } else {
                *v = (1.0 + v.exp()).ln();
            }
        }
    }
}
