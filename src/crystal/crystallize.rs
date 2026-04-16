//! Crystallization — generate ternary weight modules from distilled patterns.
//!
//! Takes a distilled pattern (avg_input → avg_correction mapping) and
//! produces a small ternary weight matrix that encodes the correction.
//! This is not gradient descent — it's a direct closed-form solution.

use crate::crystal::distill::DistilledPattern;
use crate::crystal::module::CrystalModule;
use crate::tensor::ternary::TernaryMatrix;

/// Crystallize a distilled pattern into a ternary module.
///
/// The module learns the mapping: input_pattern → correction_delta.
/// Uses a rank-1 outer product approximation ternarized via round-clamp.
pub fn crystallize(pattern: &DistilledPattern, module_id: u64) -> CrystalModule {
    let d = pattern.avg_input.len();

    // Compute the rank-1 approximation: W ≈ correction ⊗ input^T / ||input||²
    // This is the simplest closed-form "training": find the matrix that maps
    // the average input to the average correction.
    let input_norm_sq: f32 = pattern.avg_input.iter().map(|v| v * v).sum();

    let mut weights_f32 = vec![0.0f32; d * d];
    if input_norm_sq > 1e-8 {
        let scale = 1.0 / input_norm_sq;
        for r in 0..d {
            for c in 0..d {
                weights_f32[r * d + c] = pattern.avg_correction[r] * pattern.avg_input[c] * scale;
            }
        }
    }

    // Ternarize: round-clamp to {-1, 0, 1}
    // Use per-row absmean as the scale factor (BitNet b1.58 approach)
    let mut trits = vec![0i8; d * d];
    let mut scales = vec![0.0f32; d];

    for r in 0..d {
        let row = &weights_f32[r * d..(r + 1) * d];
        let abs_mean = row.iter().map(|v| v.abs()).sum::<f32>() / d as f32;

        if abs_mean > 1e-10 {
            let inv_scale = 1.0 / abs_mean;
            for c in 0..d {
                let normalized = row[c] * inv_scale;
                trits[r * d + c] = if normalized > 0.5 {
                    1
                } else if normalized < -0.5 {
                    -1
                } else {
                    0
                };
            }
            scales[r] = abs_mean;
        }
    }

    let weight = TernaryMatrix::pack(&trits, &scales, d, d);

    // Compute sparsity (fraction of zero weights)
    let n_nonzero = trits.iter().filter(|&&t| t != 0).count();
    let sparsity = 1.0 - (n_nonzero as f32 / (d * d) as f32);

    CrystalModule {
        id: module_id,
        weight,
        domain_signature: pattern.domain_signature.clone(),
        d_model: d,
        n_source_episodes: pattern.n_episodes,
        avg_error_before: pattern.avg_error,
        mdl_ratio: pattern.mdl_ratio,
        sparsity,
        activation_count: 0,
    }
}
