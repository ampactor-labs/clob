//! Crystallization — generate ternary weight modules from distilled patterns.
//!
//! Takes a distilled pattern (avg_input → avg_correction mapping) and
//! produces a small ternary weight matrix that encodes the correction.
//! This is not gradient descent — it's a direct closed-form solution.

use crate::crystal::distill::DistilledPattern;
use crate::crystal::module::CrystalModule;
use crate::crystal::synth;
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

    // Phase E: try to synthesize an equivalent DSL program. Cheap BFS up
    // to depth 4; a hit replaces the storage payload (~d²/4 bytes) with a
    // few ops (~36 bytes). Module runtime path is unchanged either way.
    let symbolic_hint = synth::try_synthesize(&weight, synth::DEFAULT_MAX_DEPTH);

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
        symbolic_hint,
    }
}

/// Construct a CrystalModule directly from a given set of trits plus a
/// distilled pattern. Used by tests and by future code paths that want to
/// bypass the ternarization step entirely (e.g., programs that emit trits
/// by construction).
pub fn crystallize_from_trits(
    pattern: &DistilledPattern,
    trits: &[i8],
    scales: &[f32],
    module_id: u64,
) -> CrystalModule {
    let d = pattern.avg_input.len();
    assert_eq!(trits.len(), d * d);
    assert_eq!(scales.len(), d);
    let weight = TernaryMatrix::pack(trits, scales, d, d);
    let n_nonzero = trits.iter().filter(|&&t| t != 0).count();
    let sparsity = 1.0 - (n_nonzero as f32 / (d * d) as f32);
    let symbolic_hint = synth::try_synthesize(&weight, synth::DEFAULT_MAX_DEPTH);

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
        symbolic_hint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crystal::synth::{Op, Program};

    fn fake_pattern(d: usize) -> DistilledPattern {
        DistilledPattern {
            avg_input: vec![0.0; d],
            avg_correction: vec![0.0; d],
            domain_signature: vec![0.0; d],
            n_episodes: 10,
            avg_error: 0.5,
            mdl_ratio: 0.5,
            token_entropy_bits: 0.0,
            source_timestamps: vec![0; 10],
        }
    }

    #[test]
    fn planted_shift_pattern_yields_symbolic_hint() {
        let d = 16;
        let planted = Program { ops: vec![Op::Shift(3)] };
        let trits = planted.materialize_trits(d);
        let scales = vec![1.0f32; d];

        let module = crystallize_from_trits(&fake_pattern(d), &trits, &scales, 0);
        let hint = module.symbolic_hint.as_ref()
            .expect("planted shift should yield a symbolic hint");
        assert_eq!(hint.materialize_trits(d), trits,
            "recovered program should materialize to the planted trits");
        assert!(module.symbolic_compression_ratio() < 1.0,
            "symbolic form should compress; ratio = {}",
            module.symbolic_compression_ratio());
    }

    #[test]
    fn non_reducible_pattern_has_no_hint() {
        // Construct trits that our DSL cannot realize in ≤ DEFAULT_MAX_DEPTH
        // compositions: a dense block of −1 and +1 in the top-left corner,
        // zero elsewhere. This is neither a permutation, mask, nor their
        // product, so the BFS search will exhaust depth 4 without a match.
        let d = 16;
        let mut trits = vec![0i8; d * d];
        for i in 0..3 {
            for j in 0..3 {
                trits[i * d + j] = if (i + j) % 2 == 0 { 1 } else { -1 };
            }
        }
        let scales = vec![1.0f32; d];
        let module = crystallize_from_trits(&fake_pattern(d), &trits, &scales, 0);
        assert!(module.symbolic_hint.is_none(),
            "arbitrary dense block should not match any DSL program under depth {}",
            synth::DEFAULT_MAX_DEPTH);
    }
}
