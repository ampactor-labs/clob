//! Pattern distillation — extract shared structure from a cluster + MDL check.
//!
//! This is the DISTILL (step 2) of the crystallization loop:
//! Is the compressed pattern shorter than the sum of individual episodes?

use crate::crystal::cluster::Cluster;
use crate::memory::episode::Episode;

/// A distilled pattern ready for crystallization.
#[derive(Debug, Clone)]
pub struct DistilledPattern {
    /// Average input hidden state for this pattern.
    pub avg_input: Vec<f32>,
    /// Average correction delta: what the model should have done differently.
    pub avg_correction: Vec<f32>,
    /// Domain signature: the centroid of the cluster (for routing).
    pub domain_signature: Vec<f32>,
    /// Number of source episodes.
    pub n_episodes: usize,
    /// Average prediction error.
    pub avg_error: f32,
    /// MDL ratio: pattern_mdl / episodes_mdl. Must be < 1.0 to be worth crystallizing.
    pub mdl_ratio: f32,
    /// Timestamps of source episodes (for marking consumed).
    pub source_timestamps: Vec<u64>,
}

/// Attempt to distill a cluster into a pattern.
/// Returns None if the MDL check fails (not worth crystallizing).
pub fn distill(cluster: &Cluster, episodes: &[Episode]) -> Option<DistilledPattern> {
    if cluster.episode_indices.len() < 3 {
        return None; // Need at least 3 episodes to form a meaningful pattern
    }

    let d = episodes[0].hidden_state.len();
    let n = cluster.episode_indices.len();

    // Compute average input (hidden state that led to the error)
    let mut avg_input = vec![0.0f32; d];
    for &i in &cluster.episode_indices {
        for (a, h) in avg_input.iter_mut().zip(episodes[i].hidden_state.iter()) {
            *a += h;
        }
    }
    for v in avg_input.iter_mut() { *v /= n as f32; }

    // Compute average correction delta
    // The "correction" is the difference between what was predicted (centroid)
    // and what should have been predicted. In practice, we use the prediction
    // error pattern: high error episodes in the same region of hidden space
    // indicate a systematic bias.
    let mut avg_correction = vec![0.0f32; d];
    let mut total_error = 0.0f32;
    for &i in &cluster.episode_indices {
        let ep = &episodes[i];
        let error_weight = ep.prediction_error;
        total_error += error_weight;

        // Direction of the correction: difference between episode's hidden state and centroid
        for (c, (h, cen)) in avg_correction.iter_mut()
            .zip(ep.hidden_state.iter().zip(cluster.centroid.iter()))
        {
            *c += error_weight * (h - cen);
        }
    }
    if total_error > 0.0 {
        for v in avg_correction.iter_mut() { *v /= total_error; }
    }

    // MDL check: is the pattern + residuals shorter than storing all episodes?
    //
    // Individual episodes MDL ≈ n * d * 32 bits (each episode stores d f32 values)
    // Pattern MDL ≈ 2 * d * 32 bits (avg_input + avg_correction)
    // Residual MDL ≈ n * variance * d (how much the episodes deviate from the pattern)
    //
    // We approximate: if the within-cluster variance is low enough, the pattern
    // is a good compression.

    let mut variance = 0.0f32;
    for &i in &cluster.episode_indices {
        for (h, c) in episodes[i].hidden_state.iter().zip(cluster.centroid.iter()) {
            variance += (h - c) * (h - c);
        }
    }
    variance /= (n * d) as f32;

    // Pattern entropy ≈ 2 * d (the pattern itself)
    // Residual entropy ≈ n * d * log2(1 + variance)
    let pattern_bits = 2.0 * d as f32; // The distilled pattern
    let residual_bits = n as f32 * d as f32 * (1.0 + variance).log2();
    let total_pattern_mdl = pattern_bits + residual_bits;
    let episodes_mdl = n as f32 * d as f32; // Raw episodes

    let mdl_ratio = total_pattern_mdl / episodes_mdl;

    // Only crystallize if the pattern achieves genuine compression
    if mdl_ratio >= 0.95 {
        return None; // Less than 5% compression — not worth it
    }

    let source_timestamps: Vec<u64> = cluster.episode_indices.iter()
        .map(|&i| episodes[i].timestamp)
        .collect();

    Some(DistilledPattern {
        avg_input,
        avg_correction,
        domain_signature: cluster.centroid.clone(),
        n_episodes: n,
        avg_error: cluster.avg_error,
        mdl_ratio,
        source_timestamps,
    })
}
