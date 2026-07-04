//! Pattern distillation — extract shared structure from a cluster + MDL check.
//!
//! DISTILL step of the crystallization loop. Two gates decide whether a
//! cluster is worth crystallizing:
//!
//! 1. **Coherence** — episodes in the cluster must agree on what the correct
//!    next token is (or at least concentrate on a small set). A cluster that
//!    mixes unrelated `actual_token`s averages its correction signal to noise.
//!    We reject clusters whose actual-token distribution has entropy above
//!    `coherence_entropy_bits`.
//!
//! 2. **MDL gain** — the cluster must admit a compressed two-part code
//!    (pattern + Gaussian residuals) shorter than the raw baseline.
//!    The savings estimate is the Gaussian log-likelihood ratio of
//!    within-cluster variance against batch variance, which is the standard
//!    bits-saved quantity when you replace "each value is independent" with
//!    "each value is a draw from a tighter distribution around a mean".

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
    /// Number of source episodes after coherence filtering.
    pub n_episodes: usize,
    /// Average prediction error across contributing episodes.
    pub avg_error: f32,
    /// MDL ratio: total_pattern_bits / raw_bits. Must be < `mdl_gate` to be crystallized.
    pub mdl_ratio: f32,
    /// Shannon entropy (bits) of the actual-token distribution in the cluster.
    pub token_entropy_bits: f32,
    /// Timestamps of source episodes (for marking consumed).
    pub source_timestamps: Vec<u64>,
}

/// Distillation thresholds. Defaults are conservative; tune as the loop
/// matures.
#[derive(Debug, Clone, Copy)]
pub struct DistillConfig {
    /// Minimum episodes in a cluster to attempt distillation.
    pub min_episodes: usize,
    /// Reject clusters whose `actual_token` entropy exceeds this (bits).
    /// A cluster dominated by one or two tokens has entropy < 1 bit.
    pub coherence_entropy_bits: f32,
    /// Accept only if MDL ratio is below this.
    pub mdl_gate: f32,
    /// Numerical floor for per-dimension variance (avoids log(0)).
    pub var_floor: f32,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            min_episodes: 3,
            coherence_entropy_bits: 2.0,
            mdl_gate: 0.95,
            var_floor: 1e-6,
        }
    }
}

/// Attempt to distill a cluster into a pattern.
///
/// `embed_table` is the model's tied embed/unembed matrix, row-major
/// `[vocab × d]`. When present, the correction direction is the
/// gradient-aligned `avg(embed[actual] - h)` — pushing the hidden state
/// toward the actual token's embedding row so that unembed(hidden) assigns
/// higher probability to `actual`. When absent, falls back to a hidden-state
/// deviation from centroid (weaker signal; useful for smoke tests only).
///
/// Returns `None` if the cluster is too small, incoherent, or fails the MDL gate.
pub fn distill(
    cluster: &Cluster,
    episodes: &[Episode],
    embed_table: Option<&[f32]>,
    vocab_size: usize,
) -> Option<DistilledPattern> {
    distill_with(cluster, episodes, embed_table, vocab_size, DistillConfig::default())
}

/// Distillation with explicit thresholds.
pub fn distill_with(
    cluster: &Cluster,
    episodes: &[Episode],
    embed_table: Option<&[f32]>,
    vocab_size: usize,
    cfg: DistillConfig,
) -> Option<DistilledPattern> {
    let n = cluster.episode_indices.len();
    if n < cfg.min_episodes {
        return None;
    }

    let d = episodes[0].hidden_state.len();

    // ── Gate 1: coherence of actual_token distribution ──────────────────────
    // A cluster whose desired outputs are spread across many tokens cannot
    // yield a useful correction direction by averaging. Reject early.
    let token_entropy_bits = actual_token_entropy_bits(cluster, episodes);
    if token_entropy_bits > cfg.coherence_entropy_bits {
        return None;
    }

    // ── Average input hidden state ──────────────────────────────────────────
    let mut avg_input = vec![0.0f32; d];
    for &i in &cluster.episode_indices {
        for (a, h) in avg_input.iter_mut().zip(episodes[i].hidden_state.iter()) {
            *a += h;
        }
    }
    let inv_n = 1.0 / n as f32;
    for v in avg_input.iter_mut() {
        *v *= inv_n;
    }

    // ── Average correction delta, weighted by prediction error ──────────────
    // Preferred target: direction that increases log P(actual) under the tied
    // unembedding: correction_i = embed[actual_i] − h_i.
    let mut avg_correction = vec![0.0f32; d];
    let mut total_error = 0.0f32;
    for &i in &cluster.episode_indices {
        let ep = &episodes[i];
        let w = ep.prediction_error;
        if w <= 0.0 {
            continue;
        }
        total_error += w;

        match embed_table {
            Some(table) if (ep.actual_token as usize) < vocab_size => {
                let row_start = ep.actual_token as usize * d;
                let row = &table[row_start..row_start + d];
                for (c, (h, e)) in avg_correction
                    .iter_mut()
                    .zip(ep.hidden_state.iter().zip(row.iter()))
                {
                    *c += w * (e - h);
                }
            }
            _ => {
                // Diagnostic fallback: hidden deviation from centroid.
                for (c, (h, cen)) in avg_correction
                    .iter_mut()
                    .zip(ep.hidden_state.iter().zip(cluster.centroid.iter()))
                {
                    *c += w * (h - cen);
                }
            }
        }
    }
    if total_error > 0.0 {
        let inv = 1.0 / total_error;
        for v in avg_correction.iter_mut() {
            *v *= inv;
        }
    }

    // ── Gate 2: MDL ─────────────────────────────────────────────────────────
    // Compare two codes for the cluster's hidden states:
    //
    //   BASELINE: store each of n episodes' d-dim hidden states as raw f32.
    //     bits_raw = n * d * 32
    //
    //   PATTERN:  store the pattern (avg_input + avg_correction = 2·d f32s),
    //             then encode residuals (h_i − avg_input) with a diagonal
    //             Gaussian code using the per-dim within-cluster variance
    //             against a batch-level reference variance.
    //     bits_pattern = 2·d·32 + n·d·c_per_value
    //
    // Under a Gaussian code at equivalent f32 precision, the per-value code
    // cost shrinks relative to raw by (1/2)·log2(var_batch / var_within) bits
    // per dimension. (Standard likelihood-ratio MDL accounting: raw is the
    // null model "each value comes from the batch distribution"; pattern is
    // the alternative "each value comes from a tighter distribution around
    // the cluster mean".) Bits saved per value ≈ (1/2)·log2(var_batch/var_within)
    // summed over dims and floored at zero (a cluster is never *worse* than
    // the baseline — you can always encode a null pattern).
    let (var_within_sum, var_batch_sum) =
        variance_accumulators(cluster, episodes, &avg_input, cfg.var_floor);

    // Per-dim log ratios summed across d. Floor at zero: a cluster strictly
    // wider than the batch in some dimension contributes nothing.
    let mut log_ratio_sum_bits = 0.0f32;
    for (vw, vb) in var_within_sum.iter().zip(var_batch_sum.iter()) {
        let r = (vb / vw).max(1.0);
        log_ratio_sum_bits += r.log2();
    }
    let bits_saved = 0.5 * n as f32 * log_ratio_sum_bits;

    let bits_raw = n as f32 * d as f32 * 32.0;
    let bits_pattern = 2.0 * d as f32 * 32.0 + (bits_raw - bits_saved);
    let mdl_ratio = bits_pattern / bits_raw;

    if mdl_ratio >= cfg.mdl_gate {
        return None;
    }

    let source_timestamps: Vec<u64> = cluster
        .episode_indices
        .iter()
        .map(|&i| episodes[i].timestamp)
        .collect();

    Some(DistilledPattern {
        avg_input,
        avg_correction,
        domain_signature: cluster.centroid.clone(),
        n_episodes: n,
        avg_error: cluster.avg_error,
        mdl_ratio,
        token_entropy_bits,
        source_timestamps,
    })
}

/// Shannon entropy (bits) of the `actual_token` distribution inside a cluster.
/// Zero if all episodes agree on the next token; log2(k) if k tokens appear
/// with equal frequency.
fn actual_token_entropy_bits(cluster: &Cluster, episodes: &[Episode]) -> f32 {
    use std::collections::HashMap;
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &i in &cluster.episode_indices {
        *counts.entry(episodes[i].actual_token).or_insert(0) += 1;
    }
    let n = cluster.episode_indices.len() as f32;
    let mut h = 0.0f32;
    for &c in counts.values() {
        let p = c as f32 / n;
        h -= p * p.log2();
    }
    h
}

/// Compute per-dimension within-cluster variance (residuals against `avg_input`)
/// and batch variance (residuals against the batch mean across *all* episodes).
/// Both are floored at `floor` to avoid log(0) / div(0).
fn variance_accumulators(
    cluster: &Cluster,
    episodes: &[Episode],
    avg_input: &[f32],
    floor: f32,
) -> (Vec<f32>, Vec<f32>) {
    let d = avg_input.len();
    let n = cluster.episode_indices.len() as f32;
    let n_batch = episodes.len() as f32;

    // Within-cluster: var_j = (1/n) * Σ_i (h_ij − avg_input_j)²
    let mut var_within = vec![0.0f32; d];
    for &i in &cluster.episode_indices {
        for (v, (h, a)) in var_within
            .iter_mut()
            .zip(episodes[i].hidden_state.iter().zip(avg_input.iter()))
        {
            let r = h - a;
            *v += r * r;
        }
    }
    for v in var_within.iter_mut() {
        *v = (*v / n).max(floor);
    }

    // Batch mean (single pass over all episodes).
    let mut batch_mean = vec![0.0f32; d];
    for ep in episodes {
        for (m, h) in batch_mean.iter_mut().zip(ep.hidden_state.iter()) {
            *m += h;
        }
    }
    let inv_nb = 1.0 / n_batch;
    for m in batch_mean.iter_mut() {
        *m *= inv_nb;
    }

    // Batch variance: (1/n_batch) * Σ_i (h_ij − batch_mean_j)²
    let mut var_batch = vec![0.0f32; d];
    for ep in episodes {
        for (v, (h, m)) in var_batch
            .iter_mut()
            .zip(ep.hidden_state.iter().zip(batch_mean.iter()))
        {
            let r = h - m;
            *v += r * r;
        }
    }
    for v in var_batch.iter_mut() {
        *v = (*v * inv_nb).max(floor);
    }

    (var_within, var_batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(ts: u64, h: Vec<f32>, actual: u32) -> Episode {
        Episode {
            timestamp: ts,
            context: vec![],
            hidden_state: h,
            energy: 1.0,
            predictions: vec![(actual + 1, 0.4)],
            actual_token: actual,
            prediction_error: 0.6,
            consumed: false,
            future: vec![actual],
        }
    }

    #[test]
    fn rejects_incoherent_cluster() {
        // 4 episodes, same hidden, but each a different actual token —
        // entropy = 2 bits, above the 2.0 threshold is borderline; use 6 for a clear miss.
        let mut episodes = Vec::new();
        let mut indices = Vec::new();
        for (i, tok) in [10u32, 20, 30, 40, 50, 60].iter().enumerate() {
            episodes.push(ep(i as u64, vec![0.0; 4], *tok));
            indices.push(i);
        }
        let cluster = Cluster {
            episode_indices: indices,
            centroid: vec![0.0; 4],
            avg_error: 0.6,
            avg_energy: 1.0,
        };
        assert!(distill(&cluster, &episodes, None, 1024).is_none());
    }

    #[test]
    fn accepts_coherent_tight_cluster() {
        // 32 episodes, all predict token 42, tight around [1, 0, 0, 0].
        // Decoys widely scattered to inflate batch variance.
        // Pattern overhead is 2·d·32 = 256 bits, so n must be large enough
        // that the overhead amortizes — which is why the engine's
        // min_episodes default is 50 in production.
        let mut episodes = Vec::new();
        for i in 0..32 {
            let jitter = 0.001 * (i as f32);
            episodes.push(ep(i, vec![1.0 + jitter, 0.0, 0.0, 0.0], 42));
        }
        for i in 32..64 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            episodes.push(ep(i as u64, vec![sign * 10.0, sign * 8.0, sign * -7.0, sign * 12.0], 99));
        }
        let cluster = Cluster {
            episode_indices: (0..32).collect(),
            centroid: vec![1.0, 0.0, 0.0, 0.0],
            avg_error: 0.6,
            avg_energy: 1.0,
        };
        let pattern = distill(&cluster, &episodes, None, 1024)
            .expect("tight coherent cluster should distill");
        assert_eq!(pattern.n_episodes, 32);
        assert!(pattern.mdl_ratio < 0.95, "mdl_ratio = {}", pattern.mdl_ratio);
        assert!(pattern.token_entropy_bits < 0.01);
    }

    #[test]
    fn rejects_when_cluster_as_wide_as_batch() {
        // Cluster members span as widely as the full batch — no compression gain.
        let mut episodes = Vec::new();
        for i in 0..6 {
            let v = if i % 2 == 0 { 5.0 } else { -5.0 };
            episodes.push(ep(i as u64, vec![v, v, v, v], 42));
        }
        let cluster = Cluster {
            episode_indices: (0..6).collect(),
            centroid: vec![0.0, 0.0, 0.0, 0.0],
            avg_error: 0.6,
            avg_energy: 1.0,
        };
        // var_within == var_batch ⇒ no bits saved ⇒ mdl_ratio ≥ 1.0 ⇒ reject.
        assert!(distill(&cluster, &episodes, None, 1024).is_none());
    }

    #[test]
    fn too_small_cluster_rejected() {
        let episodes = vec![
            ep(0, vec![0.0; 4], 1),
            ep(1, vec![0.0; 4], 1),
        ];
        let cluster = Cluster {
            episode_indices: vec![0, 1],
            centroid: vec![0.0; 4],
            avg_error: 0.6,
            avg_energy: 1.0,
        };
        assert!(distill(&cluster, &episodes, None, 1024).is_none());
    }
}
