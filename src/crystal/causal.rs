//! Causal-state refinement — cluster episodes by their futures, not by
//! where their hidden states happen to sit.
//!
//! Computational mechanics defines the *causal states* of a process as
//! equivalence classes of histories with the same conditional future
//! distribution: two pasts belong together exactly when they predict the
//! same future. The minimal maximally-predictive model of a process (the
//! ε-machine) is built from those classes, so causal states are what a
//! crystallization loop should be converging toward.
//!
//! Hidden-state k-means approximates this only when the core's geometry
//! already separates predictive circumstances. A frozen or badly trained
//! core mixes them, and the distill coherence gate then rejects the
//! blended cluster wholesale — episodes with perfectly learnable futures
//! are discarded because they were binned by geometry rather than by
//! consequence.
//!
//! This module refines a k-means partition in two passes:
//!
//! 1. **Split** — inside each cluster, group episodes by their observed
//!    future prefix. If two or more groups can stand alone, the cluster
//!    splits along its predictive fault lines.
//! 2. **Merge** — across clusters, compare the distribution over future
//!    *prefixes* (the same horizon the split keys on) with Jensen–Shannon
//!    divergence and merge clusters that are predictively indistinguishable.
//!    Keying merge on the same horizon as split is what makes them inverse
//!    operations: a multi-token split can never be silently undone by a
//!    one-step merge. This is an h-truncated form of the causal-state merge;
//!    the MDL gate downstream still vetoes any merge whose hidden states are
//!    too scattered to compress.
//!
//! The refinement is deterministic: same episodes in, same partition out.

use crate::crystal::cluster::Cluster;
use crate::memory::episode::Episode;
use std::collections::HashMap;

/// Refinement thresholds.
#[derive(Debug, Clone, Copy)]
pub struct CausalConfig {
    /// Future-prefix length used as the split key.
    pub horizon: usize,
    /// A split subgroup must have at least this many episodes to stand
    /// alone. Mirror of `DistillConfig::min_episodes` — smaller groups
    /// couldn't distill anyway.
    pub min_split: usize,
    /// Merge two clusters when the Jensen–Shannon divergence between
    /// their next-token distributions is at most this many bits.
    /// 0 merges only exact matches; 1 would merge anything.
    pub merge_jsd_bits: f32,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            horizon: 4,
            min_split: 3,
            merge_jsd_bits: 0.05,
        }
    }
}

/// Refine a partition by predictive equivalence: split clusters along
/// future fault lines, then merge clusters with indistinguishable
/// next-token behavior. Returns the refined partition, sorted by average
/// error descending like `cluster_episodes` output.
pub fn refine(clusters: Vec<Cluster>, episodes: &[Episode], cfg: &CausalConfig) -> Vec<Cluster> {
    let mut refined: Vec<Cluster> = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        refined.extend(split_by_future(cluster, episodes, cfg));
    }
    let merged = merge_by_future(refined, episodes, cfg);
    let mut out = merged;
    out.sort_by(|a, b| {
        b.avg_error
            .partial_cmp(&a.avg_error)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Split one cluster along its future fault lines. Subgroups of at least
/// `min_split` episodes with identical future prefixes become their own
/// clusters; everything else pools into a remainder cluster (which will
/// face the coherence gate as before). A cluster with fewer than two
/// viable subgroups passes through untouched.
fn split_by_future(cluster: Cluster, episodes: &[Episode], cfg: &CausalConfig) -> Vec<Cluster> {
    let mut groups: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
    for &i in &cluster.episode_indices {
        groups
            .entry(episodes[i].future_prefix(cfg.horizon))
            .or_default()
            .push(i);
    }

    // Deterministic order: by size descending, then key ascending.
    let mut ordered: Vec<(Vec<u32>, Vec<usize>)> = groups.into_iter().collect();
    ordered.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(&b.0)));

    // Split whenever at least one subgroup can stand alone and the
    // partition is non-trivial. Even a single viable group is worth
    // separating from its stragglers: the remainder would otherwise
    // contaminate the group's correction signal.
    let viable = ordered.iter().filter(|(_, g)| g.len() >= cfg.min_split).count();
    if ordered.len() < 2 || viable == 0 {
        return vec![cluster];
    }

    let mut out = Vec::with_capacity(viable + 1);
    let mut remainder: Vec<usize> = Vec::new();
    for (_, group) in ordered {
        if group.len() >= cfg.min_split {
            out.push(build_cluster(group, episodes));
        } else {
            remainder.extend(group);
        }
    }
    if !remainder.is_empty() {
        remainder.sort_unstable();
        out.push(build_cluster(remainder, episodes));
    }
    out
}

/// Greedily merge clusters whose next-token distributions are within
/// `merge_jsd_bits` of each other. Each round merges the closest pair
/// below threshold and recomputes; terminates when no pair qualifies.
fn merge_by_future(mut clusters: Vec<Cluster>, episodes: &[Episode], cfg: &CausalConfig) -> Vec<Cluster> {
    loop {
        // Distribution over full future prefixes at the SAME horizon the
        // split used — so merge is the exact inverse of split and cannot
        // recombine sub-clusters whose futures diverge past token 0.
        let dists: Vec<HashMap<Vec<u32>, f32>> = clusters
            .iter()
            .map(|c| future_distribution(c, episodes, cfg.horizon))
            .collect();

        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let d = jensen_shannon_bits(&dists[i], &dists[j]);
                if d <= cfg.merge_jsd_bits && best.map_or(true, |(_, _, bd)| d < bd) {
                    best = Some((i, j, d));
                }
            }
        }

        match best {
            Some((i, j, _)) => {
                let absorbed = clusters.remove(j);
                let mut indices = clusters[i].episode_indices.clone();
                indices.extend(absorbed.episode_indices);
                indices.sort_unstable();
                clusters[i] = build_cluster(indices, episodes);
            }
            None => return clusters,
        }
    }
}

/// Recompute a cluster's centroid and averages from scratch for the given
/// member indices.
fn build_cluster(episode_indices: Vec<usize>, episodes: &[Episode]) -> Cluster {
    let n = episode_indices.len().max(1) as f32;
    let d = episodes[episode_indices[0]].hidden_state.len();

    let mut centroid = vec![0.0f32; d];
    let mut avg_error = 0.0f32;
    let mut avg_energy = 0.0f32;
    for &i in &episode_indices {
        for (c, h) in centroid.iter_mut().zip(episodes[i].hidden_state.iter()) {
            *c += h;
        }
        avg_error += episodes[i].prediction_error;
        avg_energy += episodes[i].energy;
    }
    for c in centroid.iter_mut() {
        *c /= n;
    }

    Cluster {
        episode_indices,
        centroid,
        avg_error: avg_error / n,
        avg_energy: avg_energy / n,
    }
}

/// Empirical distribution over future prefixes of a cluster, truncated to
/// `horizon`. This is the merge key; matching the split's horizon is what
/// keeps merge from undoing a multi-token split.
fn future_distribution(
    cluster: &Cluster,
    episodes: &[Episode],
    horizon: usize,
) -> HashMap<Vec<u32>, f32> {
    let mut counts: HashMap<Vec<u32>, f32> = HashMap::new();
    for &i in &cluster.episode_indices {
        *counts.entry(episodes[i].future_prefix(horizon)).or_insert(0.0) += 1.0;
    }
    let n = cluster.episode_indices.len().max(1) as f32;
    for v in counts.values_mut() {
        *v /= n;
    }
    counts
}

/// Jensen–Shannon divergence between two discrete distributions, in bits.
/// Symmetric, bounded in [0, 1]: 0 for identical distributions, 1 for
/// distributions with disjoint support. Generic over the symbol type so it
/// serves both single-token and future-prefix distributions.
pub fn jensen_shannon_bits<K>(p: &HashMap<K, f32>, q: &HashMap<K, f32>) -> f32
where
    K: std::hash::Hash + Eq + Ord + Clone,
{
    let mut keys: Vec<K> = p.keys().chain(q.keys()).cloned().collect();
    keys.sort_unstable();
    keys.dedup();

    let mut jsd = 0.0f32;
    for k in keys {
        let pk = p.get(&k).copied().unwrap_or(0.0);
        let qk = q.get(&k).copied().unwrap_or(0.0);
        let mk = 0.5 * (pk + qk);
        if pk > 0.0 {
            jsd += 0.5 * pk * (pk / mk).log2();
        }
        if qk > 0.0 {
            jsd += 0.5 * qk * (qk / mk).log2();
        }
    }
    jsd.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crystal::distill;

    fn ep(ts: u64, h: Vec<f32>, future: Vec<u32>) -> Episode {
        let actual = future[0];
        Episode {
            timestamp: ts,
            context: vec![],
            hidden_state: h,
            energy: 1.0,
            predictions: vec![(actual + 1, 0.4)],
            actual_token: actual,
            prediction_error: 0.6,
            consumed: false,
            future,
        }
    }

    fn whole_batch_cluster(episodes: &[Episode]) -> Cluster {
        build_cluster((0..episodes.len()).collect(), episodes)
    }

    /// The headline case. Five deterministic continuations share one
    /// region of state space — the geometry cannot tell them apart, and
    /// the blended cluster's next-token entropy (log2 5 ≈ 2.32 bits)
    /// fails the coherence gate, so state-only distillation throws all
    /// fifty episodes away. Causal refinement splits along the future
    /// fault lines and every fragment distills.
    #[test]
    fn recovers_patterns_the_coherence_gate_rejects() {
        let mut episodes = Vec::new();
        for i in 0..50u64 {
            let tok = 10 + (i % 5) as u32 * 10; // futures 10,20,30,40,50
            let jitter = 0.001 * (i as f32);
            episodes.push(ep(i, vec![1.0 + jitter, 0.0, 0.0, 0.0], vec![tok, tok + 1]));
        }
        // Scattered decoys so the batch variance is wide (MDL baseline).
        for i in 50..80u64 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            episodes.push(ep(
                i,
                vec![sign * 9.0, sign * -11.0, sign * 8.0, sign * 13.0],
                vec![99],
            ));
        }

        let blended = build_cluster((0..50).collect(), &episodes);

        // State-only: the blended cluster is incoherent and distills to nothing.
        assert!(
            distill::distill(&blended, &episodes, None, 1024).is_none(),
            "blended cluster should fail the coherence gate",
        );

        // Causal: five clean clusters, every one of them distills.
        let refined = refine(vec![blended], &episodes, &CausalConfig::default());
        assert_eq!(refined.len(), 5, "five future groups should split apart");
        for cl in &refined {
            assert_eq!(cl.episode_indices.len(), 10);
            let pattern = distill::distill(cl, &episodes, None, 1024)
                .expect("split cluster should pass both gates");
            assert!(pattern.token_entropy_bits < 0.01);
        }
    }

    #[test]
    fn merges_predictively_identical_clusters() {
        // Two clusters far apart in state space, identical futures.
        // Causal-state definition: same future distribution = same state.
        let mut episodes = Vec::new();
        for i in 0..10u64 {
            episodes.push(ep(i, vec![1.0, 1.0], vec![7, 8]));
        }
        for i in 10..20u64 {
            episodes.push(ep(i, vec![-1.0, -1.0], vec![7, 8]));
        }
        let a = build_cluster((0..10).collect(), &episodes);
        let b = build_cluster((10..20).collect(), &episodes);

        let refined = refine(vec![a, b], &episodes, &CausalConfig::default());
        assert_eq!(refined.len(), 1, "identical futures should merge");
        assert_eq!(refined[0].episode_indices.len(), 20);
        // Merged centroid sits between the two groups.
        assert!(refined[0].centroid[0].abs() < 1e-6);
    }

    #[test]
    fn merge_respects_split_horizon() {
        // The regression guard for the split/merge inverse property. Two
        // groups share their first token (10) but diverge at token 2
        // ([10,11] vs [10,22]). A one-step merge key would see identical
        // next-token distributions {10:1.0} and collapse them — undoing a
        // split that the horizon-4 key made for exactly the right reason.
        // With a horizon-aware merge they stay apart.
        let mut episodes = Vec::new();
        for i in 0..10u64 {
            episodes.push(ep(i, vec![1.0, 0.0], vec![10, 11]));
        }
        for i in 10..20u64 {
            episodes.push(ep(i, vec![1.0, 0.0], vec![10, 22]));
        }
        let a = build_cluster((0..10).collect(), &episodes);
        let b = build_cluster((10..20).collect(), &episodes);

        // Sanity: the one-step distributions ARE identical, so the old
        // next-token merge key would have collapsed these.
        let one_step_a: HashMap<Vec<u32>, f32> = future_distribution(&a, &episodes, 1);
        let one_step_b: HashMap<Vec<u32>, f32> = future_distribution(&b, &episodes, 1);
        assert!(jensen_shannon_bits(&one_step_a, &one_step_b) < 1e-6);

        // With the default horizon (4) they must not merge.
        let refined = refine(vec![a, b], &episodes, &CausalConfig::default());
        assert_eq!(refined.len(), 2, "divergent tails must survive merge");
    }

    #[test]
    fn keeps_distinct_futures_apart() {
        let mut episodes = Vec::new();
        for i in 0..10u64 {
            episodes.push(ep(i, vec![1.0, 0.0], vec![7]));
        }
        for i in 10..20u64 {
            episodes.push(ep(i, vec![1.0, 0.1], vec![42]));
        }
        let a = build_cluster((0..10).collect(), &episodes);
        let b = build_cluster((10..20).collect(), &episodes);
        let refined = refine(vec![a, b], &episodes, &CausalConfig::default());
        assert_eq!(refined.len(), 2, "disjoint futures must not merge");
    }

    #[test]
    fn subgroups_below_min_split_pool_into_remainder() {
        // One dominant future group plus two stragglers: the stragglers
        // can't stand alone, so they pool while the dominant group splits
        // clean.
        let mut episodes = Vec::new();
        for i in 0..12u64 {
            episodes.push(ep(i, vec![0.5, 0.5], vec![7]));
        }
        episodes.push(ep(100, vec![0.5, 0.5], vec![88]));
        episodes.push(ep(101, vec![0.5, 0.5], vec![99]));

        let cluster = whole_batch_cluster(&episodes);
        let cfg = CausalConfig { min_split: 3, ..Default::default() };
        let refined = refine(vec![cluster], &episodes, &cfg);

        // Dominant group of 12 + remainder of 2. The remainder's futures
        // (88, 99) diverge from the dominant group's (7), so no merge.
        assert_eq!(refined.len(), 2);
        let sizes: Vec<usize> = refined.iter().map(|c| c.episode_indices.len()).collect();
        assert!(sizes.contains(&12) && sizes.contains(&2), "sizes = {:?}", sizes);
    }

    #[test]
    fn single_future_cluster_passes_through() {
        let episodes: Vec<Episode> =
            (0..8u64).map(|i| ep(i, vec![0.1, 0.2], vec![5, 6])).collect();
        let cluster = whole_batch_cluster(&episodes);
        let refined = refine(vec![cluster], &episodes, &CausalConfig::default());
        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0].episode_indices.len(), 8);
    }

    #[test]
    fn jsd_bounds() {
        let mut p = HashMap::new();
        p.insert(1u32, 1.0f32);
        let mut q = HashMap::new();
        q.insert(2u32, 1.0f32);
        // Disjoint support: exactly 1 bit.
        assert!((jensen_shannon_bits(&p, &q) - 1.0).abs() < 1e-6);
        // Identity: exactly 0.
        assert!(jensen_shannon_bits(&p, &p).abs() < 1e-9);
    }
}
