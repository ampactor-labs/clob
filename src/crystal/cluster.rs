//! Episode clustering — groups episodes by hidden-state similarity.
//!
//! This is the DISTILL (step 1) of the crystallization loop.

use crate::memory::episode::Episode;

/// A cluster of related episodes.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Indices into the original episode list.
    pub episode_indices: Vec<usize>,
    /// Centroid (average hidden state).
    pub centroid: Vec<f32>,
    /// Average prediction error across the cluster.
    pub avg_error: f32,
    /// Average energy score.
    pub avg_energy: f32,
}

/// Cluster episodes by hidden-state similarity using k-means.
pub fn cluster_episodes(episodes: &[Episode], k: usize, max_iters: usize) -> Vec<Cluster> {
    if episodes.is_empty() || k == 0 { return Vec::new(); }

    let k = k.min(episodes.len());
    let _d = episodes[0].hidden_state.len();

    // Initialize centroids from first k episodes
    let mut centroids: Vec<Vec<f32>> = episodes.iter().take(k)
        .map(|e| e.hidden_state.clone())
        .collect();

    let mut assignments = vec![0usize; episodes.len()];

    for _iter in 0..max_iters {
        let mut changed = false;

        // Assign each episode to nearest centroid
        for (i, ep) in episodes.iter().enumerate() {
            let mut best_dist = f32::INFINITY;
            let mut best_k = 0;

            for (ki, centroid) in centroids.iter().enumerate() {
                let dist = squared_distance(&ep.hidden_state, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ki;
                }
            }

            if assignments[i] != best_k {
                assignments[i] = best_k;
                changed = true;
            }
        }

        if !changed { break; }

        // Recompute centroids
        for (ki, centroid) in centroids.iter_mut().enumerate() {
            let mut count = 0usize;
            for v in centroid.iter_mut() { *v = 0.0; }

            for (i, ep) in episodes.iter().enumerate() {
                if assignments[i] == ki {
                    for (c, h) in centroid.iter_mut().zip(ep.hidden_state.iter()) {
                        *c += h;
                    }
                    count += 1;
                }
            }

            if count > 0 {
                for v in centroid.iter_mut() { *v /= count as f32; }
            }
        }
    }

    // Build cluster structs
    let mut clusters = Vec::with_capacity(k);
    for ki in 0..k {
        let episode_indices: Vec<usize> = assignments.iter().enumerate()
            .filter(|(_, &a)| a == ki)
            .map(|(i, _)| i)
            .collect();

        if episode_indices.is_empty() { continue; }

        let avg_error = episode_indices.iter()
            .map(|&i| episodes[i].prediction_error)
            .sum::<f32>() / episode_indices.len() as f32;

        let avg_energy = episode_indices.iter()
            .map(|&i| episodes[i].energy)
            .sum::<f32>() / episode_indices.len() as f32;

        clusters.push(Cluster {
            episode_indices,
            centroid: centroids[ki].clone(),
            avg_error,
            avg_energy,
        });
    }

    // Sort by average error descending (highest error clusters first — most useful to crystallize)
    clusters.sort_by(|a, b| b.avg_error.partial_cmp(&a.avg_error).unwrap());

    clusters
}

fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
