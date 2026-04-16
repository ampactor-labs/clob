//! Experience replay — sample mini-batches from episodic memory
//! for online learning. Prioritized by prediction error.

use crate::memory::episode::Episode;
use rand::Rng;

/// Sample a mini-batch from episodes, prioritized by prediction error.
pub fn sample_batch<'a>(
    episodes: &'a [Episode],
    batch_size: usize,
    rng: &mut impl Rng,
) -> Vec<&'a Episode> {
    if episodes.is_empty() { return Vec::new(); }
    let batch_size = batch_size.min(episodes.len());

    // Compute sampling weights (proportional to prediction error)
    let errors: Vec<f32> = episodes.iter()
        .map(|e| e.prediction_error.max(0.01)) // minimum weight to avoid starvation
        .collect();
    let total: f32 = errors.iter().sum();

    let mut batch = Vec::with_capacity(batch_size);
    let mut selected = vec![false; episodes.len()];

    for _ in 0..batch_size {
        let r: f32 = rng.gen::<f32>() * total;
        let mut cumulative = 0.0;
        let mut idx = 0;
        for (i, &e) in errors.iter().enumerate() {
            cumulative += e;
            if cumulative >= r && !selected[i] {
                idx = i;
                break;
            }
        }
        // Fallback: pick first unselected
        if selected[idx] {
            idx = selected.iter().position(|&s| !s).unwrap_or(0);
        }
        selected[idx] = true;
        batch.push(&episodes[idx]);
    }

    batch
}

/// Compute the correction target for an episode.
/// The correction is: what should the hidden state have been?
/// We approximate: if the model predicted wrong, the hidden state
/// should have been closer to states where it predicts correctly.
pub fn compute_correction(episode: &Episode, cluster_centroid: &[f32]) -> Vec<f32> {
    let d = episode.hidden_state.len();
    let mut correction = vec![0.0f32; d];

    // Correction = error-weighted direction toward centroid
    let scale = episode.prediction_error;
    for i in 0..d {
        correction[i] = scale * (cluster_centroid[i] - episode.hidden_state[i]);
    }

    correction
}
