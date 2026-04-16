//! Energy Critic — novelty detection via hidden state projection.
//!
//! Projects hidden state → scalar energy. Low = familiar, high = novel.
//! Maintains an adaptive baseline via exponential moving average.

use crate::tensor::Tensor;
use rand::Rng;

/// Energy critic with adaptive novelty threshold.
pub struct EnergyCritic {
    /// Projection weights: [d_model]
    weights: Vec<f32>,
    bias: f32,
    /// EMA of recent energy scores (the baseline).
    baseline: f32,
    /// EMA decay factor.
    decay: f32,
    /// Number of scores observed.
    n_observed: u64,
}

impl EnergyCritic {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias, baseline: 0.0, decay: 0.99, n_observed: 0 }
    }

    pub fn random(d_model: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        Self {
            weights: (0..d_model).map(|_| rng.gen_range(-scale..scale)).collect(),
            bias: 0.0,
            baseline: 0.0,
            decay: 0.99,
            n_observed: 0,
        }
    }

    /// Compute raw energy score (no threshold check).
    pub fn score(&self, hidden_state: &Tensor) -> f32 {
        assert_eq!(hidden_state.len(), self.weights.len());
        let dot: f32 = hidden_state.data().iter()
            .zip(self.weights.iter())
            .map(|(h, w)| h * w)
            .sum();
        (dot + self.bias).abs()
    }

    /// Score AND update the baseline. Returns (energy, is_novel).
    pub fn score_and_detect(&mut self, hidden_state: &Tensor) -> (f32, bool) {
        let energy = self.score(hidden_state);
        self.n_observed += 1;

        if self.n_observed == 1 {
            self.baseline = energy;
            return (energy, true); // Everything is novel at first
        }

        let is_novel = energy > self.baseline * 1.5; // Weber's Law: 50% above baseline

        // Update EMA baseline
        self.baseline = self.decay * self.baseline + (1.0 - self.decay) * energy;

        (energy, is_novel)
    }

    /// Current baseline energy.
    pub fn baseline(&self) -> f32 { self.baseline }

    /// Total observations.
    pub fn n_observed(&self) -> u64 { self.n_observed }

    pub fn dim(&self) -> usize { self.weights.len() }
}
