//! Energy Critic — novelty detection via hidden state projection.
//!
//! Projects hidden state → scalar energy. Low = familiar, high = novel.
//! Maintains an adaptive baseline via exponential moving average.

use crate::tensor::Tensor;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Energy critic with adaptive novelty threshold.
#[derive(Clone, Serialize, Deserialize)]
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
    /// Number of training steps applied.
    n_trained: u64,
    /// Running sum of squared errors between predicted and target (NLL), for diagnostics.
    train_sse: f64,
}

impl EnergyCritic {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self {
            weights, bias, baseline: 0.0, decay: 0.99,
            n_observed: 0, n_trained: 0, train_sse: 0.0,
        }
    }

    pub fn random(d_model: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        Self {
            weights: (0..d_model).map(|_| rng.gen_range(-scale..scale)).collect(),
            bias: 0.0,
            baseline: 0.0,
            decay: 0.99,
            n_observed: 0,
            n_trained: 0,
            train_sse: 0.0,
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

    /// Raw (signed) linear prediction — used as the training target.
    /// `score()` returns its absolute value; training operates on the signed form.
    pub fn predict(&self, hidden: &[f32]) -> f32 {
        assert_eq!(hidden.len(), self.weights.len());
        let dot: f32 = hidden.iter().zip(self.weights.iter()).map(|(h, w)| h * w).sum();
        dot + self.bias
    }

    /// SGD step on MSE(predict(h), target). Plain gradient — no momentum.
    /// Returns the squared error for this sample.
    ///
    /// Training target is typically NLL(actual | h), so the critic learns to
    /// estimate expected surprise from hidden state alone, making `score()` a
    /// principled novelty signal rather than a random projection.
    pub fn train_step(&mut self, hidden: &[f32], target: f32, lr: f32) -> f32 {
        assert_eq!(hidden.len(), self.weights.len());
        let pred = self.predict(hidden);
        let err = pred - target;
        // dL/dw_i = 2 * err * h_i ; dL/db = 2 * err. Absorb the 2 into lr.
        for (w, &h) in self.weights.iter_mut().zip(hidden.iter()) {
            *w -= lr * err * h;
        }
        self.bias -= lr * err;
        self.n_trained += 1;
        self.train_sse += (err * err) as f64;
        err * err
    }

    /// Mean squared error across all training steps seen, or 0 if none.
    pub fn train_mse(&self) -> f64 {
        if self.n_trained == 0 { 0.0 } else { self.train_sse / self.n_trained as f64 }
    }

    pub fn n_trained(&self) -> u64 { self.n_trained }

    /// Serialize to bytes (bincode).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("serialize critic")
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
