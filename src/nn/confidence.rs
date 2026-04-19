//! Confidence head — predicts the per-step cross-entropy from the hidden state.
//!
//! This is Head C of the multi-head critic (see plan "one critic, one meter,
//! five phases"). It has the same shape and training loop as `EnergyCritic`
//! (Head N), but with an independent weight vector so the two heads don't
//! cannibalize each other — the gradients for "I'm surprised" (Head N) and
//! "I'm about to be wrong" (Head C) collide if they share projection weights.
//!
//! Head C's output is consumed by the adaptive-decode loop in
//! `CoreModel::decode_step`: when Head C's own novelty detector z-score
//! crosses a threshold, the block stack is iterated `k` extra times on the
//! current buffer before unembedding — the poor-man's "pondering" loop.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// A standalone linear probe over the hidden state.
///
/// Trained via MSE(predict(h), target) — same primitive as EnergyCritic's
/// `train_step`. For Head C, the target is the actual NLL of the true next
/// token; the head learns to estimate expected surprise ahead of seeing it.
#[derive(Clone, Serialize, Deserialize)]
pub struct ConfidenceHead {
    weights: Vec<f32>,
    bias: f32,
    n_trained: u64,
    train_sse: f64,
}

impl ConfidenceHead {
    pub fn new(weights: Vec<f32>, bias: f32) -> Self {
        Self { weights, bias, n_trained: 0, train_sse: 0.0 }
    }

    pub fn random(d_model: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        Self {
            weights: (0..d_model).map(|_| rng.gen_range(-scale..scale)).collect(),
            bias: 0.0,
            n_trained: 0,
            train_sse: 0.0,
        }
    }

    pub fn dim(&self) -> usize { self.weights.len() }
    pub fn n_trained(&self) -> u64 { self.n_trained }

    /// Predicted per-step NLL. Clamped to `>= 0` since NLL is non-negative —
    /// an untrained head can emit negatives which the detector would then
    /// misinterpret.
    pub fn predict(&self, hidden: &[f32]) -> f32 {
        assert_eq!(hidden.len(), self.weights.len());
        let dot: f32 = hidden.iter().zip(self.weights.iter()).map(|(h, w)| h * w).sum();
        (dot + self.bias).max(0.0)
    }

    /// Raw (signed) prediction — what `train_step` operates on. Exposed for
    /// diagnostics; most callers want `predict`.
    pub fn predict_raw(&self, hidden: &[f32]) -> f32 {
        let dot: f32 = hidden.iter().zip(self.weights.iter()).map(|(h, w)| h * w).sum();
        dot + self.bias
    }

    /// SGD step on MSE(predict_raw(h), target_nll). Returns squared error.
    pub fn train_step(&mut self, hidden: &[f32], target_nll: f32, lr: f32) -> f32 {
        assert_eq!(hidden.len(), self.weights.len());
        let pred = self.predict_raw(hidden);
        let err = pred - target_nll;
        for (w, &h) in self.weights.iter_mut().zip(hidden.iter()) {
            *w -= lr * err * h;
        }
        self.bias -= lr * err;
        self.n_trained += 1;
        self.train_sse += (err * err) as f64;
        err * err
    }

    pub fn train_mse(&self) -> f64 {
        if self.n_trained == 0 { 0.0 } else { self.train_sse / self.n_trained as f64 }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("serialize confidence head")
    }

    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Adaptive-decode configuration. When a model has a `ConfidenceHead` installed
/// and `enabled` is true, the block stack iterates up to `max_extra_steps`
/// additional times on any token whose predicted NLL z-score exceeds
/// `z_threshold`.
#[derive(Clone, Copy, Debug)]
pub struct AdaptiveConfig {
    pub enabled: bool,
    pub max_extra_steps: u8,
    pub z_threshold: f32,
    /// Meta-critic reliability threshold: when the MetaCritic predicts
    /// Head C's absolute error above this, we SKIP adaptive iterations
    /// rather than take them (Head C's uncertainty signal is itself
    /// unreliable at this hidden state). Only consulted if the model has
    /// a MetaCritic installed.
    pub meta_unreliable_threshold: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_extra_steps: 4,
            z_threshold: 1.0,
            meta_unreliable_threshold: 2.0,
        }
    }
}

impl AdaptiveConfig {
    pub fn enabled_default() -> Self {
        Self { enabled: true, ..Self::default() }
    }
}

/// Meta-critic ("critic of Head C") — predicts the absolute error of the
/// confidence head: `|head.predict(h) - actual_nll|`. Trained via a
/// target-network: its training label is produced by a *frozen snapshot*
/// of the confidence head, refreshed every `refresh_period` training steps.
///
/// The frozen snapshot is the standard fix for self-signal feedback
/// oscillation. If MetaCritic trained against the *live* head, positive
/// feedback would couple the two heads' parameter updates and cause their
/// MSEs to co-drift. Training against a lagged target breaks that loop.
///
/// Consumer: `CoreModel::decode_step` reads `predict(hidden)` to decide
/// whether Head C is trustworthy at this state. High MetaCritic output =
/// "Head C is unreliable here, don't take adaptive steps".
#[derive(Clone, Serialize, Deserialize)]
pub struct MetaCritic {
    weights: Vec<f32>,
    bias: f32,
    /// The frozen Head-C snapshot used as training target. Refreshed every
    /// `refresh_period` training steps.
    target_snapshot: ConfidenceHead,
    refresh_period: u64,
    steps_since_refresh: u64,
    n_trained: u64,
    train_sse: f64,
}

impl MetaCritic {
    pub fn new(d_model: usize, rng: &mut impl Rng, initial_head: ConfidenceHead, refresh_period: u64) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        Self {
            weights: (0..d_model).map(|_| rng.gen_range(-scale..scale)).collect(),
            bias: 0.0,
            target_snapshot: initial_head,
            refresh_period: refresh_period.max(1),
            steps_since_refresh: 0,
            n_trained: 0,
            train_sse: 0.0,
        }
    }

    pub fn dim(&self) -> usize { self.weights.len() }
    pub fn n_trained(&self) -> u64 { self.n_trained }
    pub fn refresh_period(&self) -> u64 { self.refresh_period }
    pub fn steps_since_refresh(&self) -> u64 { self.steps_since_refresh }

    /// Predicted absolute error of Head C at this hidden state. Non-negative
    /// by construction (MetaCritic's target is `|...|`).
    pub fn predict(&self, hidden: &[f32]) -> f32 {
        assert_eq!(hidden.len(), self.weights.len());
        let dot: f32 = hidden.iter().zip(self.weights.iter()).map(|(h, w)| h * w).sum();
        (dot + self.bias).max(0.0)
    }

    fn predict_raw(&self, hidden: &[f32]) -> f32 {
        let dot: f32 = hidden.iter().zip(self.weights.iter()).map(|(h, w)| h * w).sum();
        dot + self.bias
    }

    /// One training step. The label comes from the FROZEN target snapshot,
    /// not the live head — this is the target-network trick that keeps
    /// training stable. Refreshes the snapshot from `live_head` every
    /// `refresh_period` steps.
    pub fn train_step(
        &mut self,
        hidden: &[f32],
        actual_nll: f32,
        live_head: &ConfidenceHead,
        lr: f32,
    ) -> f32 {
        assert_eq!(hidden.len(), self.weights.len());

        let target_prediction = self.target_snapshot.predict_raw(hidden);
        let label = (target_prediction - actual_nll).abs();

        let pred = self.predict_raw(hidden);
        let err = pred - label;
        for (w, &h) in self.weights.iter_mut().zip(hidden.iter()) {
            *w -= lr * err * h;
        }
        self.bias -= lr * err;
        self.n_trained += 1;
        self.train_sse += (err * err) as f64;

        self.steps_since_refresh += 1;
        if self.steps_since_refresh >= self.refresh_period {
            self.target_snapshot = live_head.clone();
            self.steps_since_refresh = 0;
        }

        err * err
    }

    pub fn train_mse(&self) -> f64 {
        if self.n_trained == 0 { 0.0 } else { self.train_sse / self.n_trained as f64 }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("serialize meta critic")
    }

    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_is_nonnegative() {
        let mut rng = rand::rngs::StdRng::from_seed([7u8; 32]);
        let head = ConfidenceHead::random(16, &mut rng);
        let h: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // Default ConfidenceHead::random can emit negative dot products; predict clamps.
        let p = head.predict(&h);
        assert!(p >= 0.0);
    }

    #[test]
    fn train_reduces_error() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_seed([11u8; 32]);
        let mut head = ConfidenceHead::random(8, &mut rng);
        let h = vec![1.0f32, 0.0, -0.5, 0.25, 0.0, 0.1, -0.3, 0.8];
        let target = 2.5f32;

        let first_err = head.train_step(&h, target, 0.05);
        for _ in 0..50 {
            head.train_step(&h, target, 0.05);
        }
        let last_err = (head.predict_raw(&h) - target).powi(2);
        assert!(last_err < first_err, "head did not converge: first={}, last={}", first_err, last_err);
        assert!(last_err < 0.01);
    }

    #[test]
    fn serde_roundtrip() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_seed([13u8; 32]);
        let head = ConfidenceHead::random(32, &mut rng);
        let bytes = head.to_bytes();
        let restored = ConfidenceHead::from_bytes(&bytes).unwrap();
        assert_eq!(restored.dim(), 32);
        assert_eq!(restored.weights, head.weights);
        assert_eq!(restored.bias, head.bias);
    }

    #[test]
    fn meta_target_network_refreshes_on_schedule() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_seed([17u8; 32]);
        let d = 8;
        let initial_head = ConfidenceHead::random(d, &mut rng);
        let mut meta = MetaCritic::new(d, &mut rng, initial_head.clone(), 5);

        // Simulate a drifting live head: after each meta step, perturb the
        // live head's bias slightly. MetaCritic's snapshot should only pick
        // up the change every 5 training steps.
        let mut live = initial_head.clone();
        let initial_target_bias = meta.target_snapshot.predict_raw(&vec![0.0; d]);
        let h = vec![0.1f32; d];

        for i in 1..=10 {
            live.bias += 0.5;
            meta.train_step(&h, 1.0, &live, 0.01);
            let current_target_bias = meta.target_snapshot.predict_raw(&vec![0.0; d]);
            if i < 5 {
                // Before the first refresh, target bias should match the initial.
                assert!((current_target_bias - initial_target_bias).abs() < 1e-6,
                    "target refreshed too early at step {}", i);
            }
            if i == 5 || i == 10 {
                // After a refresh, target should reflect live's drift.
                assert!((current_target_bias - initial_target_bias).abs() > 0.1,
                    "target failed to refresh at step {}", i);
            }
        }
    }

    #[test]
    fn meta_predict_nonnegative() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::from_seed([19u8; 32]);
        let d = 16;
        let head = ConfidenceHead::random(d, &mut rng);
        let meta = MetaCritic::new(d, &mut rng, head, 100);
        let h: Vec<f32> = (0..d).map(|i| (i as f32) - 8.0).collect();
        assert!(meta.predict(&h) >= 0.0);
    }
}

// Re-import for the test closures above (they use SeedableRng).
#[cfg(test)]
use rand::SeedableRng;
