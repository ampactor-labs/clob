//! Expert router — top-k softmax routing for MoE layers.

use crate::tensor::Tensor;
use rand::Rng;

/// Routing result: selected expert indices and weights.
pub struct RouteResult {
    pub expert_indices: Vec<usize>,
    pub expert_weights: Vec<f32>,
}

/// Top-k expert router. Dense f32 (NOT ternary) for routing precision.
pub struct ExpertRouter {
    weights: Vec<f32>,
    n_experts: usize,
    d_model: usize,
    top_k: usize,
}

impl ExpertRouter {
    pub fn random(d_model: usize, n_experts: usize, top_k: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        let weights: Vec<f32> = (0..n_experts * d_model)
            .map(|_| rng.gen_range(-scale..scale)).collect();
        Self { weights, n_experts, d_model, top_k }
    }

    pub fn from_weights(weights: Vec<f32>, d_model: usize, n_experts: usize, top_k: usize) -> Self {
        assert_eq!(weights.len(), n_experts * d_model);
        Self { weights, n_experts, d_model, top_k }
    }

    /// Route hidden state to top-k experts.
    pub fn route(&self, hidden: &Tensor) -> RouteResult {
        assert_eq!(hidden.len(), self.d_model);

        let mut logits = vec![0.0f32; self.n_experts];
        for e in 0..self.n_experts {
            let start = e * self.d_model;
            let w = &self.weights[start..start + self.d_model];
            logits[e] = hidden.data().iter().zip(w.iter()).map(|(h, w)| h * w).sum();
        }

        // Softmax
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() { *p /= sum; }

        // Top-k
        let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = self.top_k.min(self.n_experts);
        let expert_indices: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
        let raw_weights: Vec<f32> = indexed[..k].iter().map(|(_, w)| *w).collect();

        let w_sum: f32 = raw_weights.iter().sum();
        let expert_weights: Vec<f32> = raw_weights.iter().map(|w| w / w_sum).collect();

        RouteResult { expert_indices, expert_weights }
    }

    pub fn n_experts(&self) -> usize { self.n_experts }
}
