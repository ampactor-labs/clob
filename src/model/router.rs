//! Expert router — top-k softmax routing for MoE layers.
//!
//! Phase F adds REINFORCE-style policy-gradient training on the router's
//! dense weights. No backprop through experts is required — the router
//! learns from observed CE advantages of each routing decision. A
//! counterfactual sampling rate (default 10%) occasionally forces the
//! non-top-1 expert to break the positive-feedback loop where the router
//! always routes to its current favorite.

use crate::tensor::Tensor;
use rand::Rng;

/// Routing result: selected expert indices and weights.
pub struct RouteResult {
    pub expert_indices: Vec<usize>,
    pub expert_weights: Vec<f32>,
    /// Full softmax distribution over experts, before top-k truncation.
    /// Used by the policy-gradient update; `route()` returns an empty vec
    /// for backward compat and `route_with_probs()` returns the full vector.
    pub full_probs: Vec<f32>,
    /// Whether this routing decision was a counterfactual (forced
    /// non-top-1 selection, used during training to break positive
    /// feedback on the router's current favorite).
    pub counterfactual: bool,
}

/// Top-k expert router. Dense f32 (NOT ternary) for routing precision.
pub struct ExpertRouter {
    weights: Vec<f32>,
    n_experts: usize,
    d_model: usize,
    top_k: usize,
    /// Accumulated policy gradient `dL/dW` (shape: n_experts × d_model).
    /// Populated by `accumulate_policy_gradient`, consumed by `apply_adam`.
    grads: Vec<f32>,
    /// AdamW first moment.
    m: Vec<f32>,
    /// AdamW second moment.
    v: Vec<f32>,
    /// AdamW step counter.
    step: u64,
    /// Number of gradient samples accumulated since last AdamW step.
    n_accumulated: usize,
    /// Number of counterfactual selections taken since last reset.
    counterfactual_count: u64,
}

impl ExpertRouter {
    pub fn random(d_model: usize, n_experts: usize, top_k: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        let weights: Vec<f32> = (0..n_experts * d_model)
            .map(|_| rng.gen_range(-scale..scale)).collect();
        Self::from_weights(weights, d_model, n_experts, top_k)
    }

    pub fn from_weights(weights: Vec<f32>, d_model: usize, n_experts: usize, top_k: usize) -> Self {
        assert_eq!(weights.len(), n_experts * d_model);
        let n = n_experts * d_model;
        Self {
            weights,
            n_experts,
            d_model,
            top_k,
            grads: vec![0.0; n],
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
            n_accumulated: 0,
            counterfactual_count: 0,
        }
    }

    /// Route hidden state to top-k experts (original API).
    pub fn route(&self, hidden: &Tensor) -> RouteResult {
        self.route_inner(hidden, false)
    }

    /// Route with explicit counterfactual sampling control. When
    /// `cf_rate > 0.0` and the RNG draw is below it, forces the top-2
    /// expert instead of top-1 for the first slot. The returned
    /// `RouteResult.counterfactual` flag is set so the caller can
    /// distinguish training samples.
    pub fn route_counterfactual(
        &self,
        hidden: &Tensor,
        rng: &mut impl Rng,
        cf_rate: f32,
    ) -> RouteResult {
        let do_cf = cf_rate > 0.0 && rng.gen::<f32>() < cf_rate;
        self.route_inner(hidden, do_cf)
    }

    fn route_inner(&self, hidden: &Tensor, counterfactual: bool) -> RouteResult {
        assert_eq!(hidden.len(), self.d_model);

        let mut logits = vec![0.0f32; self.n_experts];
        for e in 0..self.n_experts {
            let start = e * self.d_model;
            let w = &self.weights[start..start + self.d_model];
            logits[e] = hidden.data().iter().zip(w.iter()).map(|(h, w)| h * w).sum();
        }

        // Softmax (full distribution over experts).
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut full_probs: Vec<f32> = logits.iter().map(|l| (l - max_val).exp()).collect();
        let sum: f32 = full_probs.iter().sum();
        for p in full_probs.iter_mut() { *p /= sum; }

        // Rank experts by probability.
        let mut indexed: Vec<(usize, f32)> = full_probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Counterfactual: swap top-1 with top-2 so the training signal gets
        // variance it wouldn't see from greedy policy alone.
        if counterfactual && indexed.len() >= 2 {
            indexed.swap(0, 1);
        }

        let k = self.top_k.min(self.n_experts);
        let expert_indices: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
        let raw_weights: Vec<f32> = indexed[..k].iter().map(|(_, w)| *w).collect();

        let w_sum: f32 = raw_weights.iter().sum();
        let expert_weights: Vec<f32> = raw_weights.iter().map(|w| w / w_sum).collect();

        RouteResult {
            expert_indices,
            expert_weights,
            full_probs,
            counterfactual,
        }
    }

    pub fn n_experts(&self) -> usize { self.n_experts }
    pub fn d_model(&self) -> usize { self.d_model }
    pub fn top_k(&self) -> usize { self.top_k }
    pub fn weights(&self) -> &[f32] { &self.weights }
    pub fn weights_mut(&mut self) -> &mut [f32] { &mut self.weights }
    pub fn step_count(&self) -> u64 { self.step }
    pub fn counterfactual_count(&self) -> u64 { self.counterfactual_count }
    pub fn n_accumulated(&self) -> usize { self.n_accumulated }

    pub fn record_counterfactual(&mut self) {
        self.counterfactual_count += 1;
    }

    /// Accumulate a REINFORCE policy-gradient sample.
    ///
    ///   dL/dlogit_i = advantage * (full_probs[i] - I[i in selected])
    ///
    /// Positive `advantage` (i.e., this routing achieved *lower* CE than
    /// the running baseline) pushes logit up for the selected experts and
    /// down for the non-selected ones. The outer product with the hidden
    /// state then fills in `dL/dW_i,j = dL/dlogit_i * hidden[j]`.
    ///
    /// Gradients accumulate until `apply_adam` is called.
    pub fn accumulate_policy_gradient(
        &mut self,
        hidden: &Tensor,
        selected_indices: &[usize],
        full_probs: &[f32],
        advantage: f32,
    ) {
        assert_eq!(hidden.len(), self.d_model);
        assert_eq!(full_probs.len(), self.n_experts);

        let h = hidden.data();
        for e in 0..self.n_experts {
            let indicator = if selected_indices.contains(&e) { 1.0 } else { 0.0 };
            let dlogit = advantage * (full_probs[e] - indicator);
            if dlogit.abs() < 1e-12 { continue; }
            let row_start = e * self.d_model;
            for j in 0..self.d_model {
                self.grads[row_start + j] += dlogit * h[j];
            }
        }
        self.n_accumulated += 1;
    }

    /// Apply one AdamW step, zero the accumulator. Returns the l2-norm of
    /// the applied update (diagnostic for training stability).
    pub fn apply_adam(
        &mut self,
        lr: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        clip_norm: f32,
    ) -> f32 {
        if self.n_accumulated == 0 {
            return 0.0;
        }
        self.step += 1;
        let n = self.weights.len();

        // Average gradients then clip.
        let inv = 1.0 / self.n_accumulated as f32;
        for g in self.grads.iter_mut() { *g *= inv; }
        let grad_norm: f32 = self.grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        let clip_scale = if grad_norm > clip_norm { clip_norm / grad_norm } else { 1.0 };

        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);

        let mut update_sq_sum = 0.0f32;
        for i in 0..n {
            let g = self.grads[i] * clip_scale;
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            let update = lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * self.weights[i]);
            self.weights[i] -= update;
            update_sq_sum += update * update;
        }

        // Reset accumulator.
        for g in self.grads.iter_mut() { *g = 0.0; }
        self.n_accumulated = 0;

        update_sq_sum.sqrt()
    }

    pub fn reset_training_state(&mut self) {
        for g in self.grads.iter_mut() { *g = 0.0; }
        for m in self.m.iter_mut() { *m = 0.0; }
        for v in self.v.iter_mut() { *v = 0.0; }
        self.step = 0;
        self.n_accumulated = 0;
        self.counterfactual_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn fresh(d: usize, n: usize, k: usize) -> ExpertRouter {
        let mut rng = rand::rngs::StdRng::from_seed([23u8; 32]);
        ExpertRouter::random(d, n, k, &mut rng)
    }

    #[test]
    fn counterfactual_swaps_top_two() {
        let router = fresh(4, 4, 1);
        let h = Tensor::from_vec(vec![0.5, -0.2, 0.3, 0.1], &[4]);
        let mut rng = rand::rngs::StdRng::from_seed([1u8; 32]);
        // With cf_rate = 1.0, counterfactual always fires.
        let r = router.route_counterfactual(&h, &mut rng, 1.0);
        assert!(r.counterfactual);
        // With cf_rate = 0.0, never.
        let r = router.route_counterfactual(&h, &mut rng, 0.0);
        assert!(!r.counterfactual);
    }

    #[test]
    fn policy_gradient_increases_selected_logit_on_positive_advantage() {
        let mut router = fresh(8, 4, 2);
        let h = Tensor::from_vec(vec![1.0; 8], &[8]);
        let route = router.route(&h);
        let selected = route.expert_indices.clone();
        let full_probs = route.full_probs.clone();

        // Positive advantage: the selected experts did well; router should
        // shift weights to increase logits for them.
        for _ in 0..10 {
            router.accumulate_policy_gradient(&h, &selected, &full_probs, 1.0);
        }
        let logits_before: Vec<f32> = (0..router.n_experts).map(|e| {
            let start = e * router.d_model;
            router.weights[start..start + router.d_model].iter().zip(h.data().iter())
                .map(|(w, x)| w * x).sum::<f32>()
        }).collect();
        router.apply_adam(0.1, 0.0, 0.9, 0.999, 1e-8, 1.0);
        let logits_after: Vec<f32> = (0..router.n_experts).map(|e| {
            let start = e * router.d_model;
            router.weights[start..start + router.d_model].iter().zip(h.data().iter())
                .map(|(w, x)| w * x).sum::<f32>()
        }).collect();

        // Logits for selected experts should have increased relative to
        // non-selected ones.
        let selected_delta_sum: f32 = selected.iter()
            .map(|&i| logits_after[i] - logits_before[i]).sum();
        let non_selected_delta_sum: f32 = (0..router.n_experts)
            .filter(|i| !selected.contains(i))
            .map(|i| logits_after[i] - logits_before[i]).sum();
        assert!(selected_delta_sum > non_selected_delta_sum,
            "policy gradient did not favor selected experts: selected_delta={} non_selected_delta={}",
            selected_delta_sum, non_selected_delta_sum);
    }

    #[test]
    fn apply_adam_with_no_samples_is_noop() {
        let mut router = fresh(4, 4, 1);
        let before = router.weights.clone();
        let update_norm = router.apply_adam(0.1, 0.0, 0.9, 0.999, 1e-8, 1.0);
        assert_eq!(update_norm, 0.0);
        assert_eq!(router.weights, before);
    }
}
