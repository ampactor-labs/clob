//! AdamW optimizer on latent weights with ternary re-quantization.

use crate::learn::grad::{GradBuffer, LatentWeights};

/// AdamW optimizer state.
pub struct AdamW {
    /// First moment (mean).
    m: Vec<f32>,
    /// Second moment (variance).
    v: Vec<f32>,
    /// Learning rate.
    pub lr: f32,
    /// Weight decay.
    pub weight_decay: f32,
    /// Beta1 (momentum).
    pub beta1: f32,
    /// Beta2 (RMSprop).
    pub beta2: f32,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Gradient clipping (max norm).
    pub clip_norm: f32,
    /// Step counter.
    pub step: u64,
}

impl AdamW {
    pub fn new(n_params: usize) -> Self {
        Self {
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            lr: 1e-3,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            clip_norm: 1.0,
            step: 0,
        }
    }

    /// Apply one optimization step.
    /// Returns true if the ternary pattern changed.
    pub fn step(&mut self, latent: &mut LatentWeights, grads: &GradBuffer) -> bool {
        self.step += 1;
        let n = latent.weights.len();
        assert_eq!(grads.grads.len(), n);

        // Gradient clipping
        let grad_norm: f32 = grads.grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        let clip_scale = if grad_norm > self.clip_norm {
            self.clip_norm / grad_norm
        } else {
            1.0
        };

        // Bias correction
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);

        for i in 0..n {
            let g = grads.grads[i] * clip_scale;

            // Update moments
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias-corrected estimates
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // AdamW update (decoupled weight decay)
            latent.weights[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps)
                + self.weight_decay * latent.weights[i]);
        }

        // Re-ternarize and check if pattern changed
        latent.re_ternarize()
    }

    /// Learning rate with warmup + cosine decay.
    pub fn lr_schedule(&mut self, warmup_steps: u64, total_steps: u64) {
        if self.step < warmup_steps {
            self.lr = 1e-3 * (self.step as f32 / warmup_steps as f32);
        } else {
            let progress = (self.step - warmup_steps) as f32
                / (total_steps - warmup_steps).max(1) as f32;
            self.lr = 1e-3 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        }
    }
}
