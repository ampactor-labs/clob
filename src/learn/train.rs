//! Supervised dense-core training — Path B, Phase 4.
//!
//! Streams a token corpus into truncated-BPTT windows, backpropagates through
//! the gradient-checked full-model backward (`backprop.rs`), clips the global
//! gradient norm, and steps the f32 latent weights with AdamW. Recurrent state
//! is detached at every window boundary (each window starts from zero state).
//!
//! This trains the f32 latent *surrogate* — exactly the object `backprop.rs`
//! gradient-checks — and projects to ternary on a cadence to measure the
//! deployment gap. Training the surrogate keeps every step inside the verified
//! regime; forward-through-ternary (STE quantization-aware training) is a later
//! refinement, and its cost is what the cadence's latent→ternary loss gap
//! previews.

use crate::learn::backprop::{init_dense_params, ternarize_matrix_families, DenseDims, DenseModel};

/// Training hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct TrainConfig {
    /// BPTT window length (tokens per step).
    pub window: usize,
    /// AdamW learning rate.
    pub lr: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// Global gradient-norm clip.
    pub clip_norm: f32,
    /// Optimizer steps (windows processed; cycles the corpus).
    pub steps: usize,
    /// Re-ternarize + measure the deployment gap every N steps (0 = never).
    pub ternarize_every: usize,
    /// Emit a log report every N steps (0 = never).
    pub log_every: usize,
    /// Fire the checkpoint callback every N steps (0 = never).
    pub checkpoint_every: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            window: 16,
            lr: 1e-3,
            weight_decay: 0.01,
            clip_norm: 1.0,
            steps: 2000,
            ternarize_every: 0,
            log_every: 100,
            checkpoint_every: 0,
        }
    }
}

/// One logged training point.
#[derive(Debug, Clone)]
pub struct TrainReport {
    pub step: usize,
    /// Mean per-token NLL over the most recent window(s), latent forward.
    pub train_nll: f32,
    /// Mean per-token NLL of the ternary-projected model on the same window,
    /// when a re-ternarization landed on this step.
    pub ternary_nll: Option<f32>,
    /// Global gradient norm before clipping.
    pub grad_norm: f32,
}

/// Result of a training run.
pub struct TrainResult {
    pub dims: DenseDims,
    /// Trained f32 latent parameters (canonical order).
    pub latents: Vec<f32>,
    /// Effective ternary projection of the final latents (deployment weights).
    pub ternary_effective: Vec<f32>,
    /// Token-marginal (unigram) NLL of the corpus — the beat-me baseline.
    pub unigram_nll: f32,
    /// Mean per-token NLL over the final epoch of windows (latent forward).
    pub final_train_nll: f32,
    pub history: Vec<TrainReport>,
}

/// Flat AdamW with decoupled weight decay and global gradient-norm clipping.
struct FlatAdamW {
    m: Vec<f32>,
    v: Vec<f32>,
    lr: f32,
    wd: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    clip: f32,
    t: i32,
}

impl FlatAdamW {
    fn new(n: usize, cfg: &TrainConfig) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            lr: cfg.lr,
            wd: cfg.weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            clip: cfg.clip_norm,
            t: 0,
        }
    }

    /// Step `params` by `grad` (an f64 gradient). Returns the pre-clip global
    /// gradient norm.
    fn step(&mut self, params: &mut [f32], grad: &[f64]) -> f32 {
        self.t += 1;
        let norm = (grad.iter().map(|g| g * g).sum::<f64>().sqrt()) as f32;
        let scale = if norm > self.clip && norm > 0.0 { self.clip / norm } else { 1.0 };
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        for i in 0..params.len() {
            let g = grad[i] as f32 * scale;
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.wd * params[i]);
        }
        norm
    }
}

/// Mean per-token NLL of a dense core over a token stream, evaluated in the
/// same windowed/detached regime it was trained in (state reset every
/// `window` tokens). This is the held-out capability measure Bet 7 tracks:
/// build the model from a checkpoint's params, run it over the *holdout*
/// tokens, and average `-log p(next)`.
pub fn eval_nll(
    dims: DenseDims,
    params: &[f32],
    tokens: &[u32],
    window: usize,
    max_tokens: usize,
) -> f32 {
    let mut model = DenseModel::from_flat(dims, params);
    let limit = if max_tokens == 0 { tokens.len() } else { tokens.len().min(max_tokens) };
    let mut total = 0.0f64;
    let mut n = 0usize;
    let mut i = 0usize;
    while i + 1 < limit {
        model.reset_state();
        for _ in 0..window {
            if i + 1 >= limit {
                break;
            }
            let logits = model.decode_step(tokens[i]);
            // Stable softmax probability of the actual next token.
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &l in &logits {
                sum += (l - max).exp();
            }
            let p = (logits[tokens[i + 1] as usize] - max).exp() / sum.max(1e-30);
            total += -(p.max(1e-12) as f64).ln();
            n += 1;
            i += 1;
        }
    }
    if n == 0 { 0.0 } else { (total / n as f64) as f32 }
}

/// Token-marginal (unigram) NLL, in nats — the baseline a trained core must
/// beat. `-Σ p_i ln p_i` over the corpus token frequencies.
pub fn unigram_nll(tokens: &[u32], vocab: usize) -> f32 {
    let mut counts = vec![0u64; vocab];
    for &t in tokens {
        counts[t as usize] += 1;
    }
    let n = tokens.len().max(1) as f32;
    let mut h = 0.0f32;
    for &c in &counts {
        if c > 0 {
            let p = c as f32 / n;
            h -= p * p.ln();
        }
    }
    h
}

/// A trained dense core, ready to serialize. Holds the f32 latents (to resume
/// training) and their effective ternary projection (deployment weights).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct TrainedDense {
    pub dims: DenseDims,
    pub latents: Vec<f32>,
    pub ternary_effective: Vec<f32>,
}

impl TrainedDense {
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("serialize trained dense")
    }
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_bytes())
    }
}

/// Train a dense core. `on_log` fires at each logged step; `on_checkpoint`
/// fires every `cfg.checkpoint_every` steps with the current latents.
pub fn train_dense(
    dims: DenseDims,
    tokens: &[u32],
    cfg: &TrainConfig,
    seed: u64,
    mut on_log: impl FnMut(&TrainReport),
    mut on_checkpoint: impl FnMut(usize, DenseDims, &[f32]),
) -> TrainResult {
    assert!(tokens.len() > cfg.window, "corpus shorter than one BPTT window");
    let n_windows = (tokens.len() - 1) / cfg.window;
    assert!(n_windows > 0, "need at least one full window");

    let mut params = init_dense_params(dims, seed);
    let mut opt = FlatAdamW::new(params.len(), cfg);
    let base = unigram_nll(tokens, dims.vocab);

    let mut history = Vec::new();
    // Ring of recent per-token losses for a stable final-epoch estimate.
    let mut recent: Vec<f32> = Vec::new();

    let mut targets_buf = vec![0usize; cfg.window];
    for step in 0..cfg.steps {
        let w = step % n_windows;
        let start = w * cfg.window;
        let inputs = &tokens[start..start + cfg.window];
        for (dst, &t) in targets_buf.iter_mut().zip(tokens[start + 1..start + 1 + cfg.window].iter()) {
            *dst = t as usize;
        }

        let model = DenseModel::from_flat(dims, &params);
        let (loss, tape) = model.forward_window(inputs, &targets_buf);
        let per_token = (loss / cfg.window as f64) as f32;

        let mut grad = vec![0.0f64; params.len()];
        model.backward_window(&tape, &mut grad);
        let grad_norm = opt.step(&mut params, &grad);

        recent.push(per_token);
        if recent.len() > n_windows {
            recent.remove(0);
        }

        let do_tern = cfg.ternarize_every > 0 && (step + 1) % cfg.ternarize_every == 0;
        let ternary_nll = if do_tern {
            let eff = ternarize_matrix_families(dims, &params);
            let tmodel = DenseModel::from_flat(dims, &eff);
            let (tloss, _) = tmodel.forward_window(inputs, &targets_buf);
            Some((tloss / cfg.window as f64) as f32)
        } else {
            None
        };

        if (cfg.log_every > 0 && (step + 1) % cfg.log_every == 0)
            || do_tern
            || step + 1 == cfg.steps
        {
            let report = TrainReport { step: step + 1, train_nll: per_token, ternary_nll, grad_norm };
            on_log(&report);
            history.push(report);
        }

        if cfg.checkpoint_every > 0 && (step + 1) % cfg.checkpoint_every == 0 {
            on_checkpoint(step + 1, dims, &params);
        }
    }

    let final_train_nll = if recent.is_empty() {
        0.0
    } else {
        recent.iter().sum::<f32>() / recent.len() as f32
    };
    let ternary_effective = ternarize_matrix_families(dims, &params);

    TrainResult {
        dims,
        latents: params,
        ternary_effective,
        unigram_nll: base,
        final_train_nll,
        history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Phase 4 gate: on a corpus with real structure the trained core's
    /// loss must fall well below the unigram baseline. The pattern [0,1,0,2]
    /// is deliberately memory-requiring — token 0 is followed by 1 at one
    /// phase and 2 at another, so a memoryless map cannot get below ~0.35 nats
    /// and only the recurrence can drive it toward zero. Windows are
    /// phase-locked (length = 2 periods) so the detached-state recurrence can
    /// count position.
    #[test]
    fn trivial_structured_corpus_beats_unigram() {
        let pattern = [0u32, 1, 0, 2];
        let mut tokens = Vec::new();
        for _ in 0..80 {
            tokens.extend_from_slice(&pattern);
        }
        let dims = DenseDims::mini(3);
        let cfg = TrainConfig {
            window: 8,
            lr: 5e-3,
            weight_decay: 0.0,
            clip_norm: 1.0,
            steps: 2000,
            ternarize_every: 0,
            log_every: 0,
            checkpoint_every: 0,
        };
        let res = train_dense(dims, &tokens, &cfg, 1, |_| {}, |_, _, _| {});

        // Unigram baseline for [0,1,0,2]: p0=.5, p1=.25, p2=.25 → ~1.04 nats.
        assert!(
            (res.unigram_nll - 1.0397).abs() < 0.01,
            "unigram baseline {} unexpected",
            res.unigram_nll,
        );
        // Plan gate: well below unigram.
        assert!(
            res.final_train_nll < 0.5 * res.unigram_nll,
            "final NLL {:.4} not well below unigram {:.4}",
            res.final_train_nll, res.unigram_nll,
        );
        // Stronger: below the memoryless floor (~0.347) proves the recurrence
        // actually trained, not just the value→next readout.
        assert!(
            res.final_train_nll < 0.30,
            "final NLL {:.4} did not beat the memoryless floor — recurrence not learning",
            res.final_train_nll,
        );
    }

    #[test]
    fn unigram_nll_matches_hand_computation() {
        // Uniform over 4 tokens → ln 4.
        let tokens: Vec<u32> = (0..4).cycle().take(400).map(|x| x as u32).collect();
        let h = unigram_nll(&tokens, 4);
        assert!((h - (4.0f32).ln()).abs() < 1e-4, "got {}", h);
    }
}
