//! Ternary-faithful gradient computation.
//!
//! Forward pass runs exact ternary math but maintains f32 latent weights
//! behind the ternary mask. Gradients flow via Straight-Through Estimator.

use crate::tensor::ternary::TernaryMatrix;

/// Latent f32 weights behind a ternary matrix.
/// The ternary mask is the "published" version; latent weights track
/// the gradient-updated values that haven't yet changed the ternary pattern.
pub struct LatentWeights {
    /// f32 latent weights [rows × cols], row-major.
    pub weights: Vec<f32>,
    /// Current ternary version (derived from latent weights).
    pub ternary: TernaryMatrix,
    /// Per-row scale factors.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    /// Version counter — incremented when ternary pattern changes.
    pub version: u64,
}

impl LatentWeights {
    /// Initialize from an existing ternary matrix (cold start).
    pub fn from_ternary(mat: &TernaryMatrix) -> Self {
        let rows = mat.rows();
        let cols = mat.cols();
        let trits = mat.unpack();
        let scales = mat.scales().to_vec();

        // Initialize latent weights from trit * scale
        let mut weights = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                weights[r * cols + c] = trits[r * cols + c] as f32 * scales[r];
            }
        }

        Self {
            weights,
            ternary: mat.clone(),
            scales,
            rows,
            cols,
            version: 0,
        }
    }

    /// Initialize from a distilled pattern (warm start for crystallization).
    pub fn from_pattern(avg_input: &[f32], avg_correction: &[f32]) -> Self {
        let d = avg_input.len();
        let input_norm_sq: f32 = avg_input.iter().map(|v| v * v).sum();

        let mut weights = vec![0.0f32; d * d];
        if input_norm_sq > 1e-8 {
            let scale = 1.0 / input_norm_sq;
            for r in 0..d {
                for c in 0..d {
                    weights[r * d + c] = avg_correction[r] * avg_input[c] * scale;
                }
            }
        }

        let (ternary, scales) = ternarize(&weights, d, d);
        Self {
            weights,
            ternary,
            scales,
            rows: d,
            cols: d,
            version: 0,
        }
    }

    /// Re-ternarize latent weights. Returns true if the ternary pattern changed.
    pub fn re_ternarize(&mut self) -> bool {
        let (new_ternary, new_scales) = ternarize(&self.weights, self.rows, self.cols);
        let old_trits = self.ternary.unpack();
        let new_trits = new_ternary.unpack();
        let changed = old_trits != new_trits;
        if changed {
            self.ternary = new_ternary;
            self.scales = new_scales;
            self.version += 1;
        }
        changed
    }
}

/// Gradient buffer for accumulation.
pub struct GradBuffer {
    /// Gradients [rows × cols].
    pub grads: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    /// Number of accumulated samples.
    pub n_accumulated: usize,
}

impl GradBuffer {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            grads: vec![0.0; rows * cols],
            rows,
            cols,
            n_accumulated: 0,
        }
    }

    pub fn zero(&mut self) {
        self.grads.fill(0.0);
        self.n_accumulated = 0;
    }

    /// Accumulate gradient from a single sample.
    /// STE: gradient of the ternary quantization is 1 where |w| < threshold.
    pub fn accumulate_ste(&mut self, latent: &LatentWeights, input: &[f32], output_grad: &[f32]) {
        assert_eq!(input.len(), self.cols);
        assert_eq!(output_grad.len(), self.rows);

        // dL/dW = output_grad ⊗ input (outer product)
        // STE mask: pass gradient through where |latent_w| is close to threshold
        for r in 0..self.rows {
            let scale = latent.scales[r];
            let threshold = if scale > 0.0 { scale * 0.5 } else { 0.5 };
            for c in 0..self.cols {
                let w = latent.weights[r * self.cols + c].abs();
                // STE: pass gradient through if weight is in the "active zone"
                let ste_mask = if w < threshold * 3.0 { 1.0 } else { 0.3 };
                self.grads[r * self.cols + c] += output_grad[r] * input[c] * ste_mask;
            }
        }
        self.n_accumulated += 1;
    }

    /// Average gradients.
    pub fn average(&mut self) {
        if self.n_accumulated > 1 {
            let inv = 1.0 / self.n_accumulated as f32;
            for g in self.grads.iter_mut() {
                *g *= inv;
            }
        }
    }
}

/// Dense f32 surrogate forward for a latent linear layer.
///
/// This is the differentiable forward used for gradient checks. The deployed
/// inference path still uses the ternary matrix derived from these latents.
pub fn linear_forward_latent(latent: &LatentWeights, input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), latent.cols);
    assert_eq!(output.len(), latent.rows);

    for r in 0..latent.rows {
        let row = &latent.weights[r * latent.cols..(r + 1) * latent.cols];
        output[r] = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
    }
}

/// Input gradient for the f32 latent linear surrogate.
///
/// Given `y = W x` and upstream `dL/dy`, computes `dL/dx = W^T dL/dy`.
pub fn linear_input_grad_latent(latent: &LatentWeights, upstream: &[f32], input_grad: &mut [f32]) {
    assert_eq!(upstream.len(), latent.rows);
    assert_eq!(input_grad.len(), latent.cols);

    input_grad.fill(0.0);
    linear_input_grad_latent_add(latent, upstream, input_grad);
}

/// Add input gradient for the f32 latent linear surrogate into `input_grad`.
pub fn linear_input_grad_latent_add(
    latent: &LatentWeights,
    upstream: &[f32],
    input_grad: &mut [f32],
) {
    assert_eq!(upstream.len(), latent.rows);
    assert_eq!(input_grad.len(), latent.cols);

    for r in 0..latent.rows {
        let scale = upstream[r];
        if scale == 0.0 {
            continue;
        }
        let row = &latent.weights[r * latent.cols..(r + 1) * latent.cols];
        for c in 0..latent.cols {
            input_grad[c] += scale * row[c];
        }
    }
}

fn silu(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    x * sig
}

fn silu_grad(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig + x * sig * (1.0 - sig)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Dense f32 surrogate forward for the ternary GLU channel mixer.
///
/// The cache slices are caller-owned so the later training path can reuse this
/// shape without heap allocation: `gate_pre = W_gate x`, `gate =
/// SiLU(gate_pre)`, `up = W_up x`, `fused = gate * up`, `output = W_down fused`.
pub fn glu_forward_latent(
    w_gate: &LatentWeights,
    w_up: &LatentWeights,
    w_down: &LatentWeights,
    input: &[f32],
    gate_pre: &mut [f32],
    gate: &mut [f32],
    up: &mut [f32],
    fused: &mut [f32],
    output: &mut [f32],
) {
    assert_eq!(w_gate.cols, input.len());
    assert_eq!(w_up.cols, input.len());
    assert_eq!(w_gate.rows, w_up.rows);
    assert_eq!(w_down.cols, w_gate.rows);
    assert_eq!(w_down.rows, output.len());
    assert_eq!(gate_pre.len(), w_gate.rows);
    assert_eq!(gate.len(), w_gate.rows);
    assert_eq!(up.len(), w_gate.rows);
    assert_eq!(fused.len(), w_gate.rows);

    linear_forward_latent(w_gate, input, gate_pre);
    for i in 0..gate.len() {
        gate[i] = silu(gate_pre[i]);
    }

    linear_forward_latent(w_up, input, up);
    for i in 0..fused.len() {
        fused[i] = gate[i] * up[i];
    }

    linear_forward_latent(w_down, fused, output);
}

/// Backward pass for the dense f32 GLU surrogate.
///
/// Accumulates into `grad_input` and the three weight gradient buffers. Scratch
/// slices are caller-owned and overwritten.
pub fn glu_backward_latent(
    w_gate: &LatentWeights,
    w_up: &LatentWeights,
    w_down: &LatentWeights,
    input: &[f32],
    gate_pre: &[f32],
    gate: &[f32],
    up: &[f32],
    fused: &[f32],
    upstream: &[f32],
    grad_input: &mut [f32],
    grad_w_gate: &mut GradBuffer,
    grad_w_up: &mut GradBuffer,
    grad_w_down: &mut GradBuffer,
    grad_fused: &mut [f32],
    grad_gate_pre: &mut [f32],
    grad_up: &mut [f32],
) {
    assert_eq!(w_gate.cols, input.len());
    assert_eq!(w_up.cols, input.len());
    assert_eq!(w_gate.rows, w_up.rows);
    assert_eq!(w_down.cols, w_gate.rows);
    assert_eq!(w_down.rows, upstream.len());
    assert_eq!(grad_input.len(), input.len());
    assert_eq!(gate_pre.len(), w_gate.rows);
    assert_eq!(gate.len(), w_gate.rows);
    assert_eq!(up.len(), w_gate.rows);
    assert_eq!(fused.len(), w_gate.rows);
    assert_eq!(grad_fused.len(), w_gate.rows);
    assert_eq!(grad_gate_pre.len(), w_gate.rows);
    assert_eq!(grad_up.len(), w_gate.rows);

    grad_w_down.accumulate_ste(w_down, fused, upstream);
    linear_input_grad_latent(w_down, upstream, grad_fused);

    for i in 0..w_gate.rows {
        let grad_gate = grad_fused[i] * up[i];
        grad_gate_pre[i] = grad_gate * silu_grad(gate_pre[i]);
        grad_up[i] = grad_fused[i] * gate[i];
    }

    grad_w_gate.accumulate_ste(w_gate, input, grad_gate_pre);
    grad_w_up.accumulate_ste(w_up, input, grad_up);
    linear_input_grad_latent_add(w_gate, grad_gate_pre, grad_input);
    linear_input_grad_latent_add(w_up, grad_up, grad_input);
}

/// Dense f32 surrogate forward for one MLGRU timestep.
///
/// Computes the current inference recurrence:
/// `f = sigmoid(W_f x + b_f)`, `c = SiLU(W_c x + b_c)`,
/// `o = sigmoid(W_o x + b_o)`, `next = f * prev + (1 - f) * c`,
/// `output = next * o`.
pub fn mlgru_forward_latent(
    w_f: &LatentWeights,
    b_f: &[f32],
    w_c: &LatentWeights,
    b_c: &[f32],
    w_o: &LatentWeights,
    b_o: &[f32],
    input: &[f32],
    prev_state: &[f32],
    f_pre: &mut [f32],
    f_gate: &mut [f32],
    c_pre: &mut [f32],
    c_value: &mut [f32],
    o_pre: &mut [f32],
    o_gate: &mut [f32],
    next_state: &mut [f32],
    output: &mut [f32],
) {
    let d = input.len();
    assert_eq!(w_f.rows, d);
    assert_eq!(w_f.cols, d);
    assert_eq!(w_c.rows, d);
    assert_eq!(w_c.cols, d);
    assert_eq!(w_o.rows, d);
    assert_eq!(w_o.cols, d);
    assert_eq!(b_f.len(), d);
    assert_eq!(b_c.len(), d);
    assert_eq!(b_o.len(), d);
    assert_eq!(prev_state.len(), d);
    assert_eq!(f_pre.len(), d);
    assert_eq!(f_gate.len(), d);
    assert_eq!(c_pre.len(), d);
    assert_eq!(c_value.len(), d);
    assert_eq!(o_pre.len(), d);
    assert_eq!(o_gate.len(), d);
    assert_eq!(next_state.len(), d);
    assert_eq!(output.len(), d);

    linear_forward_latent(w_f, input, f_pre);
    linear_forward_latent(w_c, input, c_pre);
    linear_forward_latent(w_o, input, o_pre);

    for i in 0..d {
        f_pre[i] += b_f[i];
        c_pre[i] += b_c[i];
        o_pre[i] += b_o[i];
        f_gate[i] = sigmoid(f_pre[i]);
        c_value[i] = silu(c_pre[i]);
        o_gate[i] = sigmoid(o_pre[i]);
        next_state[i] = f_gate[i] * prev_state[i] + (1.0 - f_gate[i]) * c_value[i];
        output[i] = next_state[i] * o_gate[i];
    }
}

/// Backward pass for one dense f32 MLGRU timestep.
///
/// Accumulates into input/state/weight/bias gradients. `upstream_next_state`
/// carries the through-time gradient from the next timestep; pass zeros for an
/// isolated one-step loss.
pub fn mlgru_backward_latent(
    w_f: &LatentWeights,
    w_c: &LatentWeights,
    w_o: &LatentWeights,
    input: &[f32],
    prev_state: &[f32],
    f_pre: &[f32],
    f_gate: &[f32],
    c_pre: &[f32],
    c_value: &[f32],
    o_gate: &[f32],
    next_state: &[f32],
    upstream_output: &[f32],
    upstream_next_state: &[f32],
    grad_input: &mut [f32],
    grad_prev_state: &mut [f32],
    grad_w_f: &mut GradBuffer,
    grad_b_f: &mut [f32],
    grad_w_c: &mut GradBuffer,
    grad_b_c: &mut [f32],
    grad_w_o: &mut GradBuffer,
    grad_b_o: &mut [f32],
    grad_state_total: &mut [f32],
    grad_f_pre: &mut [f32],
    grad_c_pre: &mut [f32],
    grad_o_pre: &mut [f32],
) {
    let d = input.len();
    assert_eq!(w_f.rows, d);
    assert_eq!(w_f.cols, d);
    assert_eq!(w_c.rows, d);
    assert_eq!(w_c.cols, d);
    assert_eq!(w_o.rows, d);
    assert_eq!(w_o.cols, d);
    assert_eq!(prev_state.len(), d);
    assert_eq!(f_pre.len(), d);
    assert_eq!(f_gate.len(), d);
    assert_eq!(c_pre.len(), d);
    assert_eq!(c_value.len(), d);
    assert_eq!(o_gate.len(), d);
    assert_eq!(next_state.len(), d);
    assert_eq!(upstream_output.len(), d);
    assert_eq!(upstream_next_state.len(), d);
    assert_eq!(grad_input.len(), d);
    assert_eq!(grad_prev_state.len(), d);
    assert_eq!(grad_b_f.len(), d);
    assert_eq!(grad_b_c.len(), d);
    assert_eq!(grad_b_o.len(), d);
    assert_eq!(grad_state_total.len(), d);
    assert_eq!(grad_f_pre.len(), d);
    assert_eq!(grad_c_pre.len(), d);
    assert_eq!(grad_o_pre.len(), d);

    for i in 0..d {
        grad_state_total[i] = upstream_next_state[i] + upstream_output[i] * o_gate[i];
        let grad_o = upstream_output[i] * next_state[i];
        grad_o_pre[i] = grad_o * o_gate[i] * (1.0 - o_gate[i]);

        let grad_f = grad_state_total[i] * (prev_state[i] - c_value[i]);
        grad_f_pre[i] = grad_f * f_gate[i] * (1.0 - f_gate[i]);

        let grad_c = grad_state_total[i] * (1.0 - f_gate[i]);
        grad_c_pre[i] = grad_c * silu_grad(c_pre[i]);

        grad_prev_state[i] += grad_state_total[i] * f_gate[i];
        grad_b_f[i] += grad_f_pre[i];
        grad_b_c[i] += grad_c_pre[i];
        grad_b_o[i] += grad_o_pre[i];
    }

    grad_w_f.accumulate_ste(w_f, input, grad_f_pre);
    grad_w_c.accumulate_ste(w_c, input, grad_c_pre);
    grad_w_o.accumulate_ste(w_o, input, grad_o_pre);
    linear_input_grad_latent_add(w_f, grad_f_pre, grad_input);
    linear_input_grad_latent_add(w_c, grad_c_pre, grad_input);
    linear_input_grad_latent_add(w_o, grad_o_pre, grad_input);
}

/// Input gradient for a deployed ternary matrix using its effective scaled
/// weights.
///
/// Given `y = (trit * row_scale) x` and upstream `dL/dy`, computes
/// `dL/dx = W_eff^T dL/dy`.
pub fn linear_input_grad_ternary(weight: &TernaryMatrix, upstream: &[f32], input_grad: &mut [f32]) {
    assert_eq!(upstream.len(), weight.rows());
    assert_eq!(input_grad.len(), weight.cols());

    input_grad.fill(0.0);
    let trits = weight.unpack();
    for r in 0..weight.rows() {
        let row_scale = weight.scales()[r] * upstream[r];
        if row_scale == 0.0 {
            continue;
        }
        for c in 0..weight.cols() {
            input_grad[c] += row_scale * trits[r * weight.cols() + c] as f32;
        }
    }
}

/// Dense unembedding forward over a row-major `[vocab_size x d_model]` table.
///
/// Computes `logits[v] = dot(hidden, table[v])`. This mirrors
/// `Embedding::unembed` but works on slices for gradient checks and future
/// training code.
pub fn unembed_forward(
    table: &[f32],
    vocab_size: usize,
    d_model: usize,
    hidden: &[f32],
    logits: &mut [f32],
) {
    assert_eq!(table.len(), vocab_size * d_model);
    assert_eq!(hidden.len(), d_model);
    assert_eq!(logits.len(), vocab_size);

    for v in 0..vocab_size {
        let row = &table[v * d_model..(v + 1) * d_model];
        logits[v] = row.iter().zip(hidden.iter()).map(|(w, h)| w * h).sum();
    }
}

/// Stable softmax cross-entropy and its gradient with respect to logits.
///
/// Returns `-log softmax(logits)[target]` and writes `softmax(logits) - onehot`
/// into `grad_logits`.
pub fn cross_entropy_logits_grad(logits: &[f32], target: usize, grad_logits: &mut [f32]) -> f32 {
    assert!(!logits.is_empty());
    assert!(
        target < logits.len(),
        "target {} >= vocab {}",
        target,
        logits.len()
    );
    assert_eq!(grad_logits.len(), logits.len());

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for (g, &logit) in grad_logits.iter_mut().zip(logits.iter()) {
        let exp = (logit - max_logit).exp();
        *g = exp;
        sum_exp += exp;
    }

    assert!(sum_exp.is_finite() && sum_exp > 0.0);
    for g in grad_logits.iter_mut() {
        *g /= sum_exp;
    }
    grad_logits[target] -= 1.0;

    max_logit + sum_exp.ln() - logits[target]
}

/// Backward pass for dense unembedding.
///
/// Given `logits = table * hidden` and upstream `dL/dlogits`, accumulates
/// `dL/dhidden` and `dL/dtable`. The gradient buffers are additive so callers
/// can aggregate across timesteps before optimizer steps.
pub fn unembed_backward(
    table: &[f32],
    vocab_size: usize,
    d_model: usize,
    hidden: &[f32],
    grad_logits: &[f32],
    grad_hidden: &mut [f32],
    grad_table: &mut [f32],
) {
    assert_eq!(table.len(), vocab_size * d_model);
    assert_eq!(hidden.len(), d_model);
    assert_eq!(grad_logits.len(), vocab_size);
    assert_eq!(grad_hidden.len(), d_model);
    assert_eq!(grad_table.len(), table.len());

    for v in 0..vocab_size {
        let scale = grad_logits[v];
        if scale == 0.0 {
            continue;
        }
        let row_start = v * d_model;
        for d in 0..d_model {
            grad_hidden[d] += scale * table[row_start + d];
            grad_table[row_start + d] += scale * hidden[d];
        }
    }
}

/// Backward scatter for token embedding lookup.
///
/// Given `hidden = table[token]`, accumulates `dL/dtable[token] += upstream`.
/// For a tied embedding/unembedding table this gradient must be added to the
/// same table buffer used by [`unembed_backward`].
pub fn embedding_backward(
    token: u32,
    vocab_size: usize,
    d_model: usize,
    upstream: &[f32],
    grad_table: &mut [f32],
) {
    let token = token as usize;
    assert!(
        token < vocab_size,
        "token {} >= vocab {}",
        token,
        vocab_size
    );
    assert_eq!(upstream.len(), d_model);
    assert_eq!(grad_table.len(), vocab_size * d_model);

    let row = &mut grad_table[token * d_model..(token + 1) * d_model];
    for (g, &u) in row.iter_mut().zip(upstream.iter()) {
        *g += u;
    }
}

/// Out-of-place RMSNorm forward.
///
/// Computes `output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]`.
pub fn rmsnorm_forward(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    assert_eq!(input.len(), weight.len());
    assert_eq!(output.len(), input.len());
    assert!(!input.is_empty());

    let n = input.len() as f32;
    let ss: f32 = input.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (ss / n + eps).sqrt();
    for i in 0..input.len() {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

/// Backward pass for RMSNorm.
///
/// Given `output = rmsnorm(input, weight, eps)` and upstream `dL/doutput`,
/// accumulates `dL/dinput` and `dL/dweight`.
pub fn rmsnorm_backward(
    input: &[f32],
    weight: &[f32],
    eps: f32,
    upstream: &[f32],
    grad_input: &mut [f32],
    grad_weight: &mut [f32],
) {
    assert_eq!(input.len(), weight.len());
    assert_eq!(upstream.len(), input.len());
    assert_eq!(grad_input.len(), input.len());
    assert_eq!(grad_weight.len(), weight.len());
    assert!(!input.is_empty());

    let n = input.len() as f32;
    let ss: f32 = input.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (ss / n + eps).sqrt();
    let inv_rms_cubed_over_n = inv_rms * inv_rms * inv_rms / n;

    let dot_ax: f32 = upstream
        .iter()
        .zip(weight.iter())
        .zip(input.iter())
        .map(|((&u, &w), &x)| u * w * x)
        .sum();

    for i in 0..input.len() {
        grad_weight[i] += upstream[i] * input[i] * inv_rms;
        grad_input[i] +=
            upstream[i] * weight[i] * inv_rms - input[i] * inv_rms_cubed_over_n * dot_ax;
    }
}

/// Ternarize f32 weights to {-1, 0, 1} with per-row absmean scaling.
fn ternarize(weights: &[f32], rows: usize, cols: usize) -> (TernaryMatrix, Vec<f32>) {
    let mut trits = vec![0i8; rows * cols];
    let mut scales = vec![0.0f32; rows];

    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        let abs_mean = row.iter().map(|v| v.abs()).sum::<f32>() / cols as f32;

        if abs_mean > 1e-10 {
            let inv = 1.0 / abs_mean;
            for c in 0..cols {
                let norm = row[c] * inv;
                trits[r * cols + c] = if norm > 0.5 {
                    1
                } else if norm < -0.5 {
                    -1
                } else {
                    0
                };
            }
            scales[r] = abs_mean;
        }
    }

    (TernaryMatrix::pack(&trits, &scales, rows, cols), scales)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::check::{assert_gradient_close, GradCheckConfig};

    fn latent_from_weights(rows: usize, cols: usize, weights: Vec<f32>) -> LatentWeights {
        assert_eq!(weights.len(), rows * cols);
        let trits = vec![0i8; rows * cols];
        let scales = vec![1.0f32; rows];
        LatentWeights {
            weights,
            ternary: TernaryMatrix::pack(&trits, &scales, rows, cols),
            scales,
            rows,
            cols,
            version: 0,
        }
    }

    fn glu_latents_from_params(
        params: &[f32],
        d_model: usize,
        d_inner: usize,
    ) -> (LatentWeights, LatentWeights, LatentWeights) {
        let inner_len = d_inner * d_model;
        let down_len = d_model * d_inner;
        assert_eq!(params.len(), inner_len * 2 + down_len);

        let w_gate = latent_from_weights(d_inner, d_model, params[0..inner_len].to_vec());
        let w_up = latent_from_weights(d_inner, d_model, params[inner_len..inner_len * 2].to_vec());
        let w_down = latent_from_weights(d_model, d_inner, params[inner_len * 2..].to_vec());
        (w_gate, w_up, w_down)
    }

    fn glu_forward_output(
        w_gate: &LatentWeights,
        w_up: &LatentWeights,
        w_down: &LatentWeights,
        input: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut gate_pre = vec![0.0f32; w_gate.rows];
        let mut gate = vec![0.0f32; w_gate.rows];
        let mut up = vec![0.0f32; w_gate.rows];
        let mut fused = vec![0.0f32; w_gate.rows];
        let mut output = vec![0.0f32; w_down.rows];

        glu_forward_latent(
            w_gate,
            w_up,
            w_down,
            input,
            &mut gate_pre,
            &mut gate,
            &mut up,
            &mut fused,
            &mut output,
        );

        (gate_pre, gate, up, fused, output)
    }

    fn mlgru_params(d: usize) -> Vec<f32> {
        let n = d * d * 3 + d * 3;
        (0..n).map(|i| ((i as i32 % 9) - 4) as f32 * 0.08).collect()
    }

    fn mlgru_latents_from_params(
        params: &[f32],
        d: usize,
    ) -> (
        LatentWeights,
        Vec<f32>,
        LatentWeights,
        Vec<f32>,
        LatentWeights,
        Vec<f32>,
    ) {
        let matrix_len = d * d;
        let bias_len = d;
        assert_eq!(params.len(), matrix_len * 3 + bias_len * 3);

        let w_f = latent_from_weights(d, d, params[0..matrix_len].to_vec());
        let w_c = latent_from_weights(d, d, params[matrix_len..matrix_len * 2].to_vec());
        let w_o = latent_from_weights(d, d, params[matrix_len * 2..matrix_len * 3].to_vec());
        let bias_start = matrix_len * 3;
        let b_f = params[bias_start..bias_start + bias_len].to_vec();
        let b_c = params[bias_start + bias_len..bias_start + bias_len * 2].to_vec();
        let b_o = params[bias_start + bias_len * 2..].to_vec();

        (w_f, b_f, w_c, b_c, w_o, b_o)
    }

    fn mlgru_forward_output(
        w_f: &LatentWeights,
        b_f: &[f32],
        w_c: &LatentWeights,
        b_c: &[f32],
        w_o: &LatentWeights,
        b_o: &[f32],
        input: &[f32],
        prev_state: &[f32],
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let d = input.len();
        let mut f_pre = vec![0.0f32; d];
        let mut f_gate = vec![0.0f32; d];
        let mut c_pre = vec![0.0f32; d];
        let mut c_value = vec![0.0f32; d];
        let mut o_pre = vec![0.0f32; d];
        let mut o_gate = vec![0.0f32; d];
        let mut next_state = vec![0.0f32; d];
        let mut output = vec![0.0f32; d];

        mlgru_forward_latent(
            w_f,
            b_f,
            w_c,
            b_c,
            w_o,
            b_o,
            input,
            prev_state,
            &mut f_pre,
            &mut f_gate,
            &mut c_pre,
            &mut c_value,
            &mut o_pre,
            &mut o_gate,
            &mut next_state,
            &mut output,
        );

        (
            f_pre, f_gate, c_pre, c_value, o_pre, o_gate, next_state, output,
        )
    }

    #[test]
    fn latent_linear_input_grad_matches_finite_difference() {
        let latent = latent_from_weights(
            3,
            4,
            vec![
                0.20, -0.10, 0.05, 0.30, -0.40, 0.25, 0.15, -0.05, 0.12, 0.08, -0.22, 0.18,
            ],
        );
        let target = vec![0.15f32, -0.20, 0.05];
        let input = vec![0.4f32, -0.7, 0.2, 0.9];

        assert_gradient_close(
            &input,
            |x| {
                let mut output = vec![0.0f32; latent.rows];
                linear_forward_latent(&latent, x, &mut output);
                output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum()
            },
            |x, out| {
                let mut output = vec![0.0f32; latent.rows];
                linear_forward_latent(&latent, x, &mut output);
                let upstream: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut input_grad = vec![0.0f32; latent.cols];
                linear_input_grad_latent(&latent, &upstream, &mut input_grad);
                for (o, &g) in out.iter_mut().zip(input_grad.iter()) {
                    *o = g as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 2e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn ternary_input_grad_uses_scaled_effective_weights() {
        let trits = vec![1, 0, -1, -1, 1, 0];
        let scales = vec![0.5f32, 2.0];
        let weight = TernaryMatrix::pack(&trits, &scales, 2, 3);
        let upstream = vec![3.0f32, -0.25];
        let mut input_grad = vec![0.0f32; 3];

        linear_input_grad_ternary(&weight, &upstream, &mut input_grad);

        let expected = [2.0f32, -0.5, -1.5];
        for (got, want) in input_grad.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {}, want {}", got, want);
        }
    }

    #[test]
    fn ste_weight_grad_matches_outer_product_when_mask_open() {
        let latent = latent_from_weights(2, 3, vec![0.10, -0.20, 0.30, -0.15, 0.25, -0.05]);
        let input = vec![0.5f32, -1.0, 0.25];
        let upstream = vec![2.0f32, -0.5];
        let mut grads = GradBuffer::new(2, 3);

        grads.accumulate_ste(&latent, &input, &upstream);

        let expected = vec![1.0, -2.0, 0.5, -0.25, 0.5, -0.125];
        assert_eq!(grads.n_accumulated, 1);
        for (got, want) in grads.grads.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got {}, want {}", got, want);
        }
    }

    #[test]
    fn cross_entropy_logits_grad_matches_finite_difference() {
        let logits = vec![0.25f32, -0.35, 0.10, 0.80];
        let target = 2usize;

        assert_gradient_close(
            &logits,
            |x| {
                let mut grad_logits = vec![0.0f32; x.len()];
                cross_entropy_logits_grad(x, target, &mut grad_logits) as f64
            },
            |x, out| {
                let mut grad_logits = vec![0.0f32; x.len()];
                cross_entropy_logits_grad(x, target, &mut grad_logits);
                for (o, &g) in out.iter_mut().zip(grad_logits.iter()) {
                    *o = g as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 2e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn unembed_hidden_grad_matches_finite_difference() {
        let vocab_size = 4usize;
        let d_model = 3usize;
        let table = vec![
            0.20, -0.10, 0.05, -0.30, 0.40, 0.15, 0.08, 0.12, -0.22, 0.35, -0.18, 0.27,
        ];
        let hidden = vec![0.6f32, -0.2, 0.9];
        let target = 1usize;

        assert_gradient_close(
            &hidden,
            |h| {
                let mut logits = vec![0.0f32; vocab_size];
                let mut grad_logits = vec![0.0f32; vocab_size];
                unembed_forward(&table, vocab_size, d_model, h, &mut logits);
                cross_entropy_logits_grad(&logits, target, &mut grad_logits) as f64
            },
            |h, out| {
                let mut logits = vec![0.0f32; vocab_size];
                let mut grad_logits = vec![0.0f32; vocab_size];
                let mut grad_hidden = vec![0.0f32; d_model];
                let mut grad_table = vec![0.0f32; table.len()];
                unembed_forward(&table, vocab_size, d_model, h, &mut logits);
                cross_entropy_logits_grad(&logits, target, &mut grad_logits);
                unembed_backward(
                    &table,
                    vocab_size,
                    d_model,
                    h,
                    &grad_logits,
                    &mut grad_hidden,
                    &mut grad_table,
                );
                for (o, &g) in out.iter_mut().zip(grad_hidden.iter()) {
                    *o = g as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 2e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn unembed_table_grad_matches_finite_difference() {
        let vocab_size = 4usize;
        let d_model = 3usize;
        let table = vec![
            0.20, -0.10, 0.05, -0.30, 0.40, 0.15, 0.08, 0.12, -0.22, 0.35, -0.18, 0.27,
        ];
        let hidden = vec![0.6f32, -0.2, 0.9];
        let target = 3usize;

        assert_gradient_close(
            &table,
            |w| {
                let mut logits = vec![0.0f32; vocab_size];
                let mut grad_logits = vec![0.0f32; vocab_size];
                unembed_forward(w, vocab_size, d_model, &hidden, &mut logits);
                cross_entropy_logits_grad(&logits, target, &mut grad_logits) as f64
            },
            |w, out| {
                let mut logits = vec![0.0f32; vocab_size];
                let mut grad_logits = vec![0.0f32; vocab_size];
                let mut grad_hidden = vec![0.0f32; d_model];
                let mut grad_table = vec![0.0f32; w.len()];
                unembed_forward(w, vocab_size, d_model, &hidden, &mut logits);
                cross_entropy_logits_grad(&logits, target, &mut grad_logits);
                unembed_backward(
                    w,
                    vocab_size,
                    d_model,
                    &hidden,
                    &grad_logits,
                    &mut grad_hidden,
                    &mut grad_table,
                );
                for (o, &g) in out.iter_mut().zip(grad_table.iter()) {
                    *o = g as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 2e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn embedding_backward_accumulates_selected_row() {
        let vocab_size = 4usize;
        let d_model = 3usize;
        let mut grad_table = vec![0.0f32; vocab_size * d_model];
        let upstream_a = vec![0.25f32, -0.5, 1.0];
        let upstream_b = vec![-0.75f32, 0.25, 0.5];

        embedding_backward(2, vocab_size, d_model, &upstream_a, &mut grad_table);
        embedding_backward(2, vocab_size, d_model, &upstream_b, &mut grad_table);

        let expected = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -0.25, 1.5, 0.0, 0.0, 0.0,
        ];
        assert_eq!(grad_table, expected);
    }

    #[test]
    fn rmsnorm_input_grad_matches_finite_difference() {
        let input = vec![0.7f32, -0.4, 0.2, 1.1];
        let weight = vec![1.2f32, 0.8, -0.5, 1.5];
        let target = vec![0.3f32, -0.1, 0.2, 0.7];
        let eps = 1e-6f32;

        assert_gradient_close(
            &input,
            |x| {
                let mut output = vec![0.0f32; x.len()];
                rmsnorm_forward(x, &weight, eps, &mut output);
                output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum()
            },
            |x, out| {
                let mut output = vec![0.0f32; x.len()];
                rmsnorm_forward(x, &weight, eps, &mut output);
                let upstream: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut grad_input = vec![0.0f32; x.len()];
                let mut grad_weight = vec![0.0f32; weight.len()];
                rmsnorm_backward(
                    x,
                    &weight,
                    eps,
                    &upstream,
                    &mut grad_input,
                    &mut grad_weight,
                );
                for (o, &g) in out.iter_mut().zip(grad_input.iter()) {
                    *o = g as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 3e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn rmsnorm_weight_grad_matches_finite_difference() {
        let input = vec![0.7f32, -0.4, 0.2, 1.1];
        let weight = vec![1.2f32, 0.8, -0.5, 1.5];
        let target = vec![0.3f32, -0.1, 0.2, 0.7];
        let eps = 1e-6f32;

        assert_gradient_close(
            &weight,
            |w| {
                let mut output = vec![0.0f32; input.len()];
                rmsnorm_forward(&input, w, eps, &mut output);
                output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum()
            },
            |w, out| {
                let mut output = vec![0.0f32; input.len()];
                rmsnorm_forward(&input, w, eps, &mut output);
                let upstream: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut grad_input = vec![0.0f32; input.len()];
                let mut grad_weight = vec![0.0f32; w.len()];
                rmsnorm_backward(&input, w, eps, &upstream, &mut grad_input, &mut grad_weight);
                for (o, &g) in out.iter_mut().zip(grad_weight.iter()) {
                    *o = g as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 3e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn glu_input_grad_matches_finite_difference() {
        let d_model = 3usize;
        let d_inner = 4usize;
        let n_params = d_inner * d_model * 2 + d_model * d_inner;
        let params: Vec<f32> = (0..n_params)
            .map(|i| ((i as i32 % 9) - 4) as f32 * 0.12)
            .collect();
        let (w_gate, w_up, w_down) = glu_latents_from_params(&params, d_model, d_inner);
        let input = vec![0.6f32, -0.3, 0.8];
        let target = vec![10.0f32, -7.5, 5.0];

        assert_gradient_close(
            &input,
            |x| {
                let (_, _, _, _, output) = glu_forward_output(&w_gate, &w_up, &w_down, x);
                output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum()
            },
            |x, out| {
                let (gate_pre, gate, up, fused, output) =
                    glu_forward_output(&w_gate, &w_up, &w_down, x);
                let upstream: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut grad_input = vec![0.0f32; d_model];
                let mut grad_w_gate = GradBuffer::new(d_inner, d_model);
                let mut grad_w_up = GradBuffer::new(d_inner, d_model);
                let mut grad_w_down = GradBuffer::new(d_model, d_inner);
                let mut grad_fused = vec![0.0f32; d_inner];
                let mut grad_gate_pre = vec![0.0f32; d_inner];
                let mut grad_up = vec![0.0f32; d_inner];
                glu_backward_latent(
                    &w_gate,
                    &w_up,
                    &w_down,
                    x,
                    &gate_pre,
                    &gate,
                    &up,
                    &fused,
                    &upstream,
                    &mut grad_input,
                    &mut grad_w_gate,
                    &mut grad_w_up,
                    &mut grad_w_down,
                    &mut grad_fused,
                    &mut grad_gate_pre,
                    &mut grad_up,
                );
                for (o, &g) in out.iter_mut().zip(grad_input.iter()) {
                    *o = g as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 4e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn glu_weight_grads_match_finite_difference() {
        let d_model = 3usize;
        let d_inner = 4usize;
        let n_params = d_inner * d_model * 2 + d_model * d_inner;
        let params: Vec<f32> = (0..n_params)
            .map(|i| ((i as i32 % 9) - 4) as f32 * 0.12)
            .collect();
        let input = vec![0.6f32, -0.3, 0.8];
        let target = vec![10.0f32, -7.5, 5.0];

        assert_gradient_close(
            &params,
            |p| {
                let (w_gate, w_up, w_down) = glu_latents_from_params(p, d_model, d_inner);
                let (_, _, _, _, output) = glu_forward_output(&w_gate, &w_up, &w_down, &input);
                output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum()
            },
            |p, out| {
                let (w_gate, w_up, w_down) = glu_latents_from_params(p, d_model, d_inner);
                let (gate_pre, gate, up, fused, output) =
                    glu_forward_output(&w_gate, &w_up, &w_down, &input);
                let upstream: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut grad_input = vec![0.0f32; d_model];
                let mut grad_w_gate = GradBuffer::new(d_inner, d_model);
                let mut grad_w_up = GradBuffer::new(d_inner, d_model);
                let mut grad_w_down = GradBuffer::new(d_model, d_inner);
                let mut grad_fused = vec![0.0f32; d_inner];
                let mut grad_gate_pre = vec![0.0f32; d_inner];
                let mut grad_up = vec![0.0f32; d_inner];
                glu_backward_latent(
                    &w_gate,
                    &w_up,
                    &w_down,
                    &input,
                    &gate_pre,
                    &gate,
                    &up,
                    &fused,
                    &upstream,
                    &mut grad_input,
                    &mut grad_w_gate,
                    &mut grad_w_up,
                    &mut grad_w_down,
                    &mut grad_fused,
                    &mut grad_gate_pre,
                    &mut grad_up,
                );

                let mut cursor = 0usize;
                for &g in grad_w_gate.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grad_w_up.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grad_w_down.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                assert_eq!(cursor, out.len());
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 4e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn mlgru_input_and_state_grads_match_finite_difference() {
        let d = 3usize;
        let params = mlgru_params(d);
        let (w_f, b_f, w_c, b_c, w_o, b_o) = mlgru_latents_from_params(&params, d);
        let input = vec![0.7f32, -0.4, 0.9];
        let prev_state = vec![0.2f32, -0.1, 0.3];
        let target = vec![2.0f32, -1.5, 1.0];
        let future_state_grad = vec![0.3f32, -0.2, 0.1];
        let mut input_and_state = input.clone();
        input_and_state.extend_from_slice(&prev_state);

        assert_gradient_close(
            &input_and_state,
            |p| {
                let x = &p[0..d];
                let prev = &p[d..d * 2];
                let (_, _, _, _, _, _, next_state, output) =
                    mlgru_forward_output(&w_f, &b_f, &w_c, &b_c, &w_o, &b_o, x, prev);
                let output_loss: f64 = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum();
                let future_loss: f64 = next_state
                    .iter()
                    .zip(future_state_grad.iter())
                    .map(|(&s, &g)| s as f64 * g as f64)
                    .sum();
                output_loss + future_loss
            },
            |p, out| {
                let x = &p[0..d];
                let prev = &p[d..d * 2];
                let (f_pre, f_gate, c_pre, c_value, _o_pre, o_gate, next_state, output) =
                    mlgru_forward_output(&w_f, &b_f, &w_c, &b_c, &w_o, &b_o, x, prev);
                let upstream_output: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut grad_input = vec![0.0f32; d];
                let mut grad_prev_state = vec![0.0f32; d];
                let mut grad_w_f = GradBuffer::new(d, d);
                let mut grad_w_c = GradBuffer::new(d, d);
                let mut grad_w_o = GradBuffer::new(d, d);
                let mut grad_b_f = vec![0.0f32; d];
                let mut grad_b_c = vec![0.0f32; d];
                let mut grad_b_o = vec![0.0f32; d];
                let mut grad_state_total = vec![0.0f32; d];
                let mut grad_f_pre = vec![0.0f32; d];
                let mut grad_c_pre = vec![0.0f32; d];
                let mut grad_o_pre = vec![0.0f32; d];
                mlgru_backward_latent(
                    &w_f,
                    &w_c,
                    &w_o,
                    x,
                    prev,
                    &f_pre,
                    &f_gate,
                    &c_pre,
                    &c_value,
                    &o_gate,
                    &next_state,
                    &upstream_output,
                    &future_state_grad,
                    &mut grad_input,
                    &mut grad_prev_state,
                    &mut grad_w_f,
                    &mut grad_b_f,
                    &mut grad_w_c,
                    &mut grad_b_c,
                    &mut grad_w_o,
                    &mut grad_b_o,
                    &mut grad_state_total,
                    &mut grad_f_pre,
                    &mut grad_c_pre,
                    &mut grad_o_pre,
                );

                for i in 0..d {
                    out[i] = grad_input[i] as f64;
                    out[d + i] = grad_prev_state[i] as f64;
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 5e-3,
                denom_floor: 1e-8,
            },
        );
    }

    #[test]
    fn mlgru_weight_and_bias_grads_match_finite_difference() {
        let d = 3usize;
        let params = mlgru_params(d);
        let input = vec![0.7f32, -0.4, 0.9];
        let prev_state = vec![0.2f32, -0.1, 0.3];
        let target = vec![2.0f32, -1.5, 1.0];
        let future_state_grad = vec![0.3f32, -0.2, 0.1];

        assert_gradient_close(
            &params,
            |p| {
                let (w_f, b_f, w_c, b_c, w_o, b_o) = mlgru_latents_from_params(p, d);
                let (_, _, _, _, _, _, next_state, output) =
                    mlgru_forward_output(&w_f, &b_f, &w_c, &b_c, &w_o, &b_o, &input, &prev_state);
                let output_loss: f64 = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| {
                        let e = y as f64 - t as f64;
                        0.5 * e * e
                    })
                    .sum();
                let future_loss: f64 = next_state
                    .iter()
                    .zip(future_state_grad.iter())
                    .map(|(&s, &g)| s as f64 * g as f64)
                    .sum();
                output_loss + future_loss
            },
            |p, out| {
                let (w_f, b_f, w_c, b_c, w_o, b_o) = mlgru_latents_from_params(p, d);
                let (f_pre, f_gate, c_pre, c_value, _o_pre, o_gate, next_state, output) =
                    mlgru_forward_output(&w_f, &b_f, &w_c, &b_c, &w_o, &b_o, &input, &prev_state);
                let upstream_output: Vec<f32> = output
                    .iter()
                    .zip(target.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();
                let mut grad_input = vec![0.0f32; d];
                let mut grad_prev_state = vec![0.0f32; d];
                let mut grad_w_f = GradBuffer::new(d, d);
                let mut grad_w_c = GradBuffer::new(d, d);
                let mut grad_w_o = GradBuffer::new(d, d);
                let mut grad_b_f = vec![0.0f32; d];
                let mut grad_b_c = vec![0.0f32; d];
                let mut grad_b_o = vec![0.0f32; d];
                let mut grad_state_total = vec![0.0f32; d];
                let mut grad_f_pre = vec![0.0f32; d];
                let mut grad_c_pre = vec![0.0f32; d];
                let mut grad_o_pre = vec![0.0f32; d];
                mlgru_backward_latent(
                    &w_f,
                    &w_c,
                    &w_o,
                    &input,
                    &prev_state,
                    &f_pre,
                    &f_gate,
                    &c_pre,
                    &c_value,
                    &o_gate,
                    &next_state,
                    &upstream_output,
                    &future_state_grad,
                    &mut grad_input,
                    &mut grad_prev_state,
                    &mut grad_w_f,
                    &mut grad_b_f,
                    &mut grad_w_c,
                    &mut grad_b_c,
                    &mut grad_w_o,
                    &mut grad_b_o,
                    &mut grad_state_total,
                    &mut grad_f_pre,
                    &mut grad_c_pre,
                    &mut grad_o_pre,
                );

                let mut cursor = 0usize;
                for &g in grad_w_f.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grad_w_c.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grad_w_o.grads.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grad_b_f.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grad_b_c.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                for &g in grad_b_o.iter() {
                    out[cursor] = g as f64;
                    cursor += 1;
                }
                assert_eq!(cursor, out.len());
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 5e-3,
                denom_floor: 1e-8,
            },
        );
    }
}
