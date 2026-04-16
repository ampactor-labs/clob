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

        Self { weights, ternary: mat.clone(), scales, rows, cols, version: 0 }
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
        Self { weights, ternary, scales, rows: d, cols: d, version: 0 }
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
        Self { grads: vec![0.0; rows * cols], rows, cols, n_accumulated: 0 }
    }

    pub fn zero(&mut self) {
        self.grads.fill(0.0);
        self.n_accumulated = 0;
    }

    /// Accumulate gradient from a single sample.
    /// STE: gradient of the ternary quantization is 1 where |w| < threshold.
    pub fn accumulate_ste(
        &mut self,
        latent: &LatentWeights,
        input: &[f32],
        output_grad: &[f32],
    ) {
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
            for g in self.grads.iter_mut() { *g *= inv; }
        }
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
                trits[r * cols + c] = if norm > 0.5 { 1 }
                    else if norm < -0.5 { -1 }
                    else { 0 };
            }
            scales[r] = abs_mean;
        }
    }

    (TernaryMatrix::pack(&trits, &scales, rows, cols), scales)
}
