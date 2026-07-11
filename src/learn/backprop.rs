//! Full-model backward for a dense ternary core — Path B, Phase 3.
//!
//! Composes the checked leaf backwards (`grad.rs`) and the checked one-step
//! SSM backward (`ssm.rs`) into a truncated-BPTT backward over a window of
//! `T` tokens through an `L`-layer dense stack, gated end to end by a single
//! finite-difference gradient check.
//!
//! The model, per token, is
//!
//! ```text
//! x0 = embed[token]
//! per block l:
//!     h_mid = x + SSM(RMSNorm1(x))            (SSM carries state across tokens)
//!     x_out = h_mid + GLU(MLGRU(RMSNorm2(h_mid)))   (MLGRU carries state too)
//! logits = Readout(RMSNorm_final(x_L))         (readout UNTIED from embed)
//! loss  += cross_entropy(logits, target)
//! ```
//!
//! Two things make the backward non-trivial and worth checking as a whole:
//! the residual stream splits the gradient at every sublayer boundary, and
//! *two* recurrences (SSM state and MLGRU state) couple the timesteps, so the
//! outer loop runs in reverse over tokens carrying a future-state gradient
//! per layer for each recurrence.
//!
//! Untied readout: `embed` and `readout` are separate `[vocab × d_model]`
//! tables with independent gradients. The deployed inference model still ties
//! them; untying here is what lets even a readout-only training path move,
//! and is the config the manifesto's Path B calls for.
//!
//! This differentiates the f32 *latent* surrogate, exactly as the leaf checks
//! do. The straight-through mask in `GradBuffer::accumulate_ste` matches the
//! pure-linear surrogate gradient only while every latent weight sits in the
//! mask-open zone (|w| < 1.5 at unit scale); the end-to-end check keeps the
//! tiny model there, the same regime the leaf checks use. Verifying the STE
//! bridge into deployed ternary weights is separate work, per the plan.

use crate::learn::grad::{
    cross_entropy_logits_grad, embedding_backward, glu_backward_latent, glu_forward_latent,
    mlgru_backward_latent, mlgru_forward_latent, rmsnorm_backward, rmsnorm_forward,
    unembed_backward, unembed_forward, GradBuffer, LatentWeights,
};
use crate::learn::ssm::{ssm_backward_latent, ssm_forward_latent, SsmDims, SsmStepCache, SsmStepGrads};
use crate::tensor::ternary::TernaryMatrix;
use crate::util::seed::SeedTree;
use rand::Rng;
use serde::{Deserialize, Serialize};

const RMS_EPS: f32 = 1e-6;

/// Shape of a dense trainable core.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DenseDims {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_state: usize,
    pub d_inner: usize,
    pub n_layers: usize,
    pub vocab: usize,
}

impl DenseDims {
    /// A small dense preset for real training runs (d=64, L=3).
    pub fn small(vocab: usize) -> Self {
        Self { d_model: 64, n_heads: 4, d_head: 16, d_state: 16, d_inner: 128, n_layers: 3, vocab }
    }

    /// A minimal dense preset for fast smoke runs (d=16, L=2).
    pub fn mini(vocab: usize) -> Self {
        Self { d_model: 16, n_heads: 2, d_head: 8, d_state: 4, d_inner: 32, n_layers: 2, vocab }
    }

    pub fn ssm(self) -> SsmDims {
        SsmDims { n_heads: self.n_heads, d_state: self.d_state, d_head: self.d_head }
    }
    pub fn state_len(self) -> usize {
        self.n_heads * self.d_state
    }
    pub fn x_proj_rows(self) -> usize {
        self.n_heads + 2 * self.n_heads * self.d_state
    }
    /// Flat parameter count for one layer, in the canonical pack order.
    pub fn layer_len(self) -> usize {
        let d = self.d_model;
        let sq = d * d;
        d                                   // norm1
        + sq                                // in_proj
        + self.x_proj_rows() * d            // x_proj
        + sq                                // out_proj
        + self.state_len()                  // a_log
        + self.n_heads                      // d_param
        + self.n_heads                      // dt_bias
        + d                                 // norm2
        + sq + d                            // w_f, b_f
        + sq + d                            // w_c, b_c
        + sq + d                            // w_o, b_o
        + self.d_inner * d                  // w_gate
        + self.d_inner * d                  // w_up
        + d * self.d_inner                  // w_down
    }
    /// Total flat parameter count for the whole model.
    pub fn total_len(self) -> usize {
        self.vocab * self.d_model              // embed
        + self.n_layers * self.layer_len()
        + self.d_model                         // final norm
        + self.vocab * self.d_model            // readout
    }
}

/// Build a mask-open latent from a raw f32 slice (scales = 1, trits = 0).
fn latent(rows: usize, cols: usize, weights: &[f32]) -> LatentWeights {
    assert_eq!(weights.len(), rows * cols);
    let trits = vec![0i8; rows * cols];
    let scales = vec![1.0f32; rows];
    LatentWeights {
        weights: weights.to_vec(),
        ternary: TernaryMatrix::pack(&trits, &scales, rows, cols),
        scales,
        rows,
        cols,
        version: 0,
    }
}

/// Seeded initial parameters for a dense model, in canonical pack order and
/// in the STE-mask-open zone. Matrices get fan-in-scaled uniform noise, norm
/// scales start at 1, biases/dt_bias at 0, `a_log` at −0.5 (a ≈ 0.61), and
/// `d_param` at 0.1. Same seed → same init.
pub fn init_dense_params(dims: DenseDims, seed: u64) -> Vec<f32> {
    let mut rng = SeedTree::new(seed).child("train-init");
    let d = dims.d_model;
    let mut out = Vec::with_capacity(dims.total_len());

    let push_uniform = |out: &mut Vec<f32>, n: usize, fan_in: usize, rng: &mut rand::rngs::StdRng| {
        let s = 1.0 / (fan_in as f32).sqrt();
        for _ in 0..n {
            out.push(rng.gen_range(-s..s));
        }
    };

    push_uniform(&mut out, dims.vocab * d, d, &mut rng); // embed
    for _ in 0..dims.n_layers {
        out.extend(std::iter::repeat(1.0).take(d)); // norm1
        push_uniform(&mut out, d * d, d, &mut rng); // in_proj
        push_uniform(&mut out, dims.x_proj_rows() * d, d, &mut rng); // x_proj
        push_uniform(&mut out, d * d, d, &mut rng); // out_proj
        out.extend(std::iter::repeat(-0.5).take(dims.state_len())); // a_log
        out.extend(std::iter::repeat(0.1).take(dims.n_heads)); // d_param
        out.extend(std::iter::repeat(0.0).take(dims.n_heads)); // dt_bias
        out.extend(std::iter::repeat(1.0).take(d)); // norm2
        push_uniform(&mut out, d * d, d, &mut rng); // w_f
        out.extend(std::iter::repeat(0.0).take(d)); // b_f
        push_uniform(&mut out, d * d, d, &mut rng); // w_c
        out.extend(std::iter::repeat(0.0).take(d)); // b_c
        push_uniform(&mut out, d * d, d, &mut rng); // w_o
        out.extend(std::iter::repeat(0.0).take(d)); // b_o
        push_uniform(&mut out, dims.d_inner * d, d, &mut rng); // w_gate
        push_uniform(&mut out, dims.d_inner * d, d, &mut rng); // w_up
        push_uniform(&mut out, d * dims.d_inner, dims.d_inner, &mut rng); // w_down
    }
    out.extend(std::iter::repeat(1.0).take(d)); // final norm
    push_uniform(&mut out, dims.vocab * d, d, &mut rng); // readout

    debug_assert_eq!(out.len(), dims.total_len());
    out
}

/// Project a latent parameter vector to its deployed *effective* ternary form
/// — the ternarizable matrices (SSM/MLGRU/GLU projections) mapped to
/// `trit · row_absmean`, everything else (embed, readout, norms, biases,
/// `a_log`/`d_param`/`dt_bias`) passed through. Running `forward_window` on the
/// result gives the ternary model's behavior, since the deployed matmul is
/// exactly `trit · scale`. Used to measure the latent→ternary loss gap on the
/// re-ternarization cadence.
pub fn ternarize_matrix_families(dims: DenseDims, latent: &[f32]) -> Vec<f32> {
    assert_eq!(latent.len(), dims.total_len());
    let mut out = latent.to_vec();
    let d = dims.d_model;
    let mut c = 0usize;

    let ternarize = |out: &mut [f32], c: &mut usize, rows: usize, cols: usize| {
        for r in 0..rows {
            let row = &mut out[*c + r * cols..*c + (r + 1) * cols];
            let absmean = row.iter().map(|v| v.abs()).sum::<f32>() / cols as f32;
            if absmean > 1e-10 {
                for w in row.iter_mut() {
                    let n = *w / absmean;
                    let t = if n > 0.5 { 1.0 } else if n < -0.5 { -1.0 } else { 0.0 };
                    *w = t * absmean;
                }
            } else {
                row.fill(0.0);
            }
        }
        *c += rows * cols;
    };
    let skip = |c: &mut usize, n: usize| *c += n;

    skip(&mut c, dims.vocab * d); // embed
    for _ in 0..dims.n_layers {
        skip(&mut c, d); // norm1
        ternarize(&mut out, &mut c, d, d); // in_proj
        ternarize(&mut out, &mut c, dims.x_proj_rows(), d); // x_proj
        ternarize(&mut out, &mut c, d, d); // out_proj
        skip(&mut c, dims.state_len() + dims.n_heads + dims.n_heads + d); // a_log,d_param,dt_bias,norm2
        ternarize(&mut out, &mut c, d, d); // w_f
        skip(&mut c, d); // b_f
        ternarize(&mut out, &mut c, d, d); // w_c
        skip(&mut c, d); // b_c
        ternarize(&mut out, &mut c, d, d); // w_o
        skip(&mut c, d); // b_o
        ternarize(&mut out, &mut c, dims.d_inner, d); // w_gate
        ternarize(&mut out, &mut c, dims.d_inner, d); // w_up
        ternarize(&mut out, &mut c, d, dims.d_inner); // w_down
    }
    skip(&mut c, d + dims.vocab * d); // final norm, readout
    assert_eq!(c, latent.len());
    out
}

/// One layer's working parameters.
struct LayerParams {
    norm1_w: Vec<f32>,
    in_proj: LatentWeights,
    x_proj: LatentWeights,
    out_proj: LatentWeights,
    a_log: Vec<f32>,
    d_param: Vec<f32>,
    dt_bias: Vec<f32>,
    norm2_w: Vec<f32>,
    w_f: LatentWeights,
    b_f: Vec<f32>,
    w_c: LatentWeights,
    b_c: Vec<f32>,
    w_o: LatentWeights,
    b_o: Vec<f32>,
    w_gate: LatentWeights,
    w_up: LatentWeights,
    w_down: LatentWeights,
}

/// The dense trainable model, reconstructed from a flat parameter vector.
///
/// Carries per-layer recurrent state (SSM + MLGRU) so it can be stepped one
/// token at a time like the deployed core — `forward_window` ignores this
/// state (it threads local state for BPTT), but `decode_step` and the
/// state-snapshot API use it so the dynamics instrument can measure a trained
/// dense core's regime directly.
pub struct DenseModel {
    dims: DenseDims,
    embed: Vec<f32>,
    layers: Vec<LayerParams>,
    final_norm_w: Vec<f32>,
    readout: Vec<f32>,
    ssm_state: Vec<Vec<f32>>,
    gru_state: Vec<Vec<f32>>,
}

impl DenseModel {
    /// Parse a flat parameter vector in canonical order. Panics unless
    /// `flat.len() == dims.total_len()`.
    pub fn from_flat(dims: DenseDims, flat: &[f32]) -> Self {
        assert_eq!(flat.len(), dims.total_len(), "flat param length mismatch");
        let d = dims.d_model;
        let sq = d * d;
        let mut c = 0usize;
        let mut take = |n: usize| {
            let s = &flat[c..c + n];
            c += n;
            s
        };

        let embed = take(dims.vocab * d).to_vec();
        let mut layers = Vec::with_capacity(dims.n_layers);
        for _ in 0..dims.n_layers {
            let norm1_w = take(d).to_vec();
            let in_proj = latent(d, d, take(sq));
            let x_proj = latent(dims.x_proj_rows(), d, take(dims.x_proj_rows() * d));
            let out_proj = latent(d, d, take(sq));
            let a_log = take(dims.state_len()).to_vec();
            let d_param = take(dims.n_heads).to_vec();
            let dt_bias = take(dims.n_heads).to_vec();
            let norm2_w = take(d).to_vec();
            let w_f = latent(d, d, take(sq));
            let b_f = take(d).to_vec();
            let w_c = latent(d, d, take(sq));
            let b_c = take(d).to_vec();
            let w_o = latent(d, d, take(sq));
            let b_o = take(d).to_vec();
            let w_gate = latent(dims.d_inner, d, take(dims.d_inner * d));
            let w_up = latent(dims.d_inner, d, take(dims.d_inner * d));
            let w_down = latent(d, dims.d_inner, take(d * dims.d_inner));
            layers.push(LayerParams {
                norm1_w, in_proj, x_proj, out_proj, a_log, d_param, dt_bias,
                norm2_w, w_f, b_f, w_c, b_c, w_o, b_o, w_gate, w_up, w_down,
            });
        }
        let final_norm_w = take(d).to_vec();
        let readout = take(dims.vocab * d).to_vec();
        assert_eq!(c, flat.len());
        let ssm_state = vec![vec![0.0; dims.state_len()]; dims.n_layers];
        let gru_state = vec![vec![0.0; d]; dims.n_layers];
        Self { dims, embed, layers, final_norm_w, readout, ssm_state, gru_state }
    }

    /// Zero all recurrent state.
    pub fn reset_state(&mut self) {
        for s in &mut self.ssm_state {
            s.fill(0.0);
        }
        for s in &mut self.gru_state {
            s.fill(0.0);
        }
    }

    /// Total recurrent-state dimension: per layer, SSM state plus MLGRU state.
    pub fn state_dim(&self) -> usize {
        self.dims.n_layers * (self.dims.state_len() + self.dims.d_model)
    }

    /// Snapshot the full recurrent state, block-major (per layer: SSM then
    /// MLGRU). The read half of the state API the regime twin needs.
    pub fn export_state(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.state_dim());
        for l in 0..self.dims.n_layers {
            out.extend_from_slice(&self.ssm_state[l]);
            out.extend_from_slice(&self.gru_state[l]);
        }
        out
    }

    /// Restore recurrent state from an `export_state` snapshot.
    pub fn import_state(&mut self, state: &[f32]) {
        assert_eq!(state.len(), self.state_dim(), "dense state len mismatch");
        let sl = self.dims.state_len();
        let d = self.dims.d_model;
        let mut c = 0;
        for l in 0..self.dims.n_layers {
            self.ssm_state[l].copy_from_slice(&state[c..c + sl]);
            c += sl;
            self.gru_state[l].copy_from_slice(&state[c..c + d]);
            c += d;
        }
    }

    /// Step one token, carrying and updating the recurrent state. Returns the
    /// post-final-norm hidden state and the logits. The hidden is the point
    /// where crystal modules apply (additive, pre-unembed); callers testing
    /// module corrections read it here and re-unembed after applying modules.
    /// This is `forward_window`'s per-token forward without the tape.
    pub fn decode_step_capture(&mut self, token: u32) -> (Vec<f32>, Vec<f32>) {
        let d = self.dims.d_model;
        let ssm_dims = self.dims.ssm();
        let mut x = self.embed[token as usize * d..(token as usize + 1) * d].to_vec();

        for l in 0..self.dims.n_layers {
            let lp = &self.layers[l];

            let mut n1 = vec![0.0f32; d];
            rmsnorm_forward(&x, &lp.norm1_w, RMS_EPS, &mut n1);
            let mut ssm_cache = SsmStepCache::new(ssm_dims);
            let mut ssm_out = vec![0.0f32; d];
            ssm_forward_latent(
                ssm_dims, &lp.in_proj, &lp.x_proj, &lp.out_proj,
                &lp.a_log, &lp.d_param, &lp.dt_bias,
                &n1, &self.ssm_state[l], &mut ssm_cache, &mut ssm_out,
            );
            self.ssm_state[l].copy_from_slice(&ssm_cache.next_state);
            for i in 0..d {
                x[i] += ssm_out[i];
            }

            let mut n2 = vec![0.0f32; d];
            rmsnorm_forward(&x, &lp.norm2_w, RMS_EPS, &mut n2);
            let mut f_pre = vec![0.0f32; d];
            let mut f_gate = vec![0.0f32; d];
            let mut c_pre = vec![0.0f32; d];
            let mut c_value = vec![0.0f32; d];
            let mut o_pre = vec![0.0f32; d];
            let mut o_gate = vec![0.0f32; d];
            let mut next_state = vec![0.0f32; d];
            let mut gru_out = vec![0.0f32; d];
            mlgru_forward_latent(
                &lp.w_f, &lp.b_f, &lp.w_c, &lp.b_c, &lp.w_o, &lp.b_o,
                &n2, &self.gru_state[l],
                &mut f_pre, &mut f_gate, &mut c_pre, &mut c_value,
                &mut o_pre, &mut o_gate, &mut next_state, &mut gru_out,
            );
            self.gru_state[l].copy_from_slice(&next_state);

            let mut gate_pre = vec![0.0f32; self.dims.d_inner];
            let mut gate = vec![0.0f32; self.dims.d_inner];
            let mut up = vec![0.0f32; self.dims.d_inner];
            let mut fused = vec![0.0f32; self.dims.d_inner];
            let mut mixer_out = vec![0.0f32; d];
            glu_forward_latent(
                &lp.w_gate, &lp.w_up, &lp.w_down, &gru_out,
                &mut gate_pre, &mut gate, &mut up, &mut fused, &mut mixer_out,
            );
            for i in 0..d {
                x[i] += mixer_out[i];
            }
        }

        let mut fn_out = vec![0.0f32; d];
        rmsnorm_forward(&x, &self.final_norm_w, RMS_EPS, &mut fn_out);
        let logits = self.unembed(&fn_out);
        (fn_out, logits)
    }

    /// Step one token, returning logits only (the common decode path).
    pub fn decode_step(&mut self, token: u32) -> Vec<f32> {
        self.decode_step_capture(token).1
    }

    /// Unembed a post-final-norm hidden state to vocab logits through the
    /// trained readout. Exposed so a caller can re-unembed after applying a
    /// crystal-module correction to the hidden.
    pub fn unembed(&self, hidden: &[f32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.dims.vocab];
        unembed_forward(&self.readout, self.dims.vocab, self.dims.d_model, hidden, &mut logits);
        logits
    }

    /// The untied readout table, row-major `[vocab × d_model]`. This is the
    /// token-direction table the distiller needs to relate hidden-state
    /// corrections to logits — the dense analog of the deployed core's tied
    /// embed/unembed table.
    pub fn readout(&self) -> &[f32] {
        &self.readout
    }

    /// This model's dimensions.
    pub fn dims(&self) -> DenseDims {
        self.dims
    }
}

/// Per-(timestep, layer) forward activations the backward needs.
#[derive(Clone)]
struct LayerStep {
    h_in: Vec<f32>,
    n1: Vec<f32>,
    ssm_prev_state: Vec<f32>,
    ssm_cache: SsmStepCache,
    h_mid: Vec<f32>,
    n2: Vec<f32>,
    mlgru_prev_state: Vec<f32>,
    // MLGRU cache fields the backward consumes.
    f_pre: Vec<f32>,
    f_gate: Vec<f32>,
    c_pre: Vec<f32>,
    c_value: Vec<f32>,
    o_gate: Vec<f32>,
    mlgru_next_state: Vec<f32>,
    gru_out: Vec<f32>,
    // GLU cache fields.
    gate_pre: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    fused: Vec<f32>,
}

/// Per-timestep tape.
#[derive(Clone)]
struct StepTape {
    token: u32,
    target: usize,
    layers: Vec<LayerStep>,
    h_final_in: Vec<f32>,
    fn_out: Vec<f32>,
    logits: Vec<f32>,
}

/// Full forward tape for a window.
pub struct Tape {
    steps: Vec<StepTape>,
}

impl DenseModel {
    /// Run the window forward, returning the summed cross-entropy loss and the
    /// tape. SSM and MLGRU states start at zero (window boundary detached).
    pub fn forward_window(&self, tokens: &[u32], targets: &[usize]) -> (f64, Tape) {
        assert_eq!(tokens.len(), targets.len());
        let d = self.dims.d_model;
        let ssm_dims = self.dims.ssm();

        // Running recurrent states, per layer.
        let mut ssm_state: Vec<Vec<f32>> =
            vec![vec![0.0; self.dims.state_len()]; self.dims.n_layers];
        let mut gru_state: Vec<Vec<f32>> = vec![vec![0.0; d]; self.dims.n_layers];

        let mut total_loss = 0.0f64;
        let mut steps = Vec::with_capacity(tokens.len());

        for (&token, &target) in tokens.iter().zip(targets.iter()) {
            // Embed.
            let mut x = self.embed[token as usize * d..(token as usize + 1) * d].to_vec();
            let mut layer_steps = Vec::with_capacity(self.dims.n_layers);

            for l in 0..self.dims.n_layers {
                let lp = &self.layers[l];
                let h_in = x.clone();

                // Sublayer 1: SSM token mixer with pre-norm + residual.
                let mut n1 = vec![0.0f32; d];
                rmsnorm_forward(&h_in, &lp.norm1_w, RMS_EPS, &mut n1);
                let ssm_prev_state = ssm_state[l].clone();
                let mut ssm_cache = SsmStepCache::new(ssm_dims);
                let mut ssm_out = vec![0.0f32; d];
                ssm_forward_latent(
                    ssm_dims, &lp.in_proj, &lp.x_proj, &lp.out_proj,
                    &lp.a_log, &lp.d_param, &lp.dt_bias,
                    &n1, &ssm_prev_state, &mut ssm_cache, &mut ssm_out,
                );
                ssm_state[l].copy_from_slice(&ssm_cache.next_state);
                let mut h_mid = h_in.clone();
                for i in 0..d {
                    h_mid[i] += ssm_out[i];
                }

                // Sublayer 2: MLGRU→GLU channel mixer with pre-norm + residual.
                let mut n2 = vec![0.0f32; d];
                rmsnorm_forward(&h_mid, &lp.norm2_w, RMS_EPS, &mut n2);
                let mlgru_prev_state = gru_state[l].clone();
                let mut f_pre = vec![0.0f32; d];
                let mut f_gate = vec![0.0f32; d];
                let mut c_pre = vec![0.0f32; d];
                let mut c_value = vec![0.0f32; d];
                let mut o_pre = vec![0.0f32; d];
                let mut o_gate = vec![0.0f32; d];
                let mut mlgru_next_state = vec![0.0f32; d];
                let mut gru_out = vec![0.0f32; d];
                mlgru_forward_latent(
                    &lp.w_f, &lp.b_f, &lp.w_c, &lp.b_c, &lp.w_o, &lp.b_o,
                    &n2, &mlgru_prev_state,
                    &mut f_pre, &mut f_gate, &mut c_pre, &mut c_value,
                    &mut o_pre, &mut o_gate, &mut mlgru_next_state, &mut gru_out,
                );
                gru_state[l].copy_from_slice(&mlgru_next_state);

                let mut gate_pre = vec![0.0f32; self.dims.d_inner];
                let mut gate = vec![0.0f32; self.dims.d_inner];
                let mut up = vec![0.0f32; self.dims.d_inner];
                let mut fused = vec![0.0f32; self.dims.d_inner];
                let mut mixer_out = vec![0.0f32; d];
                glu_forward_latent(
                    &lp.w_gate, &lp.w_up, &lp.w_down, &gru_out,
                    &mut gate_pre, &mut gate, &mut up, &mut fused, &mut mixer_out,
                );
                let mut x_out = h_mid.clone();
                for i in 0..d {
                    x_out[i] += mixer_out[i];
                }

                layer_steps.push(LayerStep {
                    h_in, n1, ssm_prev_state, ssm_cache, h_mid, n2, mlgru_prev_state,
                    f_pre, f_gate, c_pre, c_value, o_gate, mlgru_next_state, gru_out,
                    gate_pre, gate, up, fused,
                });
                x = x_out;
            }

            // Final norm + untied readout + loss.
            let h_final_in = x.clone();
            let mut fn_out = vec![0.0f32; d];
            rmsnorm_forward(&h_final_in, &self.final_norm_w, RMS_EPS, &mut fn_out);
            let mut logits = vec![0.0f32; self.dims.vocab];
            unembed_forward(&self.readout, self.dims.vocab, d, &fn_out, &mut logits);
            let mut throwaway = vec![0.0f32; self.dims.vocab];
            let loss = cross_entropy_logits_grad(&logits, target, &mut throwaway);
            total_loss += loss as f64;

            steps.push(StepTape { token, target, layers: layer_steps, h_final_in, fn_out, logits });
        }

        (total_loss, Tape { steps })
    }
}

/// Persistent per-layer gradient accumulators.
struct LayerGrads {
    norm1_w: Vec<f32>,
    in_proj: GradBuffer,
    x_proj: GradBuffer,
    out_proj: GradBuffer,
    a_log: Vec<f32>,
    d_param: Vec<f32>,
    dt_bias: Vec<f32>,
    norm2_w: Vec<f32>,
    w_f: GradBuffer,
    b_f: Vec<f32>,
    w_c: GradBuffer,
    b_c: Vec<f32>,
    w_o: GradBuffer,
    b_o: Vec<f32>,
    w_gate: GradBuffer,
    w_up: GradBuffer,
    w_down: GradBuffer,
}

impl LayerGrads {
    fn new(dims: DenseDims) -> Self {
        let d = dims.d_model;
        Self {
            norm1_w: vec![0.0; d],
            in_proj: GradBuffer::new(d, d),
            x_proj: GradBuffer::new(dims.x_proj_rows(), d),
            out_proj: GradBuffer::new(d, d),
            a_log: vec![0.0; dims.state_len()],
            d_param: vec![0.0; dims.n_heads],
            dt_bias: vec![0.0; dims.n_heads],
            norm2_w: vec![0.0; d],
            w_f: GradBuffer::new(d, d),
            b_f: vec![0.0; d],
            w_c: GradBuffer::new(d, d),
            b_c: vec![0.0; d],
            w_o: GradBuffer::new(d, d),
            b_o: vec![0.0; d],
            w_gate: GradBuffer::new(dims.d_inner, d),
            w_up: GradBuffer::new(dims.d_inner, d),
            w_down: GradBuffer::new(d, dims.d_inner),
        }
    }
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s;
    }
}

impl DenseModel {
    /// Backward over the whole window. Writes `d loss / d param` for every
    /// trainable family into `flat_grad`, in the same canonical order
    /// `from_flat` reads. Truncated BPTT: the final recurrent states are
    /// treated as unconsumed (zero future-state gradient at the window end).
    pub fn backward_window(&self, tape: &Tape, flat_grad: &mut [f64]) {
        assert_eq!(flat_grad.len(), self.dims.total_len());
        let d = self.dims.d_model;
        let ssm_dims = self.dims.ssm();
        let vocab = self.dims.vocab;

        let mut grad_embed = vec![0.0f32; vocab * d];
        let mut grad_readout = vec![0.0f32; vocab * d];
        let mut grad_final_norm_w = vec![0.0f32; d];
        let mut layer_grads: Vec<LayerGrads> =
            (0..self.dims.n_layers).map(|_| LayerGrads::new(self.dims)).collect();

        // Through-time state gradients, per layer, carried across timesteps.
        let mut future_ssm_grad: Vec<Vec<f32>> =
            vec![vec![0.0; self.dims.state_len()]; self.dims.n_layers];
        let mut future_gru_grad: Vec<Vec<f32>> = vec![vec![0.0; d]; self.dims.n_layers];

        for st in tape.steps.iter().rev() {
            // Loss → readout → final norm.
            let mut grad_logits = vec![0.0f32; vocab];
            cross_entropy_logits_grad(&st.logits, st.target, &mut grad_logits);
            let mut grad_fn_out = vec![0.0f32; d];
            unembed_backward(
                &self.readout, vocab, d, &st.fn_out, &grad_logits,
                &mut grad_fn_out, &mut grad_readout,
            );
            let mut upstream = vec![0.0f32; d]; // dL/d (last layer h_out)
            rmsnorm_backward(
                &st.h_final_in, &self.final_norm_w, RMS_EPS, &grad_fn_out,
                &mut upstream, &mut grad_final_norm_w,
            );

            // Reverse through the layer stack.
            for l in (0..self.dims.n_layers).rev() {
                let lp = &self.layers[l];
                let lg = &mut layer_grads[l];
                let ls = &st.layers[l];

                // Sublayer 2: h_out = h_mid + GLU(MLGRU(norm2(h_mid))).
                let grad_residual2 = upstream.clone();
                let grad_mixer_out = upstream; // moved: dL/d mixer_out

                let mut grad_gru_out = vec![0.0f32; d];
                let mut g_fused = vec![0.0f32; self.dims.d_inner];
                let mut g_gate_pre = vec![0.0f32; self.dims.d_inner];
                let mut g_up = vec![0.0f32; self.dims.d_inner];
                glu_backward_latent(
                    &lp.w_gate, &lp.w_up, &lp.w_down, &ls.gru_out,
                    &ls.gate_pre, &ls.gate, &ls.up, &ls.fused, &grad_mixer_out,
                    &mut grad_gru_out, &mut lg.w_gate, &mut lg.w_up, &mut lg.w_down,
                    &mut g_fused, &mut g_gate_pre, &mut g_up,
                );

                let mut grad_n2 = vec![0.0f32; d];
                let mut grad_prev_gru = vec![0.0f32; d];
                let mut g_state_total = vec![0.0f32; d];
                let mut g_f_pre = vec![0.0f32; d];
                let mut g_c_pre = vec![0.0f32; d];
                let mut g_o_pre = vec![0.0f32; d];
                mlgru_backward_latent(
                    &lp.w_f, &lp.w_c, &lp.w_o, &ls.n2, &ls.mlgru_prev_state,
                    &ls.f_pre, &ls.f_gate, &ls.c_pre, &ls.c_value, &ls.o_gate,
                    &ls.mlgru_next_state, &grad_gru_out, &future_gru_grad[l],
                    &mut grad_n2, &mut grad_prev_gru,
                    &mut lg.w_f, &mut lg.b_f, &mut lg.w_c, &mut lg.b_c,
                    &mut lg.w_o, &mut lg.b_o,
                    &mut g_state_total, &mut g_f_pre, &mut g_c_pre, &mut g_o_pre,
                );
                future_gru_grad[l] = grad_prev_gru;

                let mut grad_h_mid = vec![0.0f32; d];
                rmsnorm_backward(
                    &ls.h_mid, &lp.norm2_w, RMS_EPS, &grad_n2,
                    &mut grad_h_mid, &mut lg.norm2_w,
                );
                add_into(&mut grad_h_mid, &grad_residual2);

                // Sublayer 1: h_mid = h_in + SSM(norm1(h_in)).
                let grad_residual1 = grad_h_mid.clone();
                let grad_ssm_out = grad_h_mid; // moved: dL/d ssm_out

                let mut step_grads = SsmStepGrads::new(ssm_dims);
                ssm_backward_latent(
                    ssm_dims, &lp.in_proj, &lp.x_proj, &lp.out_proj,
                    &lp.a_log, &lp.d_param, &lp.dt_bias,
                    &ls.n1, &ls.ssm_prev_state, &ls.ssm_cache,
                    &grad_ssm_out, &future_ssm_grad[l], &mut step_grads,
                );
                future_ssm_grad[l] = step_grads.prev_state.clone();

                // Fold SSM weight/param grads into the persistent accumulators.
                add_into(&mut lg.in_proj.grads, &step_grads.in_proj.grads);
                add_into(&mut lg.x_proj.grads, &step_grads.x_proj.grads);
                add_into(&mut lg.out_proj.grads, &step_grads.out_proj.grads);
                add_into(&mut lg.a_log, &step_grads.a_log);
                add_into(&mut lg.d_param, &step_grads.d_param);
                add_into(&mut lg.dt_bias, &step_grads.dt_bias);

                let mut grad_h_in = vec![0.0f32; d];
                rmsnorm_backward(
                    &ls.h_in, &lp.norm1_w, RMS_EPS, &step_grads.input,
                    &mut grad_h_in, &mut lg.norm1_w,
                );
                add_into(&mut grad_h_in, &grad_residual1);

                upstream = grad_h_in; // becomes dL/d h_out for layer l-1
            }

            // Embedding scatter at this timestep.
            embedding_backward(st.token, vocab, d, &upstream, &mut grad_embed);
        }

        // Pack gradients in canonical order.
        let mut c = 0usize;
        let mut put = |src: &[f32], flat_grad: &mut [f64]| {
            for &g in src {
                flat_grad[c] = g as f64;
                c += 1;
            }
        };
        put(&grad_embed, flat_grad);
        for lg in &layer_grads {
            put(&lg.norm1_w, flat_grad);
            put(&lg.in_proj.grads, flat_grad);
            put(&lg.x_proj.grads, flat_grad);
            put(&lg.out_proj.grads, flat_grad);
            put(&lg.a_log, flat_grad);
            put(&lg.d_param, flat_grad);
            put(&lg.dt_bias, flat_grad);
            put(&lg.norm2_w, flat_grad);
            put(&lg.w_f.grads, flat_grad);
            put(&lg.b_f, flat_grad);
            put(&lg.w_c.grads, flat_grad);
            put(&lg.b_c, flat_grad);
            put(&lg.w_o.grads, flat_grad);
            put(&lg.b_o, flat_grad);
            put(&lg.w_gate.grads, flat_grad);
            put(&lg.w_up.grads, flat_grad);
            put(&lg.w_down.grads, flat_grad);
        }
        put(&grad_final_norm_w, flat_grad);
        put(&grad_readout, flat_grad);
        assert_eq!(c, flat_grad.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::check::{assert_gradient_close, GradCheckConfig};

    fn tiny_dims() -> DenseDims {
        DenseDims {
            d_model: 6,
            n_heads: 2,
            d_head: 3,
            d_state: 2,
            d_inner: 5,
            n_layers: 2,
            vocab: 4,
        }
    }

    /// Smooth deterministic init in the STE-mask-open zone (|w| < 1.5).
    /// Kept large enough (up to ~0.96) that on-path gradients sit well above
    /// the f32 finite-difference noise floor.
    fn init_params(n: usize) -> Vec<f32> {
        (0..n).map(|i| (((i * 7 + 3) % 17) as f32 - 8.0) * 0.12).collect()
    }

    // Central differences over an f32 forward cannot resolve a gradient much
    // below loss·2⁻²⁴/ε ≈ a few ×1e-5. denom_floor puts near-zero gradients on
    // an absolute-error footing so noise isn't scored as relative error; the
    // precise per-op math is covered by the leaf checks in grad.rs / ssm.rs.
    // This end-to-end check exists to catch COMPOSITION bugs — a dropped
    // residual, a mis-ordered layer, a broken state carry — which move
    // on-path gradients by O(1), far above this floor.
    const E2E: GradCheckConfig = GradCheckConfig {
        epsilon: 1e-2,
        // f32 central differences over a T-step through-time composition
        // bottom out around ~1.3% on the smaller on-path gradients (analytic
        // and numerical still agree to that, e.g. -6.70e-3 vs -6.78e-3). A
        // real composition bug lands at O(1), so this margin still catches it
        // with ~10× headroom; single-token wiring is checked tighter below.
        tolerance: 1.5e-2,
        denom_floor: 5e-3,
    };

    #[test]
    fn decode_step_matches_forward_window_logits() {
        // decode_step (stateful, tape-free) and forward_window (BPTT tape) must
        // compute the same logits token-for-token from a zero-state start —
        // otherwise the regime instrument would measure different dynamics than
        // the trainer optimized.
        let dims = tiny_dims();
        let flat = init_params(dims.total_len());
        let tokens = vec![1u32, 3, 0, 2, 1];
        let targets = vec![0usize; tokens.len()];

        // forward_window computes per-step logits internally; reconstruct them
        // by asking a stepping model for each token.
        let model = DenseModel::from_flat(dims, &flat);
        let (_, tape) = model.forward_window(&tokens, &targets);

        let mut stepper = DenseModel::from_flat(dims, &flat);
        stepper.reset_state();
        for (t, tok) in tokens.iter().enumerate() {
            let logits = stepper.decode_step(*tok);
            for (a, b) in logits.iter().zip(tape_logits(&tape, t).iter()) {
                assert!((a - b).abs() < 1e-6, "logit mismatch at step {}: {} vs {}", t, a, b);
            }
        }
    }

    // Accessor for a timestep's logits from the (private) tape.
    fn tape_logits(tape: &Tape, t: usize) -> Vec<f32> {
        tape.steps[t].logits.clone()
    }

    #[test]
    fn dense_state_snapshot_roundtrips() {
        let dims = tiny_dims();
        let flat = init_params(dims.total_len());
        let drive: Vec<u32> = (0..30u32).map(|i| (i * 5 + 1) % dims.vocab as u32).collect();

        let mut model = DenseModel::from_flat(dims, &flat);
        model.reset_state();
        for &t in &drive[..15] {
            let _ = model.decode_step(t);
        }
        let snap = model.export_state();
        assert_eq!(snap.len(), model.state_dim());

        let first: Vec<Vec<f32>> = drive[15..].iter().map(|&t| model.decode_step(t)).collect();
        model.import_state(&snap);
        let second: Vec<Vec<f32>> = drive[15..].iter().map(|&t| model.decode_step(t)).collect();
        assert_eq!(first, second, "replay from imported dense state must be identical");
    }

    #[test]
    fn dense_dims_layer_len_matches_pack() {
        // total_len must equal what from_flat consumes.
        let dims = tiny_dims();
        let flat = init_params(dims.total_len());
        let _ = DenseModel::from_flat(dims, &flat); // panics on length mismatch
    }

    #[test]
    fn dense_model_end_to_end_gradient_matches_finite_difference() {
        let dims = tiny_dims();
        let flat = init_params(dims.total_len());
        // A window of three tokens with arbitrary next-token targets — the
        // check is about gradient correctness, not language semantics.
        let tokens = vec![1u32, 3, 0];
        let targets = vec![2usize, 0, 3];

        assert_gradient_close(
            &flat,
            |p| {
                let model = DenseModel::from_flat(dims, p);
                let (loss, _) = model.forward_window(&tokens, &targets);
                loss
            },
            |p, out| {
                let model = DenseModel::from_flat(dims, p);
                let (_, tape) = model.forward_window(&tokens, &targets);
                model.backward_window(&tape, out);
            },
            E2E,
        );
    }

    /// Canonical (name, length) spans of the flat gradient, in pack order.
    fn family_spans(dims: DenseDims) -> Vec<(&'static str, usize)> {
        let d = dims.d_model;
        let sq = d * d;
        let mut v = vec![("embed", dims.vocab * d)];
        for _ in 0..dims.n_layers {
            v.extend_from_slice(&[
                ("norm1", d), ("in_proj", sq), ("x_proj", dims.x_proj_rows() * d),
                ("out_proj", sq), ("a_log", dims.state_len()), ("d_param", dims.n_heads),
                ("dt_bias", dims.n_heads), ("norm2", d),
                ("w_f", sq), ("b_f", d), ("w_c", sq), ("b_c", d), ("w_o", sq), ("b_o", d),
                ("w_gate", dims.d_inner * d), ("w_up", dims.d_inner * d), ("w_down", d * dims.d_inner),
            ]);
        }
        v.push(("final_norm", d));
        v.push(("readout", dims.vocab * d));
        v
    }

    /// Guards against a vacuous pass: if a whole family's gradient were
    /// silently stuck at zero, the finite-difference check could accept it by
    /// matching a near-zero numerical gradient. Assert every family is
    /// non-trivially exercised by this window.
    #[test]
    fn every_trainable_family_has_nonzero_gradient() {
        let dims = tiny_dims();
        let flat = init_params(dims.total_len());
        let tokens = vec![1u32, 3, 0];
        let targets = vec![2usize, 0, 3];
        let model = DenseModel::from_flat(dims, &flat);
        let (_, tape) = model.forward_window(&tokens, &targets);
        let mut grad = vec![0.0f64; dims.total_len()];
        model.backward_window(&tape, &mut grad);

        let mut c = 0usize;
        for (name, len) in family_spans(dims) {
            let max_abs = grad[c..c + len].iter().fold(0.0f64, |m, &g| m.max(g.abs()));
            assert!(
                max_abs > 1e-4,
                "family '{}' has near-zero gradient (max_abs {:.2e}) — the end-to-end \
                 check would pass vacuously for it",
                name, max_abs,
            );
            c += len;
        }
        assert_eq!(c, grad.len());
    }

    #[test]
    fn single_token_window_also_checks() {
        // T=1 exercises the same tape with no through-time carry — a useful
        // isolation of the within-timestep residual/backward wiring.
        let dims = tiny_dims();
        let flat = init_params(dims.total_len());
        let tokens = vec![2u32];
        let targets = vec![1usize];
        // Tighter: with no through-time accumulation the composition holds to
        // ~1% in f32.
        let tight = GradCheckConfig { tolerance: 1e-2, ..E2E };
        assert_gradient_close(
            &flat,
            |p| DenseModel::from_flat(dims, p).forward_window(&tokens, &targets).0,
            |p, out| {
                let model = DenseModel::from_flat(dims, p);
                let (_, tape) = model.forward_window(&tokens, &targets);
                model.backward_window(&tape, out);
            },
            tight,
        );
    }
}
