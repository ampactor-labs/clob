//! Selective State Space Model — the token mixer.
//!
//! Mamba-2/SSD style SSM with input-dependent discretization.
//! Replaces attention. O(1) decode memory. No KV cache.

use crate::nn::ternary_linear::TernaryLinear;
use crate::simd::KernelDispatch;
use crate::tensor::Tensor;
use rand::Rng;

/// Selective SSM — the token mixer in each block.
///
/// h_t = Ā·h_{t-1} + B̄·x_t
/// y_t = C_t·h_t + D·x_t
pub struct SelectiveSSM {
    in_proj: TernaryLinear,
    x_proj: TernaryLinear,
    _dt_proj: TernaryLinear,
    out_proj: TernaryLinear,

    /// Log of diagonal A: [n_heads × d_state]
    a_log: Vec<f32>,
    /// D (skip connection): [n_heads]
    d_param: Vec<f32>,
    /// Δ bias: [n_heads]
    dt_bias: Vec<f32>,

    n_heads: usize,
    d_state: usize,
    _d_inner: usize,
    d_head: usize,

    /// Recurrent state: [n_heads × d_state]
    state: Vec<f32>,

    // Scratch buffers
    buf_z: Tensor,
    buf_xbc: Tensor,
    buf_y: Tensor,
}

impl SelectiveSSM {
    pub fn random(d_model: usize, n_heads: usize, d_state: usize, rng: &mut impl Rng) -> Self {
        let d_inner = d_model;
        let d_head = d_inner / n_heads;
        let x_proj_size = n_heads + 2 * n_heads * d_state;

        Self {
            in_proj: TernaryLinear::random(d_model, d_inner, false, rng),
            x_proj: TernaryLinear::random(d_inner, x_proj_size, false, rng),
            _dt_proj: TernaryLinear::random(n_heads, d_inner, false, rng),
            out_proj: TernaryLinear::random(d_inner, d_model, false, rng),
            a_log: (0..n_heads * d_state).map(|_| rng.gen_range(-2.0..-0.5)).collect(),
            d_param: (0..n_heads).map(|_| rng.gen_range(0.5..1.5)).collect(),
            dt_bias: (0..n_heads).map(|_| rng.gen_range(-0.5..0.5)).collect(),
            n_heads,
            d_state,
            _d_inner: d_inner,
            d_head,
            state: vec![0.0; n_heads * d_state],
            buf_z: Tensor::zeros(&[d_inner]),
            buf_xbc: Tensor::zeros(&[x_proj_size]),
            buf_y: Tensor::zeros(&[d_inner]),
        }
    }

    pub fn from_weights(
        in_proj: TernaryLinear, x_proj: TernaryLinear,
        dt_proj: TernaryLinear, out_proj: TernaryLinear,
        a_log: Vec<f32>, d_param: Vec<f32>, dt_bias: Vec<f32>,
        n_heads: usize, d_state: usize, d_inner: usize, d_head: usize,
    ) -> Self {
        let x_proj_size = n_heads + 2 * n_heads * d_state;
        Self {
            in_proj, x_proj, _dt_proj: dt_proj, out_proj,
            a_log, d_param, dt_bias,
            n_heads, d_state, _d_inner: d_inner, d_head,
            state: vec![0.0; n_heads * d_state],
            buf_z: Tensor::zeros(&[d_inner]),
            buf_xbc: Tensor::zeros(&[x_proj_size]),
            buf_y: Tensor::zeros(&[d_inner]),
        }
    }

    pub fn reset_state(&mut self) { self.state.fill(0.0); }
    pub fn state(&self) -> &[f32] { &self.state }

    pub fn in_proj(&self) -> &TernaryLinear { &self.in_proj }
    pub fn x_proj(&self) -> &TernaryLinear { &self.x_proj }
    pub fn dt_proj(&self) -> &TernaryLinear { &self._dt_proj }
    pub fn out_proj(&self) -> &TernaryLinear { &self.out_proj }
    pub fn a_log(&self) -> &[f32] { &self.a_log }
    pub fn d_param(&self) -> &[f32] { &self.d_param }
    pub fn dt_bias(&self) -> &[f32] { &self.dt_bias }
    pub fn n_heads(&self) -> usize { self.n_heads }
    pub fn d_state(&self) -> usize { self.d_state }
    pub fn d_head(&self) -> usize { self.d_head }

    /// Forward: single-step recurrence, O(1) memory.
    pub fn forward(&mut self, input: &Tensor, output: &mut Tensor, dispatch: &KernelDispatch) {
        let d_model = input.len();
        assert_eq!(output.len(), d_model);

        // 1. Input projection
        self.in_proj.forward(input, &mut self.buf_z, dispatch);

        // 2. Compute Δ, B, C
        self.x_proj.forward(&self.buf_z, &mut self.buf_xbc, dispatch);

        let xbc_data = self.buf_xbc.data();
        let z_data = self.buf_z.data();

        // 3. Discretize & recur per head
        let y_data = self.buf_y.data_mut();
        for v in y_data.iter_mut() { *v = 0.0; }

        for h in 0..self.n_heads {
            let dt_raw = xbc_data[h] + self.dt_bias[h];
            let dt_h = if dt_raw > 20.0 { dt_raw }
                       else if dt_raw < -20.0 { 0.0 }
                       else { (1.0 + dt_raw.exp()).ln() };

            let b_start = self.n_heads;
            let c_start = b_start + self.n_heads * self.d_state;

            let head_start = h * self.d_head;
            let mut x_avg = 0.0f32;
            for d in 0..self.d_head {
                x_avg += z_data[head_start + d];
            }
            x_avg /= self.d_head as f32;

            for s in 0..self.d_state {
                let idx = h * self.d_state + s;
                let a = self.a_log[idx].exp();
                let a_bar = (-a * dt_h).exp();
                let b_bar = xbc_data[b_start + idx] * dt_h;
                self.state[idx] = a_bar * self.state[idx] + b_bar * x_avg;
            }

            let mut y_h = 0.0f32;
            for s in 0..self.d_state {
                let idx = h * self.d_state + s;
                y_h += xbc_data[c_start + idx] * self.state[idx];
            }

            for d in 0..self.d_head {
                y_data[head_start + d] = y_h + self.d_param[h] * z_data[head_start + d];
            }
        }

        // 4. Output projection
        self.out_proj.forward(&self.buf_y, output, dispatch);
    }
}
