//! MatMul-Free Linear GRU — the channel mixer's gating mechanism.
//!
//! All projections use ternary accumulation ⊛.

use crate::nn::ternary_linear::TernaryLinear;
use crate::simd::KernelDispatch;
use crate::tensor::Tensor;
use rand::Rng;

/// MLGRU: f_t = σ(x⊛W_f), c_t = SiLU(x⊛W_c),
///        h_t = f_t⊙h_{t-1} + (1-f_t)⊙c_t, o_t = h_t⊙σ(x⊛W_o)
pub struct MLGRU {
    w_f: TernaryLinear,
    w_c: TernaryLinear,
    w_o: TernaryLinear,
    d_model: usize,
    state: Vec<f32>,
    buf_f: Tensor,
    buf_c: Tensor,
    buf_o: Tensor,
}

impl MLGRU {
    pub fn random(d_model: usize, rng: &mut impl Rng) -> Self {
        Self {
            w_f: TernaryLinear::random(d_model, d_model, true, rng),
            w_c: TernaryLinear::random(d_model, d_model, true, rng),
            w_o: TernaryLinear::random(d_model, d_model, true, rng),
            d_model,
            state: vec![0.0; d_model],
            buf_f: Tensor::zeros(&[d_model]),
            buf_c: Tensor::zeros(&[d_model]),
            buf_o: Tensor::zeros(&[d_model]),
        }
    }

    pub fn from_weights(w_f: TernaryLinear, w_c: TernaryLinear, w_o: TernaryLinear, d_model: usize) -> Self {
        Self {
            w_f, w_c, w_o, d_model,
            state: vec![0.0; d_model],
            buf_f: Tensor::zeros(&[d_model]),
            buf_c: Tensor::zeros(&[d_model]),
            buf_o: Tensor::zeros(&[d_model]),
        }
    }

    pub fn reset_state(&mut self) { self.state.fill(0.0); }
    pub fn state(&self) -> &[f32] { &self.state }

    /// Overwrite the recurrent state. Length must be `d_model`.
    pub fn set_state(&mut self, state: &[f32]) {
        assert_eq!(state.len(), self.state.len(),
            "mlgru state len {} != expected {}", state.len(), self.state.len());
        self.state.copy_from_slice(state);
    }

    pub fn w_f(&self) -> &TernaryLinear { &self.w_f }
    pub fn w_c(&self) -> &TernaryLinear { &self.w_c }
    pub fn w_o(&self) -> &TernaryLinear { &self.w_o }

    pub fn forward(&mut self, input: &Tensor, output: &mut Tensor, dispatch: &KernelDispatch) {
        assert_eq!(input.len(), self.d_model);
        assert_eq!(output.len(), self.d_model);

        self.w_f.forward(input, &mut self.buf_f, dispatch);
        dispatch.sigmoid_inplace(self.buf_f.data_mut());

        self.w_c.forward(input, &mut self.buf_c, dispatch);
        dispatch.silu_inplace(self.buf_c.data_mut());

        self.w_o.forward(input, &mut self.buf_o, dispatch);
        dispatch.sigmoid_inplace(self.buf_o.data_mut());

        let out = output.data_mut();
        let f = self.buf_f.data();
        let c = self.buf_c.data();
        let o = self.buf_o.data();
        for i in 0..self.d_model {
            self.state[i] = f[i] * self.state[i] + (1.0 - f[i]) * c[i];
            out[i] = self.state[i] * o[i];
        }
    }
}
