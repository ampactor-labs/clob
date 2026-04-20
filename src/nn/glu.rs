//! Ternary Gated Linear Unit — the channel mixer's FFN.
//! In MoE layers, each expert IS a GLU instance.

use crate::nn::ternary_linear::TernaryLinear;
use crate::simd::KernelDispatch;
use crate::tensor::Tensor;
use rand::Rng;

/// gate = SiLU(x⊛W_gate), up = x⊛W_up, output = (gate⊙up)⊛W_down
pub struct TernaryGLU {
    w_gate: TernaryLinear,
    w_up: TernaryLinear,
    w_down: TernaryLinear,
    d_model: usize,
    d_inner: usize,
    buf_gate: Tensor,
    buf_up: Tensor,
    buf_fused: Tensor,
}

impl TernaryGLU {
    pub fn random(d_model: usize, d_inner: usize, rng: &mut impl Rng) -> Self {
        Self {
            w_gate: TernaryLinear::random(d_model, d_inner, false, rng),
            w_up: TernaryLinear::random(d_model, d_inner, false, rng),
            w_down: TernaryLinear::random(d_inner, d_model, false, rng),
            d_model, d_inner,
            buf_gate: Tensor::zeros(&[d_inner]),
            buf_up: Tensor::zeros(&[d_inner]),
            buf_fused: Tensor::zeros(&[d_inner]),
        }
    }

    pub fn from_weights(
        w_gate: TernaryLinear, w_up: TernaryLinear, w_down: TernaryLinear,
        d_model: usize, d_inner: usize,
    ) -> Self {
        Self {
            w_gate, w_up, w_down, d_model, d_inner,
            buf_gate: Tensor::zeros(&[d_inner]),
            buf_up: Tensor::zeros(&[d_inner]),
            buf_fused: Tensor::zeros(&[d_inner]),
        }
    }

    pub fn forward(&mut self, input: &Tensor, output: &mut Tensor, dispatch: &KernelDispatch) {
        assert_eq!(input.len(), self.d_model);
        assert_eq!(output.len(), self.d_model);

        self.w_gate.forward(input, &mut self.buf_gate, dispatch);
        dispatch.silu_inplace(self.buf_gate.data_mut());

        self.w_up.forward(input, &mut self.buf_up, dispatch);
        dispatch.hadamard(self.buf_gate.data(), self.buf_up.data(), self.buf_fused.data_mut());

        self.w_down.forward(&self.buf_fused, output, dispatch);
    }

    pub fn d_model(&self) -> usize { self.d_model }
    pub fn d_inner(&self) -> usize { self.d_inner }

    pub fn w_gate(&self) -> &TernaryLinear { &self.w_gate }
    pub fn w_up(&self) -> &TernaryLinear { &self.w_up }
    pub fn w_down(&self) -> &TernaryLinear { &self.w_down }

    /// Raw memory extents of the ternary weights (for prefetch).
    pub fn memory_extents(&self) -> Vec<(*const u8, usize)> {
        vec![
            (self.w_gate.weight.packed_data().as_ptr(), self.w_gate.weight.packed_data().len()),
            (self.w_up.weight.packed_data().as_ptr(), self.w_up.weight.packed_data().len()),
            (self.w_down.weight.packed_data().as_ptr(), self.w_down.weight.packed_data().len()),
        ]
    }
}
