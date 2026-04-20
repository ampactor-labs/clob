//! RMS Normalization with learnable scale parameter.

use crate::simd::KernelDispatch;
use crate::tensor::Tensor;

/// RMSNorm: output = (x / RMS(x)) * weight
pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn ones(dim: usize) -> Self {
        Self { weight: vec![1.0; dim], eps: 1e-6 }
    }

    pub fn forward(&self, x: &mut Tensor, dispatch: &KernelDispatch) {
        dispatch.rmsnorm(x.data_mut(), &self.weight, self.eps);
    }

    pub fn dim(&self) -> usize { self.weight.len() }
    pub fn weight_slice(&self) -> &[f32] { &self.weight }
}
