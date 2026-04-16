//! TernaryLinear — replaces all nn.Linear.
//!
//! Forward: quantize activations → ternary accumulate ⊛ → rescale + bias.
//! No floating-point multiplications in the weight path.

use crate::simd::KernelDispatch;
use crate::tensor::ternary::TernaryMatrix;
use crate::tensor::Tensor;
use rand::Rng;

/// A linear layer with ternary {-1, 0, 1} weights.
pub struct TernaryLinear {
    pub weight: TernaryMatrix,
    pub bias: Option<Vec<f32>>,
}

impl TernaryLinear {
    pub fn new(weight: TernaryMatrix, bias: Option<Vec<f32>>) -> Self {
        if let Some(ref b) = bias {
            assert_eq!(b.len(), weight.rows());
        }
        Self { weight, bias }
    }

    pub fn random(in_features: usize, out_features: usize, with_bias: bool, rng: &mut impl Rng) -> Self {
        let weight = TernaryMatrix::random(out_features, in_features, rng);
        let bias = if with_bias {
            Some((0..out_features).map(|_| rng.gen_range(-0.1..0.1)).collect())
        } else {
            None
        };
        Self { weight, bias }
    }

    /// Forward: output = W ⊛ input + bias.
    pub fn forward(&self, input: &Tensor, output: &mut Tensor, dispatch: &KernelDispatch) {
        assert_eq!(input.len(), self.weight.cols());
        assert_eq!(output.len(), self.weight.rows());
        dispatch.ternary_accumulate(input.data(), &self.weight, output.data_mut());
        if let Some(ref bias) = self.bias {
            output.add_slice_(bias);
        }
    }

    pub fn in_features(&self) -> usize { self.weight.cols() }
    pub fn out_features(&self) -> usize { self.weight.rows() }
}
