//! Tensor system — the data foundation.
//!
//! Provides a contiguous f32 tensor for activations and intermediates,
//! plus the ternary matrix type for packed {-1,0,1} weights.
//! All operations write into pre-allocated output buffers for
//! zero-allocation hot paths.

pub mod ternary;

/// A contiguous f32 tensor with shape tracking.
///
/// Inference-only — no autograd. Callers create output tensors once
/// and reuse them across decode steps.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape: shape.to_vec(),
        }
    }

    /// Create from existing data.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        assert_eq!(data.len(), len, "data length {} != shape product {}", data.len(), len);
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Immutable reference to the underlying data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Reset all values to zero.
    pub fn zero_(&mut self) {
        self.data.fill(0.0);
    }

    /// Copy data from another tensor of the same length.
    pub fn copy_from(&mut self, other: &Tensor) {
        assert_eq!(self.len(), other.len());
        self.data.copy_from_slice(&other.data);
    }

    /// Element-wise add: self += other.
    pub fn add_(&mut self, other: &Tensor) {
        assert_eq!(self.len(), other.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    /// Element-wise add with a slice: self += slice.
    pub fn add_slice_(&mut self, other: &[f32]) {
        assert_eq!(self.len(), other.len());
        for (a, b) in self.data.iter_mut().zip(other.iter()) {
            *a += b;
        }
    }

    /// Scale all elements: self *= s.
    pub fn scale_(&mut self, s: f32) {
        for v in self.data.iter_mut() {
            *v *= s;
        }
    }

    /// Element-wise multiply (Hadamard): out = self ⊙ other.
    pub fn hadamard(&self, other: &Tensor, out: &mut Tensor) {
        assert_eq!(self.len(), other.len());
        assert_eq!(self.len(), out.len());
        for i in 0..self.len() {
            out.data[i] = self.data[i] * other.data[i];
        }
    }

    /// In-place sigmoid: self = σ(self).
    pub fn sigmoid_(&mut self) {
        for v in self.data.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }

    /// In-place SiLU: self = self * σ(self).
    pub fn silu_(&mut self) {
        for v in self.data.iter_mut() {
            let sig = 1.0 / (1.0 + (-*v).exp());
            *v *= sig;
        }
    }

    /// Softmax over the entire tensor.
    pub fn softmax_(&mut self) {
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in self.data.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in self.data.iter_mut() {
                *v /= sum;
            }
        }
    }

    /// Dot product (flat vectors).
    pub fn dot(&self, other: &Tensor) -> f32 {
        assert_eq!(self.len(), other.len());
        self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum()
    }

    /// Absmax quantization to i8. Returns (scale, quantized).
    pub fn absmax_quantize(&self) -> (f32, Vec<i8>) {
        let abs_max = self.data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;
        let quantized: Vec<i8> = self.data.iter().map(|&v| {
            (v * inv_scale).round().clamp(-127.0, 127.0) as i8
        }).collect();
        (scale, quantized)
    }

    /// L2 norm of the tensor.
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Cosine similarity with another tensor.
    pub fn cosine_similarity(&self, other: &Tensor) -> f32 {
        let dot = self.dot(other);
        let na = self.norm();
        let nb = other.norm();
        if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
    }
}
