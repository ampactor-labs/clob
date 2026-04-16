//! SIMD kernel dispatch and trait definitions.
//!
//! Runtime CPUID dispatch selects between AVX2 micro-kernels
//! and bit-exact scalar fallbacks.

pub mod avx2;
pub mod scalar;

use crate::tensor::ternary::TernaryMatrix;

/// Unified interface for ternary accumulation and element-wise kernels.
pub trait TernaryKernel {
    /// Ternary matrix-vector product: output = weights ⊛ activations.
    fn ternary_accumulate(
        activations: &[f32],
        weights: &TernaryMatrix,
        output: &mut [f32],
    );

    fn sigmoid_inplace(x: &mut [f32]);
    fn silu_inplace(x: &mut [f32]);
    fn hadamard(a: &[f32], b: &[f32], out: &mut [f32]);
    fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32);
    fn softplus_inplace(x: &mut [f32]);
}

/// Runtime kernel dispatcher — selects the fastest available backend.
pub struct KernelDispatch {
    use_avx2: bool,
}

impl KernelDispatch {
    /// Detect CPU features and select the best backend.
    pub fn new() -> Self {
        let use_avx2 = Self::detect_avx2();
        if use_avx2 {
            eprintln!("[kernel] AVX2 kernels enabled");
        } else {
            eprintln!("[kernel] scalar fallback kernels");
        }
        Self { use_avx2 }
    }

    /// Force scalar-only mode (for testing).
    pub fn scalar_only() -> Self {
        Self { use_avx2: false }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect_avx2() -> bool {
        false
    }

    pub fn ternary_accumulate(&self, activations: &[f32], weights: &TernaryMatrix, output: &mut [f32]) {
        if self.use_avx2 {
            #[cfg(target_arch = "x86_64")]
            {
                avx2::Avx2Kernel::ternary_accumulate(activations, weights, output);
                return;
            }
        }
        scalar::ScalarKernel::ternary_accumulate(activations, weights, output);
    }

    pub fn sigmoid_inplace(&self, x: &mut [f32]) {
        scalar::ScalarKernel::sigmoid_inplace(x);
    }

    pub fn silu_inplace(&self, x: &mut [f32]) {
        scalar::ScalarKernel::silu_inplace(x);
    }

    pub fn hadamard(&self, a: &[f32], b: &[f32], out: &mut [f32]) {
        scalar::ScalarKernel::hadamard(a, b, out);
    }

    pub fn rmsnorm(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        scalar::ScalarKernel::rmsnorm(x, weight, eps);
    }

    pub fn softplus_inplace(&self, x: &mut [f32]) {
        scalar::ScalarKernel::softplus_inplace(x);
    }
}

impl Default for KernelDispatch {
    fn default() -> Self {
        Self::new()
    }
}
