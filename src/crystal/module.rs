//! CrystalModule — a crystallized ternary expert.
//!
//! Contains a small TernaryLinear that encodes one learned pattern.
//! Serializable. Loadable via mmap. Has a domain signature for routing.

use crate::tensor::ternary::TernaryMatrix;
use crate::tensor::Tensor;
use crate::simd::KernelDispatch;
use serde::{Deserialize, Serialize};

/// A crystallized knowledge module.
pub struct CrystalModule {
    /// Unique module ID.
    pub id: u64,
    /// Ternary weight matrix [d_model × d_model].
    pub weight: TernaryMatrix,
    /// Domain signature (centroid of source episodes, for routing).
    pub domain_signature: Vec<f32>,
    /// Model dimension.
    pub d_model: usize,
    /// How many episodes were crystallized into this module.
    pub n_source_episodes: usize,
    /// Average error of source episodes before crystallization.
    pub avg_error_before: f32,
    /// MDL compression ratio achieved.
    pub mdl_ratio: f32,
    /// Fraction of zero weights.
    pub sparsity: f32,
    /// Number of times this module has been activated.
    pub activation_count: u64,
}

impl CrystalModule {
    /// Apply this module to a hidden state: output += W ⊛ input.
    pub fn apply(&mut self, input: &Tensor, output: &mut Tensor, dispatch: &KernelDispatch) {
        assert_eq!(input.len(), self.d_model);
        assert_eq!(output.len(), self.d_model);
        dispatch.ternary_accumulate(input.data(), &self.weight, output.data_mut());
        self.activation_count += 1;
    }

    /// Cosine similarity between input and this module's domain.
    pub fn domain_match(&self, hidden: &Tensor) -> f32 {
        assert_eq!(hidden.len(), self.d_model);
        let dot: f32 = hidden.data().iter().zip(self.domain_signature.iter())
            .map(|(a, b)| a * b).sum();
        let na: f32 = hidden.data().iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb: f32 = self.domain_signature.iter().map(|v| v * v).sum::<f32>().sqrt();
        if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 }
    }

    /// Metadata for serialization.
    pub fn metadata(&self) -> ModuleMeta {
        ModuleMeta {
            id: self.id,
            d_model: self.d_model,
            n_source_episodes: self.n_source_episodes,
            avg_error_before: self.avg_error_before,
            mdl_ratio: self.mdl_ratio,
            sparsity: self.sparsity,
            activation_count: self.activation_count,
        }
    }

    /// Byte size of this module's weight data.
    pub fn weight_bytes(&self) -> usize {
        self.weight.byte_size()
    }
}

/// Serializable module metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMeta {
    pub id: u64,
    pub d_model: usize,
    pub n_source_episodes: usize,
    pub avg_error_before: f32,
    pub mdl_ratio: f32,
    pub sparsity: f32,
    pub activation_count: u64,
}
