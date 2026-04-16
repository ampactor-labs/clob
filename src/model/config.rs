//! Kernel configuration — all hyperparameters.

use serde::{Deserialize, Serialize};

/// Complete configuration for the kernel's ternary recurrent core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Hidden dimension.
    pub d_model: usize,
    /// Number of blocks.
    pub n_layers: usize,
    /// SSM state dimension per head.
    pub d_state: usize,
    /// Number of SSM heads.
    pub n_heads: usize,
    /// Inner dimension for GLU.
    pub d_inner: usize,
    /// Total experts per MoE layer.
    pub n_experts: usize,
    /// Active experts per token.
    pub n_active_experts: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Which layers use MoE (rest are dense MLGRU+GLU).
    pub moe_layers: Vec<usize>,
}

impl KernelConfig {
    /// Tiny config for testing (d=64, L=2, V=256).
    pub fn tiny() -> Self {
        Self {
            d_model: 64,
            n_layers: 2,
            d_state: 16,
            n_heads: 4,
            d_inner: 128,
            n_experts: 4,
            n_active_experts: 2,
            vocab_size: 256,
            moe_layers: vec![1],
        }
    }

    /// Small config suitable for a 512-merge BPE tokenizer (d=128, L=4, V=1024).
    pub fn small() -> Self {
        Self {
            d_model: 128,
            n_layers: 4,
            d_state: 16,
            n_heads: 4,
            d_inner: 256,
            n_experts: 4,
            n_active_experts: 2,
            vocab_size: 1024,
            moe_layers: vec![2],
        }
    }

    /// Seed config for the T490 boot (d=512, L=8, V=4096).
    pub fn seed() -> Self {
        Self {
            d_model: 512,
            n_layers: 8,
            d_state: 32,
            n_heads: 8,
            d_inner: 1024,
            n_experts: 8,
            n_active_experts: 2,
            vocab_size: 4096,
            moe_layers: vec![2, 5, 7],
        }
    }

    pub fn d_head(&self) -> usize { self.d_model / self.n_heads }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.moe_layers.contains(&layer_idx)
    }

    /// Rough parameter count estimate.
    pub fn param_count_estimate(&self) -> usize {
        let embed = self.vocab_size * self.d_model;
        let per_block = {
            let ssm = 4 * self.d_model * self.d_model;
            let mlgru = 3 * self.d_model * self.d_model;
            let glu = 3 * self.d_model * self.d_inner;
            ssm + mlgru + glu
        };
        embed + per_block * self.n_layers
    }
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self::seed()
    }
}
