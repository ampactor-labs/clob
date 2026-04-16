//! CoreModel — the full ternary recurrent stack.
//!
//! Embed → [Block × L] → FinalNorm → LM Head.
//! O(1) decode memory. Zero heap allocations on the hot path.

use crate::model::block::Block;
use crate::model::config::KernelConfig;
use crate::nn::embed::Embedding;
use crate::nn::energy::EnergyCritic;
use crate::nn::rmsnorm::RMSNorm;
use crate::simd::KernelDispatch;
use crate::tensor::Tensor;
use rand::Rng;

/// The complete ternary recurrent core.
pub struct CoreModel {
    pub config: KernelConfig,
    embedding: Embedding,
    blocks: Vec<Block>,
    final_norm: RMSNorm,
    energy_critic: EnergyCritic,
    dispatch: KernelDispatch,
    last_hidden: Tensor,
    buf_x: Tensor,
    buf_logits: Tensor,
}

impl CoreModel {
    /// Create with random weights for testing.
    pub fn random(config: KernelConfig, rng: &mut impl Rng) -> Self {
        let embedding = Embedding::random(config.vocab_size, config.d_model, rng);
        let blocks: Vec<Block> = (0..config.n_layers)
            .map(|i| Block::random(&config, i, rng))
            .collect();
        let energy_critic = EnergyCritic::random(config.d_model, rng);
        let dispatch = KernelDispatch::new();
        let last_hidden = Tensor::zeros(&[config.d_model]);
        let buf_x = Tensor::zeros(&[config.d_model]);
        let buf_logits = Tensor::zeros(&[config.vocab_size]);

        Self {
            config: config.clone(),
            embedding,
            blocks,
            final_norm: RMSNorm::ones(config.d_model),
            energy_critic,
            dispatch,
            last_hidden,
            buf_x,
            buf_logits,
        }
    }

    /// Create from pre-loaded components.
    pub fn from_parts(
        config: KernelConfig,
        embedding: Embedding,
        blocks: Vec<Block>,
        final_norm: RMSNorm,
        energy_critic: EnergyCritic,
    ) -> Self {
        let dispatch = KernelDispatch::new();
        let last_hidden = Tensor::zeros(&[config.d_model]);
        let buf_x = Tensor::zeros(&[config.d_model]);
        let buf_logits = Tensor::zeros(&[config.vocab_size]);
        Self {
            config: config.clone(), embedding, blocks, final_norm,
            energy_critic, dispatch, last_hidden, buf_x, buf_logits,
        }
    }

    /// Process a single token. Returns logits.
    pub fn decode_step(&mut self, token: u32) -> Tensor {
        self.embedding.embed(token, &mut self.buf_x);

        for block in self.blocks.iter_mut() {
            let _ = block.forward(&mut self.buf_x, &self.dispatch);
        }

        self.final_norm.forward(&mut self.buf_x, &self.dispatch);
        self.last_hidden.copy_from(&self.buf_x);
        self.embedding.unembed(&self.buf_x, &mut self.buf_logits);

        Tensor::from_vec(self.buf_logits.data().to_vec(), &[self.config.vocab_size])
    }

    /// Prefill: process tokens sequentially, return final logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Tensor {
        let mut logits = Tensor::zeros(&[self.config.vocab_size]);
        for &token in tokens {
            logits = self.decode_step(token);
        }
        logits
    }

    /// Score + detect novelty of current hidden state.
    pub fn energy_score_and_detect(&mut self) -> (f32, bool) {
        self.energy_critic.score_and_detect(&self.last_hidden)
    }

    /// Raw energy score (no detection).
    pub fn energy_score(&self) -> f32 {
        self.energy_critic.score(&self.last_hidden)
    }

    /// Energy baseline.
    pub fn energy_baseline(&self) -> f32 {
        self.energy_critic.baseline()
    }

    /// Last hidden state (for episodic memory).
    pub fn last_hidden(&self) -> &Tensor {
        &self.last_hidden
    }

    /// Reset all recurrent states.
    pub fn reset_state(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_state();
        }
        self.last_hidden.zero_();
    }

    /// Reference to dispatch.
    pub fn dispatch(&self) -> &KernelDispatch {
        &self.dispatch
    }
}
