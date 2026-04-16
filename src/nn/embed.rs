//! Token embedding and unembedding (tied weights).
//!
//! Embeddings are dense f32 — NOT ternary — since embedding is a lookup,
//! not a matrix multiply.

use crate::tensor::Tensor;
use rand::Rng;

/// Token embedding + output projection with tied weights.
pub struct Embedding {
    /// [vocab_size × d_model], row-major.
    table: Vec<f32>,
    vocab_size: usize,
    d_model: usize,
}

impl Embedding {
    pub fn new(table: Vec<f32>, vocab_size: usize, d_model: usize) -> Self {
        assert_eq!(table.len(), vocab_size * d_model);
        Self { table, vocab_size, d_model }
    }

    pub fn random(vocab_size: usize, d_model: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (d_model as f32).sqrt();
        let table: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self { table, vocab_size, d_model }
    }

    /// Look up embedding for a single token.
    pub fn embed(&self, token_id: u32, output: &mut Tensor) {
        let id = token_id as usize;
        assert!(id < self.vocab_size, "token {} >= vocab {}", id, self.vocab_size);
        assert_eq!(output.len(), self.d_model);
        let start = id * self.d_model;
        output.data_mut().copy_from_slice(&self.table[start..start + self.d_model]);
    }

    /// Unembedding: logits[v] = dot(hidden, embedding[v]).
    pub fn unembed(&self, hidden: &Tensor, logits: &mut Tensor) {
        assert_eq!(hidden.len(), self.d_model);
        assert_eq!(logits.len(), self.vocab_size);
        let h = hidden.data();
        let out = logits.data_mut();

        use rayon::prelude::*;
        out.par_iter_mut().enumerate().with_min_len(512).for_each(|(v, logit)| {
            let start = v * self.d_model;
            let emb = &self.table[start..start + self.d_model];
            *logit = h.iter().zip(emb.iter()).map(|(a, b)| a * b).sum();
        });
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn d_model(&self) -> usize { self.d_model }
    pub fn table(&self) -> &[f32] { &self.table }
}
