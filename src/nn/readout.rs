//! Trained output head (untied readout) — a sidecar over the frozen core.
//!
//! The first real-data run exposed that clob's output projection is the *tied*
//! embedding table, which at random init predicts *worse than uniform chance*
//! (holdout perplexity ≈ 4900 over a 4096-token vocab). The reservoir probe
//! (`probe-readout`) then showed that a trained f32 readout over the frozen
//! core recovers the token marginal (≈ 135 perplexity) but no contextual
//! signal beyond it — a real fix, and the first step of Path B (training the
//! core): the output head must be untied and trainable.
//!
//! This is the inference-side, persistable form of that readout: a dense f32
//! `logits = W·h + b` installed as a sidecar (like the confidence head and
//! routers) and used in place of the tied unembedding when present. Feature
//! standardization from training is folded into `W` and `b`, so it applies
//! directly to the raw hidden state. Nothing in the on-disk core format
//! changes; v1 models load unchanged.

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Untied output projection: `logits[v] = b[v] + dot(W_row_v, hidden)`.
/// `w` is `[vocab × d]` row-major; `b` is `[vocab]`.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedReadout {
    vocab: usize,
    d: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl TrainedReadout {
    pub fn new(vocab: usize, d: usize, w: Vec<f32>, b: Vec<f32>) -> Self {
        assert_eq!(w.len(), vocab * d, "W must be vocab×d");
        assert_eq!(b.len(), vocab, "b must be vocab");
        Self { vocab, d, w, b }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }
    pub fn d_model(&self) -> usize {
        self.d
    }

    /// Write `logits[v] = b[v] + dot(W_row_v, hidden)` for all v.
    pub fn unembed(&self, hidden: &Tensor, logits: &mut Tensor) {
        assert_eq!(hidden.len(), self.d);
        assert_eq!(logits.len(), self.vocab);
        let h = hidden.data();
        let out = logits.data_mut();
        let (w, b, d) = (&self.w, &self.b, self.d);
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().with_min_len(512).for_each(|(v, lo)| {
            let wr = &w[v * d..v * d + d];
            let mut s = b[v];
            for k in 0..d {
                s += wr[k] * h[k];
            }
            *lo = s;
        });
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("serialize trained readout")
    }

    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        bincode::deserialize(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_unembed() {
        let (vocab, d) = (5, 3);
        let w: Vec<f32> = (0..vocab * d).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..vocab).map(|i| i as f32).collect();
        let r = TrainedReadout::new(vocab, d, w, b);

        let h = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[d]);
        let mut logits = Tensor::zeros(&[vocab]);
        r.unembed(&h, &mut logits);
        // logit[0] = 0 + (0*1 + 0.1*2 + 0.2*3) = 0.8
        assert!((logits.data()[0] - 0.8).abs() < 1e-5, "got {}", logits.data()[0]);
        // logit[4] = 4 + (1.2*1 + 1.3*2 + 1.4*3) = 4 + 8.0 = 12.0
        assert!((logits.data()[4] - 12.0).abs() < 1e-4, "got {}", logits.data()[4]);

        let bytes = r.to_bytes();
        let r2 = TrainedReadout::from_bytes(&bytes).unwrap();
        let mut logits2 = Tensor::zeros(&[vocab]);
        r2.unembed(&h, &mut logits2);
        assert_eq!(logits.data(), logits2.data(), "roundtrip must be exact");
    }
}
