//! Episode — a single recorded experience.
//!
//! Each episode captures a moment where the model encountered novelty:
//! what was the input, what was the hidden state, what did the model
//! predict, and what actually came next.

use serde::{Deserialize, Serialize};

/// A single recorded experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Monotonic timestamp (nanoseconds since boot).
    pub timestamp: u64,
    /// Input token IDs that led to this state.
    pub context: Vec<u32>,
    /// Hidden state snapshot (d_model floats).
    pub hidden_state: Vec<f32>,
    /// Energy score at this moment (novelty).
    pub energy: f32,
    /// Top-k predictions: (token_id, probability).
    pub predictions: Vec<(u32, f32)>,
    /// What actually came next.
    pub actual_token: u32,
    /// Prediction error: 1.0 - P(actual).
    pub prediction_error: f32,
    /// Whether this episode has been consumed by crystallization.
    pub consumed: bool,
}

impl Episode {
    /// Create a new episode.
    pub fn new(
        timestamp: u64,
        context: Vec<u32>,
        hidden_state: Vec<f32>,
        energy: f32,
        predictions: Vec<(u32, f32)>,
        actual_token: u32,
    ) -> Self {
        let pred_prob = predictions.iter()
            .find(|(id, _)| *id == actual_token)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);

        Self {
            timestamp,
            context,
            hidden_state,
            energy,
            predictions,
            actual_token,
            prediction_error: 1.0 - pred_prob,
            consumed: false,
        }
    }

    /// Serialize to bytes (bincode).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).expect("serialize episode")
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }

    /// Approximate size in bytes.
    pub fn approx_size(&self) -> usize {
        8 + // timestamp
        self.context.len() * 4 +
        self.hidden_state.len() * 4 +
        4 + // energy
        self.predictions.len() * 8 +
        4 + // actual_token
        4 + // prediction_error
        1   // consumed
    }
}
