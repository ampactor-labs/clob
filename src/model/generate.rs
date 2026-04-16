//! Autoregressive generation with sampling strategies.

use crate::model::stack::CoreModel;
use crate::tensor::Tensor;
use rand::Rng;

/// Sampling configuration.
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 50, top_p: 0.9 }
    }
}

/// Sample a token from logits.
pub fn sample_token(logits: &Tensor, config: &SamplingConfig, rng: &mut impl Rng) -> u32 {
    let mut probs: Vec<(usize, f32)> = logits.data().iter().enumerate()
        .map(|(i, &l)| (i, l)).collect();

    // Temperature
    if config.temperature > 0.0 && config.temperature != 1.0 {
        for (_, l) in probs.iter_mut() { *l /= config.temperature; }
    }

    // Sort descending
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Top-k
    if config.top_k > 0 && config.top_k < probs.len() {
        probs.truncate(config.top_k);
    }

    // Softmax
    let max_val = probs[0].1;
    let mut sum = 0.0f32;
    for (_, l) in probs.iter_mut() {
        *l = (*l - max_val).exp();
        sum += *l;
    }
    for (_, l) in probs.iter_mut() { *l /= sum; }

    // Top-p (nucleus)
    if config.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            cumulative += p;
            if cumulative >= config.top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        let new_sum: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() { *p /= new_sum; }
    }

    // Categorical sampling
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for &(idx, p) in probs.iter() {
        cumulative += p;
        if r < cumulative { return idx as u32; }
    }
    probs[0].0 as u32
}

/// Generate tokens autoregressively.
pub fn generate(
    model: &mut CoreModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    config: &SamplingConfig,
    rng: &mut impl Rng,
) -> Vec<u32> {
    let mut logits = model.prefill(prompt_tokens);
    let mut generated = Vec::with_capacity(max_tokens);

    for _step in 0..max_tokens {
        let token = sample_token(&logits, config, rng);
        generated.push(token);
        logits = model.decode_step(token);
    }

    generated
}
