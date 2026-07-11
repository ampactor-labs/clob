//! Bet 2 on the trained dense substrate — crystallize the core's own
//! prediction errors, then measure held-out NLL with modules loaded vs
//! cleared, at equal compute.
//!
//! The two arms differ only by an additive, cosine-routed correction applied
//! to the post-final-norm hidden before unembedding. The recurrent state and
//! the number of decode passes per token are identical, so any NLL change is
//! the modules' doing — Bet 2's "equal compute" condition, satisfied by
//! construction. Tokens no module routes to are bit-identical between arms,
//! which makes "drift on untouched inputs" structurally zero and lets a single
//! forward pass score both arms (one decode, two readouts).
//!
//! This tests crystallization on the exact object Phase 6 validated: the
//! trained dense core (its effective-ternary deployment view, or its latents).
//! It does not go through the deployed `CoreModel`, whose tied random readout
//! is not the trained model.

use crate::crystal::engine::CrystallizationEngine;
use crate::learn::backprop::DenseModel;
use crate::memory::episode::Episode;
use crate::memory::ring::EpisodicMemory;
use crate::tensor::Tensor;

/// Top-k predictions stored per episode (matches the deployed ingest path).
const TOP_K: usize = 10;

/// Softmax NLL, in nats, of the true token given raw logits.
fn nll_of(logits: &[f32], actual: usize) -> f64 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits {
        sum += (l - max).exp();
    }
    let p = (logits[actual] - max).exp() / sum.max(1e-30);
    -(p.max(1e-12) as f64).ln()
}

/// Stream `tokens` through the model in detached windows (matching the
/// training/eval regime — state reset every `window` tokens), recording an
/// episode wherever the prediction error `1 - P(actual)` exceeds
/// `error_threshold`, up to `max_episodes`. Each episode carries the
/// post-final-norm hidden (the space modules route and correct in), the
/// observed future up to `horizon`, and the top-k predictions. Deterministic
/// given model + tokens. Returns the number of episodes recorded.
pub fn capture_episodes(
    model: &mut DenseModel,
    tokens: &[u32],
    window: usize,
    error_threshold: f32,
    max_episodes: usize,
    horizon: usize,
    memory: &EpisodicMemory,
) -> usize {
    let limit = tokens.len();
    let mut recorded = 0usize;
    let mut ts = 0u64;
    let mut i = 0usize;
    while i + 1 < limit && recorded < max_episodes {
        model.reset_state();
        for _ in 0..window {
            if i + 1 >= limit || recorded >= max_episodes {
                break;
            }
            let (fn_out, logits) = model.decode_step_capture(tokens[i]);
            let actual = tokens[i + 1];

            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for &l in &logits {
                sum += (l - max).exp();
            }
            let p_actual = (logits[actual as usize] - max).exp() / sum.max(1e-30);
            let err = 1.0 - p_actual;

            if err > error_threshold {
                let mut top: Vec<(u32, f32)> = logits
                    .iter()
                    .enumerate()
                    .map(|(t, &l)| (t as u32, (l - max).exp() / sum.max(1e-30)))
                    .collect();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                top.truncate(TOP_K);

                let ctx_start = i.saturating_sub(31);
                let context = tokens[ctx_start..=i].to_vec();
                let fut_end = (i + 1 + horizon).min(limit);
                let future = tokens[i + 1..fut_end].to_vec();

                let ep = Episode::new(ts, context, fn_out, err, top, actual, future);
                if memory.store(&ep).is_ok() {
                    ts += 1;
                    recorded += 1;
                }
            }
            i += 1;
        }
    }
    recorded
}

/// Result of the equal-compute A/B held-out evaluation. NLLs are per-token
/// means in nats.
#[derive(Debug, Clone)]
pub struct AbResult {
    /// Tokens scored.
    pub n: usize,
    /// Mean held-out NLL, modules cleared.
    pub nll_cleared: f64,
    /// Mean held-out NLL, modules loaded.
    pub nll_loaded: f64,
    /// Tokens where at least one module routed (cosine > activation threshold).
    pub n_activated: usize,
    /// Mean cleared NLL over the activated subset (0 if none activated).
    pub nll_cleared_activated: f64,
    /// Mean loaded NLL over the activated subset (0 if none activated).
    pub nll_loaded_activated: f64,
}

impl AbResult {
    /// Relative NLL reduction of the loaded arm vs the cleared arm, in percent
    /// — Bet 2's headline metric (positive means modules helped).
    pub fn delta_pct(&self) -> f64 {
        if self.nll_cleared > 0.0 {
            (self.nll_cleared - self.nll_loaded) / self.nll_cleared * 100.0
        } else {
            0.0
        }
    }

    /// Relative NLL change on the activated subset only (positive = helped).
    pub fn delta_pct_activated(&self) -> f64 {
        if self.n_activated > 0 && self.nll_cleared_activated > 0.0 {
            (self.nll_cleared_activated - self.nll_loaded_activated) / self.nll_cleared_activated
                * 100.0
        } else {
            0.0
        }
    }
}

/// Single-pass equal-compute A/B over `tokens`, windowed/detached to match
/// training. Each step scores the cleared arm from the post-final-norm hidden,
/// then re-scores a copy of that hidden after the engine's modules apply
/// (additive, cosine-routed). One forward pass, two readouts. Tokens no module
/// routes to score identically in both arms.
pub fn ab_eval(
    model: &mut DenseModel,
    engine: &mut CrystallizationEngine,
    tokens: &[u32],
    window: usize,
    max_tokens: usize,
) -> AbResult {
    let d = model.dims().d_model;
    let limit = if max_tokens == 0 {
        tokens.len()
    } else {
        tokens.len().min(max_tokens)
    };
    let mut n = 0usize;
    let mut sum_cleared = 0.0f64;
    let mut sum_loaded = 0.0f64;
    let mut n_act = 0usize;
    let mut sum_cleared_act = 0.0f64;
    let mut sum_loaded_act = 0.0f64;

    let mut i = 0usize;
    while i + 1 < limit {
        model.reset_state();
        for _ in 0..window {
            if i + 1 >= limit {
                break;
            }
            let (fn_out, logits_cleared) = model.decode_step_capture(tokens[i]);
            let actual = tokens[i + 1] as usize;
            let nc = nll_of(&logits_cleared, actual);

            let mut hidden = Tensor::from_vec(fn_out, &[d]);
            let activated = engine.apply_modules(&mut hidden);
            let nl = if activated > 0 {
                nll_of(&model.unembed(hidden.data()), actual)
            } else {
                nc
            };

            sum_cleared += nc;
            sum_loaded += nl;
            if activated > 0 {
                n_act += 1;
                sum_cleared_act += nc;
                sum_loaded_act += nl;
            }
            n += 1;
            i += 1;
        }
    }

    AbResult {
        n,
        nll_cleared: if n > 0 { sum_cleared / n as f64 } else { 0.0 },
        nll_loaded: if n > 0 { sum_loaded / n as f64 } else { 0.0 },
        n_activated: n_act,
        nll_cleared_activated: if n_act > 0 { sum_cleared_act / n_act as f64 } else { 0.0 },
        nll_loaded_activated: if n_act > 0 { sum_loaded_act / n_act as f64 } else { 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crystal::engine::{CrystalConfig, CrystallizationEngine};
    use crate::crystal::module::CrystalModule;
    use crate::learn::backprop::{init_dense_params, DenseDims, DenseModel};
    use crate::learn::train::eval_nll;
    use crate::tensor::ternary::TernaryMatrix;
    use crate::util::seed::SeedTree;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    /// The cleared arm must reproduce the trusted `eval_nll`, and with no
    /// modules the two arms must be bit-identical (equal compute, zero drift).
    #[test]
    fn cleared_arm_matches_canonical_eval_and_zero_modules_is_identity() {
        let dims = DenseDims::mini(6);
        let params = init_dense_params(dims, 1);
        let tokens: Vec<u32> = (0..300).map(|i| (i % 6) as u32).collect();
        let canon = eval_nll(dims, &params, &tokens, 8, 0) as f64;

        let mut model = DenseModel::from_flat(dims, &params);
        let mut engine = CrystallizationEngine::new(dims.d_model, CrystalConfig::default());
        let r = ab_eval(&mut model, &mut engine, &tokens, 8, 0);

        assert!(
            (r.nll_cleared - canon).abs() < 1e-4,
            "cleared arm {} != canonical eval {}",
            r.nll_cleared,
            canon,
        );
        assert_eq!(r.nll_loaded, r.nll_cleared, "no modules → arms identical");
        assert_eq!(r.n_activated, 0);
    }

    /// An injected module must change only the loaded arm; the cleared arm is
    /// untouched and still equals the canonical eval. Proves the harness
    /// applies modules to the loaded arm alone.
    #[test]
    fn modules_touch_only_the_loaded_arm() {
        let dims = DenseDims::mini(6);
        let params = init_dense_params(dims, 2);
        let tokens: Vec<u32> = (0..300).map(|i| (i % 6) as u32).collect();
        let canon = eval_nll(dims, &params, &tokens, 8, 0) as f64;

        let mut model = DenseModel::from_flat(dims, &params);
        let mut rng = SeedTree::new(9).child("mod");
        let weight = TernaryMatrix::random(dims.d_model, dims.d_model, &mut rng);
        let module = CrystalModule {
            id: 0,
            weight,
            domain_signature: vec![1.0; dims.d_model],
            d_model: dims.d_model,
            n_source_episodes: 1,
            avg_error_before: 1.0,
            mdl_ratio: 1.0,
            sparsity: 0.0,
            activation_count: 0,
            symbolic_hint: None,
        };
        // Threshold below -1 so every token routes (cosine ∈ [-1, 1]).
        let mut engine = CrystallizationEngine::new(
            dims.d_model,
            CrystalConfig { activation_threshold: -2.0, ..CrystalConfig::default() },
        );
        engine.push_module(module);

        let r = ab_eval(&mut model, &mut engine, &tokens, 8, 0);
        assert!(
            (r.nll_cleared - canon).abs() < 1e-4,
            "cleared arm must be unchanged by modules",
        );
        assert!(r.n_activated > 0, "module should route on every token");
        assert!(
            (r.nll_loaded - r.nll_cleared).abs() > 1e-6,
            "a nonzero module must move the loaded arm",
        );
    }

    /// Capture records high-error episodes with the right shape (a random mini
    /// core is surprised almost everywhere, so a mid threshold still fires).
    #[test]
    fn capture_records_high_error_episodes() {
        let dims = DenseDims::mini(6);
        let params = init_dense_params(dims, 3);
        let mut model = DenseModel::from_flat(dims, &params);
        let tokens: Vec<u32> = (0..400).map(|i| (i % 6) as u32).collect();

        let dir = tmp("clob_bet2_capture_test");
        let _ = std::fs::remove_dir_all(&dir);
        let memory = EpisodicMemory::open(&dir, 10_000).expect("open memory");

        let n = capture_episodes(&mut model, &tokens, 8, 0.5, 100, 8, &memory);
        assert!(n > 0, "a random core should produce high-error episodes");
        assert_eq!(memory.stats().unconsumed, n);

        let eps = memory.read_unconsumed(10);
        assert!(!eps.is_empty());
        assert_eq!(eps[0].hidden_state.len(), dims.d_model, "hidden is d_model wide");
        assert!(!eps[0].future.is_empty(), "episode carries a future path");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
