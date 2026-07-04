//! Dynamics instrumentation — measuring the regime the core runs in.
//!
//! The recurrent core is a driven dynamical system: tokens are the drive,
//! the block states are the trajectory. Everything the model knows about
//! its past is encoded in where that trajectory currently sits, so the
//! *regime* of the dynamics — contractive, near-critical, expansive —
//! decides how long the past can matter at all. A core whose largest
//! Lyapunov exponent is strongly negative forgets its input in `1/|λ₁|`
//! tokens no matter how good its weights are; a core near zero can carry
//! context indefinitely.
//!
//! This module makes the regime a measured quantity, the same way
//! `metrics` made J/nat one. See [`lyapunov`] for the estimator and
//! `clob regime` for the CLI instrument.

pub mod lyapunov;

use crate::model::stack::CoreModel;
use lyapunov::TwinSystem;
use serde::Serialize;

/// Twin CoreModel trajectories driven by the same token stream.
///
/// Both models must have identical weights (load the same file twice) and
/// start from reset state; `warmup` then drives them in lockstep so their
/// trajectories are identical when the estimator perturbs one. The token
/// stream wraps if the run is longer than the corpus.
pub struct ModelTwin {
    a: CoreModel,
    b: CoreModel,
    tokens: Vec<u32>,
    pos: usize,
}

impl ModelTwin {
    /// Build a twin pair. Panics on an empty token stream or mismatched
    /// architectures.
    pub fn new(mut a: CoreModel, mut b: CoreModel, tokens: Vec<u32>) -> Self {
        assert!(!tokens.is_empty(), "ModelTwin needs a non-empty token stream");
        assert_eq!(
            a.config.config_hash(), b.config.config_hash(),
            "twin models must share an architecture",
        );
        a.reset_state();
        b.reset_state();
        Self { a, b, tokens, pos: 0 }
    }

    /// Drive both trajectories `n` steps to reach the working region of
    /// state space before measurement begins.
    pub fn warmup(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }
}

impl TwinSystem for ModelTwin {
    fn state_dim(&self) -> usize {
        self.a.state_dim()
    }

    fn step(&mut self) {
        let token = self.tokens[self.pos];
        self.pos = (self.pos + 1) % self.tokens.len();
        let _ = self.a.decode_step(token);
        let _ = self.b.decode_step(token);
    }

    fn state_a(&self) -> Vec<f32> {
        self.a.export_state()
    }

    fn state_b(&self) -> Vec<f32> {
        self.b.export_state()
    }

    fn set_state_b(&mut self, state: &[f32]) {
        self.b.import_state(state);
    }
}

/// Half-width of the near-critical band, nats/token. Descriptive binning
/// for reports — the pre-registered Bet 7 thresholds live in ATTRACTOR.md
/// and are about correlation across checkpoints, not these bins.
pub const EDGE_BAND: f64 = 0.05;

/// A regime measurement, serialized as TOML by `clob regime`.
#[derive(Debug, Clone, Serialize)]
pub struct RegimeReport {
    /// λ₁ in nats per token. The one number.
    pub lambda1_nats_per_token: f64,
    /// λ₁ over the first half of measured steps (stationarity check —
    /// halves that disagree badly mean the run was too short or the
    /// corpus drive is non-stationary at this scale).
    pub lambda1_first_half: f64,
    /// λ₁ over the second half.
    pub lambda1_second_half: f64,
    /// e-folding horizon of a state perturbation, `1/|λ₁|` tokens.
    /// Present only in the contractive regime, where it is the hard
    /// ceiling on how far back the state can remember.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_horizon_tokens: Option<f64>,
    /// "contractive" | "near-critical" | "expansive" (±EDGE_BAND bins).
    pub regime: String,
    /// Flat recurrent-state dimension measured.
    pub state_dim: usize,
    /// Measured steps (after warmup).
    pub n_steps: usize,
    /// Warmup tokens driven before perturbation.
    pub warmup_tokens: usize,
    /// Perturbation norm.
    pub eps: f32,
    /// Root seed (perturbation directions derive from it).
    pub seed: u64,
}

impl RegimeReport {
    /// Assemble a report from an estimate plus run parameters.
    pub fn from_estimate(
        est: &lyapunov::LyapunovEstimate,
        warmup_tokens: usize,
        eps: f32,
        seed: u64,
    ) -> Self {
        let l = est.lambda1;
        let regime = if l < -EDGE_BAND {
            "contractive"
        } else if l > EDGE_BAND {
            "expansive"
        } else {
            "near-critical"
        };
        Self {
            lambda1_nats_per_token: l,
            lambda1_first_half: est.lambda1_first_half,
            lambda1_second_half: est.lambda1_second_half,
            memory_horizon_tokens: if l < 0.0 { Some(1.0 / l.abs()) } else { None },
            regime: regime.to_string(),
            state_dim: est.state_dim,
            n_steps: est.n_steps,
            warmup_tokens,
            eps,
            seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::KernelConfig;
    use lyapunov::{estimate, LyapunovConfig};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn tiny_model(seed: u64) -> CoreModel {
        let mut rng = StdRng::seed_from_u64(seed);
        CoreModel::random(KernelConfig::tiny(), &mut rng)
    }

    #[test]
    fn state_roundtrip_is_exact_and_complete() {
        // Export/import must capture *all* evolving state: drive a model,
        // snapshot, keep driving and record logits, restore the snapshot,
        // re-drive the same tokens, and demand bit-identical logits. If any
        // state escaped the snapshot, the replay diverges.
        let mut model = tiny_model(42);
        let drive: Vec<u32> = (0..40u32).map(|i| (i * 7 + 3) % 256).collect();

        model.reset_state();
        for &t in &drive[..20] {
            let _ = model.decode_step(t);
        }
        let snap = model.export_state();
        assert_eq!(snap.len(), model.state_dim());

        let first: Vec<Vec<f32>> = drive[20..]
            .iter()
            .map(|&t| model.decode_step(t).data().to_vec())
            .collect();

        model.import_state(&snap);
        let second: Vec<Vec<f32>> = drive[20..]
            .iter()
            .map(|&t| model.decode_step(t).data().to_vec())
            .collect();

        assert_eq!(first, second, "replay from imported state must be bit-identical");
    }

    #[test]
    fn state_dim_counts_ssm_and_dense_gru() {
        let cfg = KernelConfig::tiny();
        let model = tiny_model(1);
        // Per block: SSM carries n_heads×d_state. Dense blocks add a
        // d_model MLGRU state; MoE blocks don't.
        let ssm = cfg.n_heads * cfg.d_state;
        let expected: usize = (0..cfg.n_layers)
            .map(|i| ssm + if cfg.is_moe_layer(i) { 0 } else { cfg.d_model })
            .sum();
        assert_eq!(model.state_dim(), expected);
    }

    #[test]
    fn model_twin_estimates_finite_lambda() {
        // Identical random models, identical drive: λ₁ must come out
        // finite, and the twins must actually track (no NaN separation).
        let drive: Vec<u32> = (0..128u32).map(|i| (i * 31 + 5) % 256).collect();
        let mut twin = ModelTwin::new(tiny_model(7), tiny_model(7), drive);
        twin.warmup(32);
        let mut rng = StdRng::seed_from_u64(99);
        let est = estimate(&mut twin, &LyapunovConfig { eps: 1e-4, n_steps: 64 }, &mut rng);
        assert!(est.lambda1.is_finite(), "lambda1 = {}", est.lambda1);
        assert!(est.log_stretch.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn regime_report_bins_and_horizon() {
        let est = lyapunov::LyapunovEstimate {
            lambda1: -0.25,
            lambda1_first_half: -0.24,
            lambda1_second_half: -0.26,
            n_steps: 100,
            state_dim: 8,
            log_stretch: vec![],
        };
        let report = RegimeReport::from_estimate(&est, 64, 1e-4, 1);
        assert_eq!(report.regime, "contractive");
        let horizon = report.memory_horizon_tokens.unwrap();
        assert!((horizon - 4.0).abs() < 1e-9, "1/0.25 = 4 tokens, got {}", horizon);

        let near = lyapunov::LyapunovEstimate { lambda1: 0.01, ..est.clone() };
        assert_eq!(RegimeReport::from_estimate(&near, 0, 1e-4, 1).regime, "near-critical");
        assert!(RegimeReport::from_estimate(&near, 0, 1e-4, 1).memory_horizon_tokens.is_none());

        let expansive = lyapunov::LyapunovEstimate { lambda1: 0.3, ..est };
        assert_eq!(RegimeReport::from_estimate(&expansive, 0, 1e-4, 1).regime, "expansive");
    }
}
