//! Largest-Lyapunov-exponent estimation via twin trajectories.
//!
//! Benettin's method, the finite-perturbation form: run two copies of the
//! same system driven by the same input, hold their separation at a fixed
//! small norm `eps` by renormalizing after every step, and average the
//! per-step log stretch `ln(d_t / eps)`. The average converges to the
//! largest Lyapunov exponent λ₁ of the driven dynamics, in nats per step.
//!
//! Reading λ₁:
//!
//! ```text
//! λ₁ < 0   contractive — perturbations (and therefore the past) decay
//!          with e-folding horizon 1/|λ₁| steps. The state forgets.
//! λ₁ ≈ 0   near-critical — perturbations persist; the state carries
//!          its past forward. Reservoir computing lives here.
//! λ₁ > 0   expansive — perturbations grow; trajectories decorrelate
//!          and the state scrambles its own history.
//! ```
//!
//! The estimator is generic over [`TwinSystem`] so it can be verified
//! against maps with known exponents before being pointed at the model.

use rand::Rng;

/// Two synchronized trajectories of one dynamical system.
///
/// `step()` must advance both trajectories with the *same* drive input;
/// everything else about the system stays opaque to the estimator.
pub trait TwinSystem {
    /// Dimension of the flat state vector.
    fn state_dim(&self) -> usize;
    /// Advance both trajectories one step under identical drive.
    fn step(&mut self);
    /// Current state of the reference trajectory.
    fn state_a(&self) -> Vec<f32>;
    /// Current state of the perturbed trajectory.
    fn state_b(&self) -> Vec<f32>;
    /// Overwrite the perturbed trajectory's state (used for renormalization).
    fn set_state_b(&mut self, state: &[f32]);
}

/// Estimator parameters.
#[derive(Debug, Clone, Copy)]
pub struct LyapunovConfig {
    /// Perturbation norm. Must sit well above f32 rounding noise on the
    /// state's scale and well below the attractor's diameter.
    pub eps: f32,
    /// Number of measured steps (after any caller-side warmup).
    pub n_steps: usize,
}

impl Default for LyapunovConfig {
    fn default() -> Self {
        Self { eps: 1e-4, n_steps: 4096 }
    }
}

/// Result of a λ₁ estimation run.
#[derive(Debug, Clone)]
pub struct LyapunovEstimate {
    /// Mean log stretch per step — the λ₁ estimate, nats/step.
    pub lambda1: f64,
    /// λ₁ over the first half of the run (stationarity check).
    pub lambda1_first_half: f64,
    /// λ₁ over the second half of the run.
    pub lambda1_second_half: f64,
    /// Steps measured.
    pub n_steps: usize,
    /// State dimension.
    pub state_dim: usize,
    /// Per-step `ln(d_t / eps)` series, for plotting or windowed stats.
    pub log_stretch: Vec<f64>,
}

/// Distance floor: a twin pair that lands exactly on the reference
/// trajectory reads as contraction at least this severe rather than
/// `ln(0)`. At f32 precision, exact collapse means the true separation
/// is below representable resolution anyway.
const D_FLOOR: f64 = 1e-30;

/// Estimate λ₁ of `sys` by the twin-trajectory method.
///
/// The caller is responsible for warmup: drive the system to its working
/// region *before* calling, with both trajectories identical. This
/// function perturbs trajectory B by `eps` in a random direction, then
/// measures `cfg.n_steps` steps with per-step renormalization.
pub fn estimate<S: TwinSystem>(
    sys: &mut S,
    cfg: &LyapunovConfig,
    rng: &mut impl Rng,
) -> LyapunovEstimate {
    let dim = sys.state_dim();
    assert!(dim > 0, "cannot estimate lyapunov of a zero-dim system");
    assert!(cfg.eps > 0.0, "eps must be positive");
    assert!(cfg.n_steps > 0, "n_steps must be positive");

    // Perturb B off A by eps in a uniformly random direction.
    let a = sys.state_a();
    let mut b = a.clone();
    add_perturbation(&mut b, cfg.eps, rng);
    sys.set_state_b(&b);

    let eps = cfg.eps as f64;
    let mut log_stretch = Vec::with_capacity(cfg.n_steps);

    for _ in 0..cfg.n_steps {
        sys.step();
        let a = sys.state_a();
        let b = sys.state_b();

        let d = l2_distance(&a, &b).max(D_FLOOR);
        log_stretch.push((d / eps).ln());

        // Renormalize: place B back at distance eps from A, along the
        // current separation direction (or a fresh random direction if
        // the twins collapsed to numerical identity).
        let mut b_new = a.clone();
        if d > D_FLOOR {
            let scale = (eps / d) as f32;
            for (bn, (av, bv)) in b_new.iter_mut().zip(a.iter().zip(b.iter())) {
                *bn = av + scale * (bv - av);
            }
        } else {
            add_perturbation(&mut b_new, cfg.eps, rng);
        }
        sys.set_state_b(&b_new);
    }

    let half = cfg.n_steps / 2;
    let mean = |s: &[f64]| -> f64 {
        if s.is_empty() { 0.0 } else { s.iter().sum::<f64>() / s.len() as f64 }
    };

    LyapunovEstimate {
        lambda1: mean(&log_stretch),
        lambda1_first_half: mean(&log_stretch[..half]),
        lambda1_second_half: mean(&log_stretch[half..]),
        n_steps: cfg.n_steps,
        state_dim: dim,
        log_stretch,
    }
}

/// Add a random perturbation of norm `eps` to `state`.
fn add_perturbation(state: &mut [f32], eps: f32, rng: &mut impl Rng) {
    let mut dir: Vec<f32> = (0..state.len()).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let norm = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 {
        // Degenerate draw; fall back to the first axis.
        dir.iter_mut().for_each(|x| *x = 0.0);
        dir[0] = 1.0;
        state[0] += eps;
        return;
    }
    let scale = eps / norm;
    for (s, d) in state.iter_mut().zip(dir.iter()) {
        *s += scale * d;
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x - *y) as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Twin pair over a pure map (no external drive), for verifying the
    /// estimator against systems with known exponents.
    struct MapTwin<F: Fn(&mut [f32])> {
        a: Vec<f32>,
        b: Vec<f32>,
        f: F,
    }

    impl<F: Fn(&mut [f32])> TwinSystem for MapTwin<F> {
        fn state_dim(&self) -> usize { self.a.len() }
        fn step(&mut self) {
            (self.f)(&mut self.a);
            (self.f)(&mut self.b);
        }
        fn state_a(&self) -> Vec<f32> { self.a.clone() }
        fn state_b(&self) -> Vec<f32> { self.b.clone() }
        fn set_state_b(&mut self, s: &[f32]) { self.b.copy_from_slice(s); }
    }

    #[test]
    fn uniform_contraction_gives_log_half() {
        // x ← 0.5·x contracts every direction by exactly 0.5 per step,
        // so λ₁ = ln(0.5) to within f32 rounding.
        let mut sys = MapTwin {
            a: vec![0.3, -0.2, 0.7, 0.1],
            b: vec![0.3, -0.2, 0.7, 0.1],
            f: |s: &mut [f32]| s.iter_mut().for_each(|x| *x *= 0.5),
        };
        let mut rng = StdRng::seed_from_u64(7);
        let est = estimate(&mut sys, &LyapunovConfig { eps: 1e-4, n_steps: 64 }, &mut rng);
        let expected = (0.5f64).ln();
        assert!(
            (est.lambda1 - expected).abs() < 1e-3,
            "lambda1 = {}, expected {}", est.lambda1, expected,
        );
    }

    #[test]
    fn expanding_sawtooth_gives_log_slope() {
        // x ← frac(1.9·x + 0.3) has derivative 1.9 everywhere away from the
        // wrap, so λ₁ = ln(1.9). The slope is deliberately non-dyadic: a
        // slope of exactly 2 degenerates in binary floating point (each
        // step shifts mantissa bits out until the orbit collapses).
        let mut sys = MapTwin {
            a: vec![0.234_567],
            b: vec![0.234_567],
            f: |s: &mut [f32]| {
                let y = 1.9 * s[0] + 0.3;
                s[0] = y - y.floor();
            },
        };
        let mut rng = StdRng::seed_from_u64(11);
        let est = estimate(&mut sys, &LyapunovConfig { eps: 1e-5, n_steps: 500 }, &mut rng);
        let expected = (1.9f64).ln();
        let rel_err = (est.lambda1 - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "lambda1 = {}, expected {} (rel err {})", est.lambda1, expected, rel_err,
        );
    }

    #[test]
    fn rotation_gives_zero() {
        // A rigid rotation preserves distances exactly: λ₁ = 0.
        let (c, s) = (1.0f32.cos(), 1.0f32.sin());
        let mut sys = MapTwin {
            a: vec![0.6, 0.8],
            b: vec![0.6, 0.8],
            f: move |v: &mut [f32]| {
                let (x, y) = (v[0], v[1]);
                v[0] = c * x - s * y;
                v[1] = s * x + c * y;
            },
        };
        let mut rng = StdRng::seed_from_u64(13);
        let est = estimate(&mut sys, &LyapunovConfig { eps: 1e-4, n_steps: 256 }, &mut rng);
        assert!(
            est.lambda1.abs() < 1e-3,
            "lambda1 = {}, expected ~0", est.lambda1,
        );
    }

    #[test]
    fn collapse_to_identity_stays_finite() {
        // x ← 0 collapses the twins to numerical identity on the first
        // step. The estimate must stay finite (floored), not -inf.
        let mut sys = MapTwin {
            a: vec![0.5, 0.5],
            b: vec![0.5, 0.5],
            f: |s: &mut [f32]| s.iter_mut().for_each(|x| *x = 0.0),
        };
        let mut rng = StdRng::seed_from_u64(17);
        let est = estimate(&mut sys, &LyapunovConfig { eps: 1e-4, n_steps: 16 }, &mut rng);
        assert!(est.lambda1.is_finite());
        assert!(est.lambda1 < -10.0, "total collapse should read as strong contraction");
    }
}
