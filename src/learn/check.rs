//! Finite-difference gradient checks for hand-written backwards.
//!
//! The checker is for differentiable surrogate forwards. For ternary layers,
//! finite-difference the f32 latent-weight surrogate, then separately verify
//! the straight-through bridge into deployed ternary weights.

/// Configuration for central-difference gradient checks.
#[derive(Debug, Clone, Copy)]
pub struct GradCheckConfig {
    /// Perturbation used for central differences.
    pub epsilon: f32,
    /// Maximum allowed relative error.
    pub tolerance: f64,
    /// Denominator floor for relative error when both gradients are tiny.
    pub denom_floor: f64,
}

impl Default for GradCheckConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-3,
            tolerance: 1e-3,
            denom_floor: 1e-8,
        }
    }
}

/// Summary of one gradient-check run.
#[derive(Debug, Clone)]
pub struct GradCheckResult {
    pub n_params: usize,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub worst_index: usize,
    pub analytical_at_worst: f64,
    pub numerical_at_worst: f64,
}

impl GradCheckResult {
    pub fn within_tolerance(&self, cfg: GradCheckConfig) -> bool {
        self.max_rel_error <= cfg.tolerance
    }
}

/// Compare an analytic gradient against central finite differences.
///
/// `loss(params)` must be deterministic and side-effect free for the same
/// parameter slice. `analytic(params, out)` writes `d loss / d params` into
/// `out`.
pub fn check_gradient<L, A>(
    params: &[f32],
    mut loss: L,
    mut analytic: A,
    cfg: GradCheckConfig,
) -> GradCheckResult
where
    L: FnMut(&[f32]) -> f64,
    A: FnMut(&[f32], &mut [f64]),
{
    assert!(
        !params.is_empty(),
        "gradient check needs at least one parameter"
    );
    assert!(cfg.epsilon > 0.0, "epsilon must be positive");
    assert!(cfg.denom_floor > 0.0, "denom_floor must be positive");

    let mut analytical = vec![0.0f64; params.len()];
    analytic(params, &mut analytical);

    let mut plus = params.to_vec();
    let mut minus = params.to_vec();

    let mut result = GradCheckResult {
        n_params: params.len(),
        max_abs_error: 0.0,
        max_rel_error: 0.0,
        worst_index: 0,
        analytical_at_worst: 0.0,
        numerical_at_worst: 0.0,
    };

    let denom = 2.0 * cfg.epsilon as f64;
    for i in 0..params.len() {
        plus[i] = params[i] + cfg.epsilon;
        minus[i] = params[i] - cfg.epsilon;
        let numerical = (loss(&plus) - loss(&minus)) / denom;
        plus[i] = params[i];
        minus[i] = params[i];

        let analytical_i = analytical[i];
        let abs_error = (analytical_i - numerical).abs();
        let scale = analytical_i.abs().max(numerical.abs()).max(cfg.denom_floor);
        let rel_error = abs_error / scale;

        if rel_error > result.max_rel_error {
            result.max_abs_error = abs_error;
            result.max_rel_error = rel_error;
            result.worst_index = i;
            result.analytical_at_worst = analytical_i;
            result.numerical_at_worst = numerical;
        }
    }

    result
}

/// Panicking convenience wrapper for tests.
pub fn assert_gradient_close<L, A>(
    params: &[f32],
    loss: L,
    analytic: A,
    cfg: GradCheckConfig,
) -> GradCheckResult
where
    L: FnMut(&[f32]) -> f64,
    A: FnMut(&[f32], &mut [f64]),
{
    let result = check_gradient(params, loss, analytic, cfg);
    assert!(
        result.within_tolerance(cfg),
        "gradient check failed: max_rel_error={:.6e} max_abs_error={:.6e} \
         worst_index={} analytical={:.6e} numerical={:.6e}",
        result.max_rel_error,
        result.max_abs_error,
        result.worst_index,
        result.analytical_at_worst,
        result.numerical_at_worst,
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_accepts_correct_quadratic_gradient() {
        let params = vec![0.5, -1.25, 2.0, 0.1];
        let cfg = GradCheckConfig::default();
        let result = assert_gradient_close(
            &params,
            |p| p.iter().map(|&x| 0.5 * (x as f64) * (x as f64)).sum(),
            |p, out| {
                for (o, &x) in out.iter_mut().zip(p.iter()) {
                    *o = x as f64;
                }
            },
            cfg,
        );
        assert_eq!(result.n_params, params.len());
    }

    #[test]
    fn checker_rejects_bad_gradient() {
        let params = vec![0.5, -1.25, 2.0, 0.1];
        let cfg = GradCheckConfig::default();
        let result = check_gradient(
            &params,
            |p| p.iter().map(|&x| 0.5 * (x as f64) * (x as f64)).sum(),
            |_p, out| {
                for o in out {
                    *o = 0.0;
                }
            },
            cfg,
        );
        assert!(!result.within_tolerance(cfg));
        assert!(result.max_rel_error > 0.9);
    }

    #[test]
    fn checker_handles_cubic_gradient() {
        let params = vec![-0.8, -0.2, 0.3, 1.1];
        assert_gradient_close(
            &params,
            |p| p.iter().map(|&x| (x as f64).powi(3)).sum(),
            |p, out| {
                for (o, &x) in out.iter_mut().zip(p.iter()) {
                    *o = 3.0 * (x as f64) * (x as f64);
                }
            },
            GradCheckConfig {
                epsilon: 1e-3,
                tolerance: 2e-3,
                denom_floor: 1e-8,
            },
        );
    }
}
