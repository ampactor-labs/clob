//! Novelty detector — z-score threshold on energy scores.
//!
//! Previously used Weber's multiplicative threshold (`energy > baseline * 1.5`),
//! which only made sense when the critic was a random projection. A calibrated
//! critic's outputs concentrate around mean NLL, and the Weber rule under-fires
//! to the point of storing near-zero episodes.
//!
//! This is the NOTICE step of the crystallization loop.

/// Novelty detection result.
#[derive(Debug, Clone)]
pub struct NoveltyResult {
    /// Raw energy score.
    pub energy: f32,
    /// Whether this exceeds the adaptive threshold.
    pub is_novel: bool,
    /// Running mean.
    pub baseline: f32,
    /// Running standard deviation.
    pub stddev: f32,
    /// Z-score of this observation.
    pub z: f32,
}

/// Standalone novelty detector with a running mean / variance and z-score
/// threshold: `is_novel` iff `(energy - mean) / std >= z_threshold`.
///
/// Scale-invariant: works whether energy is bounded in [0, 10] (trained critic
/// on NLL) or spread over [0, 100] (random projection of large hidden states).
pub struct NoveltyDetector {
    /// EMA of energy scores (running mean).
    baseline: f32,
    /// EMA of squared deviations from the mean (running variance).
    variance: f32,
    /// EMA decay factor. 0.995 ≈ half-life ~140 steps.
    decay: f32,
    /// Sigma-threshold above the mean. Default 1.0 flags the right-tail ~16% of
    /// observations for Gaussian energy — enough to keep episode flow healthy.
    pub z_threshold: f32,
    /// Number of observations.
    n_observed: u64,
}

impl NoveltyDetector {
    pub fn new() -> Self {
        Self::with_threshold(1.0)
    }

    pub fn with_threshold(z_threshold: f32) -> Self {
        Self {
            baseline: 0.0,
            variance: 1.0,
            decay: 0.995,
            z_threshold,
            n_observed: 0,
        }
    }

    /// Evaluate novelty of an energy score.
    pub fn evaluate(&mut self, energy: f32) -> NoveltyResult {
        self.n_observed += 1;

        // Warmup: everything is novel for the first 10 observations while the
        // running stats bootstrap.
        if self.n_observed <= 10 {
            let alpha = 1.0 / self.n_observed as f32;
            let prev_mean = self.baseline;
            self.baseline = prev_mean + alpha * (energy - prev_mean);
            let dev = energy - prev_mean;
            self.variance = (1.0 - alpha) * self.variance + alpha * dev * dev;
            return NoveltyResult {
                energy, is_novel: true,
                baseline: self.baseline,
                stddev: self.variance.max(1e-8).sqrt(),
                z: 0.0,
            };
        }

        let std = self.variance.max(1e-8).sqrt();
        let z = (energy - self.baseline) / std;
        let is_novel = z >= self.z_threshold;

        // Update running stats (EMA).
        let prev_mean = self.baseline;
        self.baseline = self.decay * prev_mean + (1.0 - self.decay) * energy;
        let dev = energy - prev_mean;
        self.variance = self.decay * self.variance + (1.0 - self.decay) * dev * dev;

        NoveltyResult { energy, is_novel, baseline: self.baseline, stddev: std, z }
    }

    pub fn baseline(&self) -> f32 { self.baseline }
    pub fn stddev(&self) -> f32 { self.variance.max(1e-8).sqrt() }
    pub fn n_observed(&self) -> u64 { self.n_observed }
}

impl Default for NoveltyDetector {
    fn default() -> Self { Self::new() }
}
