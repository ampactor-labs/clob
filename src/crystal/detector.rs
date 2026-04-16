//! Novelty detector — wraps the energy critic with Weber-law adaptive threshold.
//!
//! This is the NOTICE step of the crystallization loop.

/// Novelty detection result.
#[derive(Debug, Clone)]
pub struct NoveltyResult {
    /// Raw energy score.
    pub energy: f32,
    /// Whether this exceeds the adaptive threshold.
    pub is_novel: bool,
    /// Current baseline.
    pub baseline: f32,
    /// Current threshold.
    pub threshold: f32,
}

/// Standalone novelty detector with adaptive threshold.
///
/// Uses Weber's Law: the just-noticeable difference is proportional
/// to the baseline intensity. As the system learns more, fewer things
/// surprise it.
pub struct NoveltyDetector {
    /// EMA of energy scores.
    baseline: f32,
    /// EMA decay factor.
    decay: f32,
    /// Weber fraction (threshold = baseline * (1 + weber_fraction)).
    weber_fraction: f32,
    /// Number of observations.
    n_observed: u64,
}

impl NoveltyDetector {
    pub fn new() -> Self {
        Self {
            baseline: 0.0,
            decay: 0.995,
            weber_fraction: 0.5, // 50% above baseline = novel
            n_observed: 0,
        }
    }

    /// Evaluate novelty of an energy score.
    pub fn evaluate(&mut self, energy: f32) -> NoveltyResult {
        self.n_observed += 1;

        if self.n_observed <= 10 {
            self.baseline = if self.n_observed == 1 {
                energy
            } else {
                self.baseline * 0.9 + energy * 0.1
            };
            // Everything is novel during warmup
            return NoveltyResult {
                energy,
                is_novel: true,
                baseline: self.baseline,
                threshold: 0.0,
            };
        }

        let threshold = self.baseline * (1.0 + self.weber_fraction);
        let is_novel = energy > threshold;

        // Update baseline
        self.baseline = self.decay * self.baseline + (1.0 - self.decay) * energy;

        NoveltyResult {
            energy,
            is_novel,
            baseline: self.baseline,
            threshold,
        }
    }

    pub fn baseline(&self) -> f32 { self.baseline }
    pub fn n_observed(&self) -> u64 { self.n_observed }
}

impl Default for NoveltyDetector {
    fn default() -> Self { Self::new() }
}
