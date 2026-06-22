//! Dense f32 linear readout fitted over frozen-core hidden states — the
//! "reservoir probe".
//!
//! clob's core is ternary, recurrent, and (today) frozen at random init; its
//! readout is the *tied* embedding table, also random. The open fork in
//! `IF_FOUND.md` is whether that frozen random core is a usable *reservoir*:
//! does its hidden state expose linearly-readable contextual structure, the
//! way an echo-state network's reservoir does?
//!
//! Reservoir computing requires a *trainable* readout — which clob does not
//! have, because embed and unembed are one tied frozen table. This module
//! supplies that missing readout as a throwaway diagnostic: fit a dense f32
//! `logits = W·h + b` over the frozen core's hidden states by multinomial
//! logistic regression (convex), warm-started at the unigram marginal
//! (`W = 0`, `b = log p_unigram`). Its held-out NLL is then compared against
//! that same marginal. The gap is exactly the contextual information the
//! frozen core linearly carries:
//!
//! * gap ≈ 0  ⇒ the trained readout recovers only the marginal; the frozen
//!   random core carries no usable context. **Reservoir dead** — Path B
//!   (train the core) is mandatory.
//! * real gap ⇒ the random projection exposes linearly-readable structure.
//!   **Reservoir alive** — Path A (reservoir + crystallization) is viable,
//!   and the crystallizer finally has signal to compress.
//!
//! Nothing here touches the on-disk format or the tied table; it reads
//! hidden states and reports numbers.

use rayon::prelude::*;

/// Momentum-SGD hyperparameters for the readout fit.
///
/// Plain SGD is deliberate, not lazy: this is a *flat-objective* diagnostic.
/// When the reservoir is dead the true gradient is ≈0, and an adaptive
/// optimizer (Adam) would divide that ≈0 gradient by an ≈0 second moment,
/// normalizing pure noise into O(1) steps and random-walking *away* from the
/// optimum — manufacturing a false "divergence" precisely in the dead case.
/// SGD's step scales with the gradient: dead ⇒ tiny steps ⇒ stays at the
/// marginal; alive ⇒ real gradient ⇒ descends below it. That is the property
/// that makes the verdict trustworthy.
#[derive(Clone, Copy, Debug)]
pub struct FitConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    /// Multiplicative lr decay applied once per epoch (1.0 = none).
    pub lr_decay: f32,
    /// Freeze `W = 0` and train only the bias. This yields the proper
    /// "no context" null: a readout that can re-fit the token marginal under
    /// the identical objective but cannot use the core's hidden state. The
    /// gap between a full fit and this null is the contextual contribution.
    pub freeze_w: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            batch_size: 512,
            lr: 0.5,
            weight_decay: 1e-5,
            momentum: 0.9,
            lr_decay: 0.7,
            freeze_w: false,
        }
    }
}

/// Per-epoch training trace.
#[derive(Clone, Copy, Debug)]
pub struct EpochStat {
    pub epoch: usize,
    pub train_nll: f64,
    pub holdout_nll: f64,
}

/// Dense f32 linear readout: `logits[v] = b[v] + dot(W_row_v, h)`.
///
/// `w` is row-major `[vocab × d]`; `b` is `[vocab]`. Carries its own AdamW
/// moment buffers so a fit can run in a single owned object.
pub struct LinearReadout {
    pub vocab: usize,
    pub d: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    // Momentum-SGD velocity buffers.
    vel_w: Vec<f32>,
    vel_b: Vec<f32>,
}

impl LinearReadout {
    /// `W = 0`, `b = bias` (length `vocab`). Warm-starting `b` at the unigram
    /// log-marginal makes the fit begin *exactly* at the marginal baseline,
    /// so any held-out improvement is unambiguously the core's contribution.
    pub fn with_bias(vocab: usize, d: usize, bias: Vec<f32>) -> Self {
        assert_eq!(bias.len(), vocab, "bias must have length vocab");
        Self {
            vocab,
            d,
            w: vec![0.0; vocab * d],
            b: bias,
            vel_w: vec![0.0; vocab * d],
            vel_b: vec![0.0; vocab],
        }
    }

    /// Stable softmax NLL of `target` given features `h`. Used for eval.
    /// Single pass: materialize logits into `buf`, then max + log-sum-exp.
    fn example_nll_buf(&self, h: &[f32], target: u32, buf: &mut [f32]) -> f64 {
        let t = target as usize;
        let mut maxl = f32::NEG_INFINITY;
        for v in 0..self.vocab {
            let wr = &self.w[v * self.d..v * self.d + self.d];
            let mut s = self.b[v];
            for k in 0..self.d {
                s += wr[k] * h[k];
            }
            buf[v] = s;
            if s > maxl {
                maxl = s;
            }
        }
        let mut sum = 0.0f32;
        for v in 0..self.vocab {
            sum += (buf[v] - maxl).exp();
        }
        // -log p(target) = -(logit_t - max) + log sum.
        let logp = (buf[t] - maxl) as f64 - (sum as f64).ln();
        -logp
    }

    /// Mean NLL over an entire (features, targets) set. `feats` is row-major
    /// `[n × d]`; `targets` is `[n]`.
    pub fn eval(&self, feats: &[f32], targets: &[u32]) -> f64 {
        let n = targets.len();
        if n == 0 {
            return 0.0;
        }
        let total: f64 = (0..n)
            .into_par_iter()
            .map_init(
                || vec![0.0f32; self.vocab],
                |buf, e| self.example_nll_buf(&feats[e * self.d..e * self.d + self.d], targets[e], buf),
            )
            .sum();
        total / n as f64
    }

    /// One momentum-SGD minibatch step over the indices in `batch` at
    /// learning rate `lr`. Returns the batch mean NLL (pre-update). `feats`
    /// row-major `[n × d]`.
    fn step(&mut self, feats: &[f32], targets: &[u32], batch: &[usize], lr: f32, cfg: &FitConfig) -> f64 {
        let bsz = batch.len();
        let vocab = self.vocab;
        let d = self.d;

        // ── Forward: residuals R = softmax(logits); R[target] -= 1 ──────────
        // Flat [bsz × vocab]. Loss accumulated in the same parallel pass.
        let mut resid = vec![0.0f32; bsz * vocab];
        let loss: f64 = resid
            .par_chunks_mut(vocab)
            .zip(batch.par_iter())
            .map(|(row, &e)| {
                let h = &feats[e * d..e * d + d];
                let mut maxl = f32::NEG_INFINITY;
                for v in 0..vocab {
                    let wr = &self.w[v * d..v * d + d];
                    let mut s = self.b[v];
                    for k in 0..d {
                        s += wr[k] * h[k];
                    }
                    row[v] = s;
                    if s > maxl {
                        maxl = s;
                    }
                }
                let mut sum = 0.0f32;
                for v in 0..vocab {
                    let p = (row[v] - maxl).exp();
                    row[v] = p;
                    sum += p;
                }
                let inv = 1.0 / sum;
                for v in 0..vocab {
                    row[v] *= inv;
                }
                let tgt = targets[e] as usize;
                let p_t = row[tgt].max(1e-12);
                row[tgt] -= 1.0;
                -(p_t as f64).ln()
            })
            .sum();

        // ── Momentum-SGD update ─────────────────────────────────────────────
        let inv_b = 1.0 / bsz as f32;
        let (mom, wd) = (cfg.momentum, cfg.weight_decay);

        // Weight rows (parallel, disjoint chunks). L2 weight decay on W only.
        // Skipped entirely for the bias-only null (W stays 0).
        if !cfg.freeze_w {
        self.w
            .par_chunks_mut(d)
            .zip(self.vel_w.par_chunks_mut(d))
            .enumerate()
            .for_each(|(v, (wr, velr))| {
                // Accumulate grad row: g[k] = mean_e R[e][v] * h_e[k].
                let mut g = vec![0.0f32; d];
                for (bi, &e) in batch.iter().enumerate() {
                    let r = resid[bi * vocab + v];
                    if r == 0.0 {
                        continue;
                    }
                    let h = &feats[e * d..e * d + d];
                    for k in 0..d {
                        g[k] += r * h[k];
                    }
                }
                for k in 0..d {
                    let grad = g[k] * inv_b + wd * wr[k];
                    velr[k] = mom * velr[k] + grad;
                    wr[k] -= lr * velr[k];
                }
            });
        }

        // Bias (parallel over vocab; no weight decay on bias).
        self.b
            .par_iter_mut()
            .zip(self.vel_b.par_iter_mut())
            .enumerate()
            .for_each(|(v, (bv, velv))| {
                let mut gb = 0.0f32;
                for bi in 0..bsz {
                    gb += resid[bi * vocab + v];
                }
                let grad = gb * inv_b;
                *velv = mom * *velv + grad;
                *bv -= lr * *velv;
            });

        loss / bsz as f64
    }

    /// Fit the readout. `feats`/`targets` are the training set (row-major
    /// `[n × d]`), `hf`/`ht` the held-out set. `shuffle` supplies a
    /// deterministic permutation per epoch (Fisher–Yates via the caller's
    /// seeded RNG). Returns the per-epoch trace.
    pub fn fit(
        &mut self,
        feats: &[f32],
        targets: &[u32],
        hf: &[f32],
        ht: &[u32],
        cfg: &FitConfig,
        rng: &mut impl rand::Rng,
    ) -> Vec<EpochStat> {
        let n = targets.len();
        let mut order: Vec<usize> = (0..n).collect();
        let mut trace = Vec::with_capacity(cfg.epochs);

        // Snapshot the best-held-out weights so the readout left behind (and
        // any persisted artifact) is the best epoch, not merely the last —
        // which matters whenever a later epoch overfits past the optimum.
        let mut best_holdout = f64::INFINITY;
        let mut best_w = self.w.clone();
        let mut best_b = self.b.clone();

        let mut lr = cfg.lr;
        for epoch in 0..cfg.epochs {
            // Fisher–Yates shuffle for this epoch.
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                order.swap(i, j);
            }
            let mut epoch_loss = 0.0f64;
            let mut nb = 0usize;
            for chunk in order.chunks(cfg.batch_size) {
                epoch_loss += self.step(feats, targets, chunk, lr, cfg);
                nb += 1;
            }
            lr *= cfg.lr_decay;
            let train_nll = if nb > 0 { epoch_loss / nb as f64 } else { 0.0 };
            let holdout_nll = self.eval(hf, ht);
            if holdout_nll < best_holdout {
                best_holdout = holdout_nll;
                best_w.copy_from_slice(&self.w);
                best_b.copy_from_slice(&self.b);
            }
            trace.push(EpochStat {
                epoch: epoch + 1,
                train_nll,
                holdout_nll,
            });
        }
        // Leave the readout at its best-held-out epoch.
        self.w = best_w;
        self.b = best_b;
        trace
    }
}

/// Empirical unigram log-marginal with add-one (Laplace) smoothing over
/// `vocab` classes, fit on `targets`. Returns `b[v] = ln p_unigram(v)`,
/// suitable as a warm-start bias and as the marginal baseline.
pub fn unigram_log_bias(targets: &[u32], vocab: usize) -> Vec<f32> {
    let mut counts = vec![1.0f64; vocab]; // Laplace add-one
    let mut total = vocab as f64;
    for &t in targets {
        let i = t as usize;
        if i < vocab {
            counts[i] += 1.0;
            total += 1.0;
        }
    }
    counts
        .iter()
        .map(|&c| (c / total).ln() as f32)
        .collect()
}

/// Mean NLL of `targets` under a fixed log-probability vector `log_p`
/// (length `vocab`). Used to score the unigram marginal on held-out data.
pub fn fixed_logprob_nll(log_p: &[f32], targets: &[u32]) -> f64 {
    let n = targets.len();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0f64;
    for &t in targets {
        let i = t as usize;
        let lp = if i < log_p.len() { log_p[i] as f64 } else { f64::NEG_INFINITY };
        total += -lp;
    }
    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    // A trained readout on linearly-separable features must beat the unigram
    // marginal on held-out data; a readout on pure-noise features must not.
    #[test]
    fn readout_beats_marginal_on_separable_features() {
        let d = 4;
        let vocab = 3;
        let n = 1200;
        let mut rng = StdRng::seed_from_u64(42);

        // Class c has its c-th feature elevated; rest noise. Linearly
        // separable, so a linear readout should drive NLL well below ln(3).
        let mut feats = vec![0.0f32; n * d];
        let mut targets = vec![0u32; n];
        for e in 0..n {
            let c = (e % vocab) as u32;
            for k in 0..d {
                feats[e * d + k] = (rng.gen::<f32>() - 0.5) * 0.3;
            }
            feats[e * d + c as usize] += 2.0;
            targets[e] = c;
        }
        // Hold out the last 200.
        let split = 1000 * d;
        let (tf, hf) = feats.split_at(split);
        let (tt, ht) = targets.split_at(1000);

        let bias = unigram_log_bias(tt, vocab);
        let marginal = fixed_logprob_nll(&bias, ht);

        let mut ro = LinearReadout::with_bias(vocab, d, bias);
        let cfg = FitConfig {
            epochs: 12,
            batch_size: 64,
            lr: 0.3,
            lr_decay: 1.0,
            ..FitConfig::default()
        };
        let mut frng = StdRng::seed_from_u64(7);
        let trace = ro.fit(tf, tt, hf, ht, &cfg, &mut frng);
        let final_holdout = trace.iter().map(|s| s.holdout_nll).fold(f64::INFINITY, f64::min);

        assert!(
            final_holdout < marginal - 0.2,
            "separable: trained holdout NLL {:.4} should beat marginal {:.4}",
            final_holdout,
            marginal
        );
        // ln(3) ≈ 1.0986; a clean linear separation should get well under it.
        assert!(final_holdout < 0.5, "holdout NLL {:.4} too high", final_holdout);
    }

    #[test]
    fn readout_matches_marginal_on_noise_features() {
        let d = 4;
        let vocab = 3;
        let n = 1200;
        let mut rng = StdRng::seed_from_u64(99);

        // Features carry no information about the (skewed) target.
        let mut feats = vec![0.0f32; n * d];
        let mut targets = vec![0u32; n];
        for e in 0..n {
            for k in 0..d {
                feats[e * d + k] = (rng.gen::<f32>() - 0.5) * 0.3;
            }
            // Skewed marginal: class 0 twice as likely.
            targets[e] = if rng.gen::<f32>() < 0.5 { 0 } else { 1 + (e as u32 % 2) };
        }
        let split = 1000 * d;
        let (tf, hf) = feats.split_at(split);
        let (tt, ht) = targets.split_at(1000);

        let bias = unigram_log_bias(tt, vocab);
        let marginal = fixed_logprob_nll(&bias, ht);

        let mut ro = LinearReadout::with_bias(vocab, d, bias);
        let cfg = FitConfig { epochs: 12, batch_size: 64, lr: 0.3, lr_decay: 1.0, ..FitConfig::default() };
        let mut frng = StdRng::seed_from_u64(7);
        let trace = ro.fit(tf, tt, hf, ht, &cfg, &mut frng);
        let final_holdout = trace.iter().map(|s| s.holdout_nll).fold(f64::INFINITY, f64::min);

        // With no signal, the readout cannot meaningfully beat the marginal.
        assert!(
            final_holdout > marginal - 0.05,
            "noise: trained holdout NLL {:.4} should not beat marginal {:.4} by much",
            final_holdout,
            marginal
        );
    }

    // The bias-only null (freeze_w) must NOT exploit separable features: with
    // W pinned at 0 it can only re-fit the marginal, so a full fit on the same
    // data should beat it. This is the control that isolates context.
    #[test]
    fn bias_only_null_cannot_use_features() {
        let d = 4;
        let vocab = 3;
        let n = 1200;
        let mut rng = StdRng::seed_from_u64(42);
        let mut feats = vec![0.0f32; n * d];
        let mut targets = vec![0u32; n];
        for e in 0..n {
            let c = (e % vocab) as u32;
            for k in 0..d {
                feats[e * d + k] = (rng.gen::<f32>() - 0.5) * 0.3;
            }
            feats[e * d + c as usize] += 2.0;
            targets[e] = c;
        }
        let split = 1000 * d;
        let (tf, hf) = feats.split_at(split);
        let (tt, ht) = targets.split_at(1000);
        let bias = unigram_log_bias(tt, vocab);
        let marginal = fixed_logprob_nll(&bias, ht);

        let mut null = LinearReadout::with_bias(vocab, d, bias.clone());
        let cfg = FitConfig {
            epochs: 12, batch_size: 64, lr: 0.3, lr_decay: 1.0, freeze_w: true,
            ..FitConfig::default()
        };
        let mut frng = StdRng::seed_from_u64(7);
        let ntrace = null.fit(tf, tt, hf, ht, &cfg, &mut frng);
        let null_holdout = ntrace.iter().map(|s| s.holdout_nll).fold(f64::INFINITY, f64::min);

        // W stayed frozen at 0, so the null can't exploit the separable signal.
        assert!(
            null_holdout > marginal - 0.05,
            "bias-only null {:.4} must not beat marginal {:.4} (W should be frozen)",
            null_holdout, marginal
        );
        assert!(null.w.iter().all(|&w| w == 0.0), "W must remain exactly 0 under freeze_w");
    }
}
