//! Rolling aggregation window — sums per-step measurements into snapshots.
//!
//! A `MetricsWindow` is emptied by `snapshot()`, which returns a
//! `WindowSnapshot` carrying the top-line numbers (J/nat, tokens/sec,
//! mean NLL, held-out CE). Snapshots are serialized to jsonl by
//! `JsonlWriter` for the `clob metrics` dashboard to tail.
//!
//! JSON is hand-serialized to avoid pulling in `serde_json` — the snapshot
//! is a flat record of named f64/u64 fields, trivially round-trippable.

use std::time::Instant;

#[derive(Debug)]
pub struct MetricsWindow {
    start: Instant,
    window_idx: u64,
    tokens: u64,
    nanos: u128,
    joules: f64,
    sum_nll: f64,
    sum_nll_holdout: f64,
    n_holdout: u64,
    /// Counter of extra SSM steps triggered by adaptive compute
    /// (Phase B+C will begin populating this; stays zero until then).
    extra_steps: u64,
    /// Count of tokens gated as "novel" by the energy critic.
    novel_tokens: u64,
}

impl MetricsWindow {
    pub fn new(window_idx: u64) -> Self {
        Self {
            start: Instant::now(),
            window_idx,
            tokens: 0,
            nanos: 0,
            joules: 0.0,
            sum_nll: 0.0,
            sum_nll_holdout: 0.0,
            n_holdout: 0,
            extra_steps: 0,
            novel_tokens: 0,
        }
    }

    pub fn record_step(&mut self, nll: f64, joules: f64, nanos: u128) {
        self.tokens += 1;
        self.nanos += nanos;
        self.joules += joules;
        self.sum_nll += nll;
    }

    pub fn record_holdout(&mut self, nll: f64) {
        self.sum_nll_holdout += nll;
        self.n_holdout += 1;
    }

    pub fn record_extra_step(&mut self) {
        self.extra_steps += 1;
    }

    pub fn record_novel(&mut self) {
        self.novel_tokens += 1;
    }

    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    /// Produce a snapshot and reset the window for the next interval.
    pub fn snapshot_and_reset(&mut self) -> WindowSnapshot {
        let snap = self.snapshot();
        let next_idx = self.window_idx + 1;
        *self = Self::new(next_idx);
        snap
    }

    pub fn snapshot(&self) -> WindowSnapshot {
        let seconds = self.start.elapsed().as_secs_f64().max(1e-9);
        let mean_nll = if self.tokens > 0 {
            self.sum_nll / self.tokens as f64
        } else {
            0.0
        };
        let mean_nll_holdout = if self.n_holdout > 0 {
            Some(self.sum_nll_holdout / self.n_holdout as f64)
        } else {
            None
        };
        // J/nat = joules / total_nats. Guard against division by zero.
        let j_per_nat = if self.sum_nll > 1e-9 {
            self.joules / self.sum_nll
        } else {
            0.0
        };
        let tokens_per_sec = self.tokens as f64 / seconds;
        WindowSnapshot {
            window_idx: self.window_idx,
            tokens: self.tokens,
            seconds,
            joules: self.joules,
            mean_nll,
            mean_nll_holdout,
            j_per_nat,
            tokens_per_sec,
            extra_steps: self.extra_steps,
            novel_tokens: self.novel_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WindowSnapshot {
    pub window_idx: u64,
    pub tokens: u64,
    pub seconds: f64,
    pub joules: f64,
    pub mean_nll: f64,
    pub mean_nll_holdout: Option<f64>,
    pub j_per_nat: f64,
    pub tokens_per_sec: f64,
    pub extra_steps: u64,
    pub novel_tokens: u64,
}

impl WindowSnapshot {
    /// One-line JSON — keep keys stable; the `clob metrics` tailer parses
    /// them positionally as a fallback when full JSON parsing is not wired.
    pub fn to_json_line(&self) -> String {
        let holdout = match self.mean_nll_holdout {
            Some(v) => format!("{}", v),
            None => "null".to_string(),
        };
        format!(
            "{{\"idx\":{},\"tokens\":{},\"seconds\":{},\"joules\":{},\"mean_nll\":{},\"mean_nll_holdout\":{},\"j_per_nat\":{},\"tokens_per_sec\":{},\"extra_steps\":{},\"novel_tokens\":{}}}",
            self.window_idx,
            self.tokens,
            self.seconds,
            self.joules,
            self.mean_nll,
            holdout,
            self.j_per_nat,
            self.tokens_per_sec,
            self.extra_steps,
            self.novel_tokens,
        )
    }

    /// Parse a jsonl line produced by `to_json_line`. Returns `None` on any
    /// malformed field — the tailer will skip silently rather than panic.
    pub fn from_json_line(line: &str) -> Option<Self> {
        let inner = line.trim().strip_prefix('{')?.strip_suffix('}')?;
        let mut idx = 0u64;
        let mut tokens = 0u64;
        let mut seconds = 0.0f64;
        let mut joules = 0.0f64;
        let mut mean_nll = 0.0f64;
        let mut mean_nll_holdout: Option<f64> = None;
        let mut j_per_nat = 0.0f64;
        let mut tokens_per_sec = 0.0f64;
        let mut extra_steps = 0u64;
        let mut novel_tokens = 0u64;

        for field in inner.split(',') {
            let (raw_k, raw_v) = field.split_once(':')?;
            let k = raw_k.trim().trim_matches('"');
            let v = raw_v.trim();
            match k {
                "idx" => idx = v.parse().ok()?,
                "tokens" => tokens = v.parse().ok()?,
                "seconds" => seconds = v.parse().ok()?,
                "joules" => joules = v.parse().ok()?,
                "mean_nll" => mean_nll = v.parse().ok()?,
                "mean_nll_holdout" => {
                    mean_nll_holdout = if v == "null" { None } else { v.parse().ok() };
                }
                "j_per_nat" => j_per_nat = v.parse().ok()?,
                "tokens_per_sec" => tokens_per_sec = v.parse().ok()?,
                "extra_steps" => extra_steps = v.parse().ok()?,
                "novel_tokens" => novel_tokens = v.parse().ok()?,
                _ => {}
            }
        }

        Some(Self {
            window_idx: idx,
            tokens,
            seconds,
            joules,
            mean_nll,
            mean_nll_holdout,
            j_per_nat,
            tokens_per_sec,
            extra_steps,
            novel_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_records_and_snapshots() {
        let mut w = MetricsWindow::new(0);
        for _ in 0..10 {
            w.record_step(0.5, 0.001, 1_000_000);
        }
        w.record_holdout(0.3);
        w.record_holdout(0.4);
        let snap = w.snapshot();
        assert_eq!(snap.tokens, 10);
        assert!((snap.mean_nll - 0.5).abs() < 1e-9);
        assert!((snap.joules - 0.01).abs() < 1e-9);
        assert!((snap.j_per_nat - 0.01 / 5.0).abs() < 1e-9);
        let holdout = snap.mean_nll_holdout.unwrap();
        assert!((holdout - 0.35).abs() < 1e-9);
    }

    #[test]
    fn json_roundtrip_preserves_fields() {
        let mut w = MetricsWindow::new(7);
        w.record_step(0.8, 0.02, 5_000_000);
        w.record_step(0.6, 0.03, 6_000_000);
        w.record_holdout(0.5);
        w.record_extra_step();
        w.record_novel();
        let snap = w.snapshot();
        let line = snap.to_json_line();
        let parsed = WindowSnapshot::from_json_line(&line).expect("parse");
        assert_eq!(parsed.window_idx, 7);
        assert_eq!(parsed.tokens, 2);
        assert!((parsed.mean_nll - 0.7).abs() < 1e-9);
        assert_eq!(parsed.extra_steps, 1);
        assert_eq!(parsed.novel_tokens, 1);
        assert!(parsed.mean_nll_holdout.is_some());
    }

    #[test]
    fn empty_window_doesnt_divide_by_zero() {
        let w = MetricsWindow::new(0);
        let snap = w.snapshot();
        assert_eq!(snap.tokens, 0);
        assert_eq!(snap.j_per_nat, 0.0);
        assert!(snap.mean_nll_holdout.is_none());
    }

    #[test]
    fn null_holdout_roundtrip() {
        let mut w = MetricsWindow::new(0);
        w.record_step(1.0, 0.001, 100);
        let snap = w.snapshot();
        let line = snap.to_json_line();
        assert!(line.contains("\"mean_nll_holdout\":null"));
        let parsed = WindowSnapshot::from_json_line(&line).unwrap();
        assert!(parsed.mean_nll_holdout.is_none());
    }
}
