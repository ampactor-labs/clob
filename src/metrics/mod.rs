//! Instrumentation and dashboard — joules-per-nat measurement.
//!
//! The top-line metric for the kernel is `J/nat`: joules spent per nat of
//! cross-entropy reduction on predictions. Every phase of the upgrade plan
//! is gated on moving this number down without regressing absolute CE on a
//! held-out corpus.
//!
//! This module has one load-bearing abstraction:
//!
//! ```text
//! pub trait EnergyMeter {
//!     fn joules_since(&mut self, t: Instant) -> (f64, Instant);
//! }
//! ```
//!
//! Current impl: `TdpProxyMeter` — wall-clock × TDP × utilization. Wrong in
//! absolute terms, correct in relative terms, which is all the dashboard
//! needs. A RAPL-backed impl can slot in later behind the same trait.

pub mod window;

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

pub use window::{MetricsWindow, WindowSnapshot};

/// A source of "how much energy was spent since time `t`?"
///
/// Implementations must be cheap enough to call on the inference hot path
/// (ideally once per token, or once per window).
pub trait EnergyMeter: Send {
    /// Returns `(joules_since_t, now)`. Callers pass `now` back into the
    /// next call as `t`.
    fn joules_since(&mut self, t: Instant) -> (f64, Instant);
}

/// A wall-clock × TDP × utilization energy estimator.
///
/// `utilization` is read from `/proc/loadavg` lazily — refreshed at most
/// once per `refresh_period`. On non-Linux platforms utilization is pinned
/// to `1.0` (single-core saturated estimate).
pub struct TdpProxyMeter {
    tdp_watts: f64,
    utilization: f64,
    last_refresh: Instant,
    refresh_period_secs: f64,
}

impl TdpProxyMeter {
    /// i7-8550U has a nominal TDP of 15 W — a reasonable default for the
    /// T490 target hardware. Pass your own TDP on other machines.
    pub fn new(tdp_watts: f64) -> Self {
        let mut m = Self {
            tdp_watts,
            utilization: 1.0,
            last_refresh: Instant::now(),
            refresh_period_secs: 5.0,
        };
        m.refresh_utilization();
        m
    }

    /// i7-8550U default.
    pub fn t490() -> Self {
        Self::new(15.0)
    }

    fn refresh_utilization(&mut self) {
        self.last_refresh = Instant::now();
        #[cfg(target_os = "linux")]
        {
            if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
                // Format: "1.05 0.98 0.92 3/456 12345"
                // Use the 1-min load average divided by core count, clamped to [0.1, 1.0].
                if let Some(first) = s.split_whitespace().next() {
                    if let Ok(load) = first.parse::<f64>() {
                        let cores = num_cores() as f64;
                        self.utilization = (load / cores).clamp(0.1, 1.0);
                    }
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.utilization = 1.0;
        }
    }
}

impl EnergyMeter for TdpProxyMeter {
    fn joules_since(&mut self, t: Instant) -> (f64, Instant) {
        let now = Instant::now();
        if now.duration_since(self.last_refresh).as_secs_f64() > self.refresh_period_secs {
            self.refresh_utilization();
        }
        let secs = now.duration_since(t).as_secs_f64();
        (secs * self.tdp_watts * self.utilization, now)
    }
}

#[cfg(target_os = "linux")]
fn num_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Append-only jsonl writer for window snapshots.
///
/// Records are self-describing (one flat JSON object per line) so the
/// `clob metrics` tailer does not need schema coordination.
pub struct JsonlWriter {
    file: File,
}

impl JsonlWriter {
    pub fn create(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }

    pub fn append(&mut self, snap: &WindowSnapshot) -> io::Result<()> {
        writeln!(self.file, "{}", snap.to_json_line())?;
        self.file.sync_data()?;
        Ok(())
    }
}

/// Tail the last `n` lines of a jsonl metrics file and parse them.
/// Used by `clob metrics` to display the dashboard.
pub fn read_tail(path: &Path, n: usize) -> io::Result<Vec<WindowSnapshot>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<io::Result<_>>()?;
    let start = lines.len().saturating_sub(n);
    let mut out = Vec::with_capacity(lines.len() - start);
    for line in &lines[start..] {
        if let Some(snap) = WindowSnapshot::from_json_line(line) {
            out.push(snap);
        }
    }
    Ok(out)
}
