//! Bench-suite: run the full ablation matrix and emit a TSV.
//!
//! The matrix axes (as of Phase K):
//!   - seeds       (repeated runs for noise estimation)
//!   - adaptive    {off, on}
//!   - modules     {absent, loaded from dir}
//!   - routers     {pristine, trained side-file}
//!
//! For each cell, load a fresh model, install the relevant artifacts,
//! run an eval pass over `max_tokens` of the held-out corpus, and record
//! mean_nll + j_per_nat + tok/s. Emit one TSV row per cell.
//!
//! The comparison that matters: cells should differ only along one axis
//! at a time so the delta in J/nat and mean_nll is interpretable. The
//! caller composes the matrix; `run_matrix` is the executor.

use crate::crystal::store;
use crate::metrics::{EnergyMeter, TdpProxyMeter};
use crate::nn::confidence::{AdaptiveConfig, ConfidenceHead, MetaCritic};
use crate::nn::energy::EnergyCritic;
use crate::token::bpe::BpeTokenizer;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// One row of the bench matrix.
#[derive(Debug, Clone)]
pub struct BenchCell {
    pub run_id: String,
    pub seed: u64,
    pub adaptive: bool,
    pub n_modules: usize,
    pub router_state: RouterState,
    pub mean_nll_holdout: f64,
    pub j_per_nat: f64,
    pub tokens_per_sec: f64,
    pub tokens: u64,
    pub joules: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterState { Pristine, Trained }

impl RouterState {
    pub fn as_str(&self) -> &'static str {
        match self { Self::Pristine => "pristine", Self::Trained => "trained" }
    }
}

/// The matrix to execute. Each vector is the set of values for one axis;
/// the cartesian product defines the cells.
pub struct BenchMatrix<'a> {
    pub seeds: Vec<u64>,
    pub adaptive: Vec<bool>,
    pub modules_dir: Vec<Option<&'a Path>>,
    pub routers_path: Vec<Option<&'a Path>>,
    pub max_tokens: usize,
    pub adaptive_max_extra: u8,
    pub adaptive_z: f32,
}

/// Artifacts passed through to each cell.
pub struct BenchInputs<'a> {
    pub model_path: &'a Path,
    pub tokenizer: &'a BpeTokenizer,
    pub holdout_tokens: &'a [u32],
    pub critic: Option<&'a EnergyCritic>,
    pub confidence_head: Option<&'a ConfidenceHead>,
    pub meta_critic: Option<&'a MetaCritic>,
}

/// Execute the full matrix. Returns one BenchCell per cell (order is
/// seed-major, then adaptive, then modules, then routers).
pub fn run_matrix(matrix: &BenchMatrix, inputs: &BenchInputs) -> std::io::Result<Vec<BenchCell>> {
    let mut cells = Vec::new();
    for &seed in &matrix.seeds {
        for &adaptive in &matrix.adaptive {
            for modules_dir in &matrix.modules_dir {
                for routers_path in &matrix.routers_path {
                    let router_state = match routers_path {
                        Some(_) => RouterState::Trained,
                        None => RouterState::Pristine,
                    };
                    let run_id = format!(
                        "s{}_{}_{}m_{}r",
                        seed,
                        if adaptive { "adaptON" } else { "adaptOFF" },
                        if modules_dir.is_some() { "with" } else { "no" },
                        router_state.as_str(),
                    );

                    let cell = run_single_cell(
                        &run_id, seed, adaptive,
                        *modules_dir, *routers_path, router_state,
                        matrix, inputs,
                    )?;
                    cells.push(cell);
                }
            }
        }
    }
    Ok(cells)
}

fn run_single_cell(
    run_id: &str,
    seed: u64,
    adaptive: bool,
    modules_dir: Option<&Path>,
    routers_path: Option<&Path>,
    router_state: RouterState,
    matrix: &BenchMatrix,
    inputs: &BenchInputs,
) -> std::io::Result<BenchCell> {
    // Fresh model per cell to avoid leakage across ablations.
    let mut model = crate::io::loader::load_model(inputs.model_path)?;

    // Install optional components.
    if let Some(c) = inputs.critic {
        model.replace_energy_critic(c.clone());
    }
    if let Some(h) = inputs.confidence_head {
        model.set_confidence_head(h.clone());
    }
    if let Some(m) = inputs.meta_critic {
        model.set_meta_critic(m.clone());
    }

    // Routers: load and apply side-file weights if provided.
    if let Some(p) = routers_path {
        let bytes = std::fs::read(p)?;
        let loaded: Vec<(u32, Vec<f32>)> = bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        for (layer_idx, new_w) in loaded {
            if let Some(router) = model.router_mut(layer_idx as usize) {
                if router.weights().len() == new_w.len() {
                    router.weights_mut().copy_from_slice(&new_w);
                }
            }
        }
    }

    // Modules: load from directory if provided.
    let mut n_modules = 0usize;
    if let Some(d) = modules_dir {
        let modules = store::load_modules(d).unwrap_or_default();
        n_modules = modules.len();
        for m in modules {
            model.push_crystal_module(m);
        }
    }

    model.adaptive_config = AdaptiveConfig {
        enabled: adaptive,
        max_extra_steps: matrix.adaptive_max_extra,
        z_threshold: matrix.adaptive_z,
        meta_unreliable_threshold: 2.0,
    };

    // Run the eval pass.
    let limit = if matrix.max_tokens == 0 {
        inputs.holdout_tokens.len()
    } else {
        inputs.holdout_tokens.len().min(matrix.max_tokens)
    };
    let mut total_nll = 0.0f64;
    let mut n_predicted = 0u64;
    let mut joules = 0.0f64;
    let mut meter = TdpProxyMeter::t490();
    model.reset_state();
    let start = Instant::now();
    let mut last = start;
    for i in 0..limit.saturating_sub(1) {
        let logits = model.decode_step(inputs.holdout_tokens[i]);
        let p = prob_of_token(logits.data(), inputs.holdout_tokens[i + 1]);
        let nll = -((p.max(1e-12)) as f64).ln();
        total_nll += nll;
        n_predicted += 1;
        let (j, now) = meter.joules_since(last);
        joules += j;
        last = now;
        // Drop the extra-steps counter (we don't record it here; bench-suite
        // focuses on outcome metrics, not runtime detail).
        let _ = model.take_extra_steps();
        let _ = model.take_meta_suppressed();
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let mean_nll_holdout = if n_predicted > 0 {
        total_nll / n_predicted as f64
    } else { 0.0 };
    let j_per_nat = if total_nll > 1e-9 { joules / total_nll } else { 0.0 };
    let tokens_per_sec = n_predicted as f64 / elapsed;

    // The seed doesn't affect eval determinism — the model is
    // deterministic — but it's captured for parity with training cells.
    let _ = seed;

    Ok(BenchCell {
        run_id: run_id.to_string(),
        seed,
        adaptive,
        n_modules,
        router_state,
        mean_nll_holdout,
        j_per_nat,
        tokens_per_sec,
        tokens: n_predicted,
        joules,
    })
}

/// Same numerical-stability helper as cmd_ingest uses.
fn prob_of_token(logits: &[f32], token: u32) -> f32 {
    let tok = token as usize;
    if tok >= logits.len() { return 0.0; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits { sum += (l - max).exp(); }
    if sum <= 0.0 { return 0.0; }
    ((logits[tok] - max).exp()) / sum
}

/// Emit a TSV table. First row is a header. Floats are fixed-precision
/// for deterministic diffs.
pub fn emit_tsv(cells: &[BenchCell], out: &Path) -> std::io::Result<()> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut f = std::fs::File::create(out)?;
    writeln!(
        f,
        "run_id\tseed\tadaptive\tn_modules\trouter\ttokens\tmean_nll\tj_per_nat\ttok_per_sec\tjoules"
    )?;
    for c in cells {
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.2}\t{:.4}",
            c.run_id, c.seed, c.adaptive, c.n_modules, c.router_state.as_str(),
            c.tokens, c.mean_nll_holdout, c.j_per_nat,
            c.tokens_per_sec, c.joules,
        )?;
    }
    Ok(())
}

/// Pivot: mean of each cell's mean_nll_holdout and j_per_nat across seeds,
/// keyed by (adaptive, n_modules>0, router). Useful for scanning which
/// ablation axis moves the number.
pub fn summarize_by_axis(cells: &[BenchCell]) -> String {
    let mut groups: HashMap<(bool, bool, &str), Vec<&BenchCell>> = HashMap::new();
    for c in cells {
        groups.entry((c.adaptive, c.n_modules > 0, c.router_state.as_str()))
            .or_default()
            .push(c);
    }
    let mut out = String::new();
    out.push_str(&format!(
        "{:<10} {:<10} {:<10} {:>8} {:>12} {:>12} {:>8}\n",
        "adaptive", "modules", "router", "n_seeds", "mean_nll", "j_per_nat", "tok/s",
    ));
    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort();
    for key in keys {
        let rows = &groups[&key];
        let n = rows.len() as f64;
        let mean_nll = rows.iter().map(|c| c.mean_nll_holdout).sum::<f64>() / n;
        let jpn = rows.iter().map(|c| c.j_per_nat).sum::<f64>() / n;
        let tps = rows.iter().map(|c| c.tokens_per_sec).sum::<f64>() / n;
        out.push_str(&format!(
            "{:<10} {:<10} {:<10} {:>8} {:>12.4} {:>12.6} {:>8.1}\n",
            key.0, if key.1 { "loaded" } else { "empty" }, key.2,
            rows.len(), mean_nll, jpn, tps,
        ));
    }
    out
}

pub fn cartesian_option_paths(paths: &[PathBuf]) -> Vec<Option<&Path>> {
    let mut v: Vec<Option<&Path>> = Vec::with_capacity(paths.len() + 1);
    v.push(None);
    for p in paths { v.push(Some(p.as_path())); }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_state_as_str() {
        assert_eq!(RouterState::Pristine.as_str(), "pristine");
        assert_eq!(RouterState::Trained.as_str(), "trained");
    }

    #[test]
    fn summarize_aggregates_across_seeds() {
        let cells = vec![
            BenchCell {
                run_id: "a".into(), seed: 1, adaptive: false, n_modules: 0,
                router_state: RouterState::Pristine, mean_nll_holdout: 7.0,
                j_per_nat: 0.001, tokens_per_sec: 1000.0, tokens: 500, joules: 0.5,
            },
            BenchCell {
                run_id: "b".into(), seed: 2, adaptive: false, n_modules: 0,
                router_state: RouterState::Pristine, mean_nll_holdout: 7.2,
                j_per_nat: 0.0011, tokens_per_sec: 950.0, tokens: 500, joules: 0.55,
            },
        ];
        let s = summarize_by_axis(&cells);
        // Both cells collapse into one row with n_seeds=2 and mean_nll=7.1.
        assert!(s.contains("7.1000"), "got:\n{}", s);
    }
}
