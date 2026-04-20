//! Run manifests — a TOML sidecar emitted alongside every persistent
//! artifact. Captures the exact conditions that produced the artifact so
//! downstream reads can cross-check compatibility.
//!
//! Every write-subcommand (synth, calibrate-confidence, train-router,
//! crystal, and ingest when it emits metrics) writes
//! `<output>.manifest.toml` next to its primary output. Every
//! read-subcommand that consumes such an artifact can load the manifest
//! via `RunManifest::load_sidecar` and cross-check.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Record of how an artifact was produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    /// Kernel version from `CARGO_PKG_VERSION`.
    pub kernel_version: String,
    /// Git commit that produced this artifact, if available at build time.
    /// `"unknown"` when the `CLOB_GIT_COMMIT` env var wasn't set.
    pub git_commit: String,
    /// Root seed passed to `--seed`. `None` for subcommands that don't
    /// take a seed.
    pub seed: Option<u64>,
    /// The subcommand name that produced this artifact.
    pub subcommand: String,
    /// Raw subcommand arguments in the order supplied.
    pub args: Vec<String>,
    /// UTC Unix timestamp (seconds) when the artifact was written.
    pub utc_unix_seconds: u64,
    /// Named sha256 digests of input files. Common keys: "model",
    /// "corpus", "tokenizer", "confidence_head", "meta_critic",
    /// "routers". `None` indicates the input was expected but the file
    /// was missing or unreadable at hash time.
    pub input_hashes: BTreeMap<String, Option<String>>,
    /// Optional free-form tag — the experiment label, if any.
    pub experiment: Option<String>,
}

impl RunManifest {
    /// Start a new manifest with current kernel version, commit, and
    /// timestamp. Input hashes default empty; caller adds them via
    /// `with_input`.
    pub fn new(subcommand: impl Into<String>) -> Self {
        let utc_unix_seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            kernel_version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: option_env!("CLOB_GIT_COMMIT").unwrap_or("unknown").to_string(),
            seed: None,
            subcommand: subcommand.into(),
            args: std::env::args().skip(1).collect(),
            utc_unix_seconds,
            input_hashes: BTreeMap::new(),
            experiment: std::env::var("CLOB_EXPERIMENT").ok(),
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Add a named input file's sha256 to the manifest. `None` if the file
    /// isn't readable.
    pub fn with_input(mut self, name: impl Into<String>, path: &Path) -> Self {
        self.input_hashes.insert(name.into(), crate::util::sha::hash_file(path));
        self
    }

    /// Serialize to TOML.
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_else(|e| {
            format!("# manifest serialization failed: {}\n", e)
        })
    }

    /// Write to `<output>.manifest.toml`.
    pub fn save_sidecar(&self, output: &Path) -> std::io::Result<()> {
        let manifest_path = sidecar_path(output);
        if let Some(parent) = manifest_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&manifest_path, self.to_toml())?;
        Ok(())
    }

    /// Load a sidecar manifest for the given artifact path. `None` if
    /// missing or malformed.
    pub fn load_sidecar(output: &Path) -> Option<Self> {
        let manifest_path = sidecar_path(output);
        let bytes = std::fs::read_to_string(&manifest_path).ok()?;
        toml::from_str(&bytes).ok()
    }
}

pub fn sidecar_path(output: &Path) -> std::path::PathBuf {
    let mut p = output.as_os_str().to_owned();
    p.push(".manifest.toml");
    std::path::PathBuf::from(p)
}

/// Outcome of `cross_check` — a concise string suitable for warn/fail.
#[derive(Debug, Clone)]
pub enum CheckResult {
    /// Manifest is present and all named hashes agree with the current
    /// file contents.
    Ok,
    /// Manifest is missing; caller's decision whether to warn or fail.
    NoManifest,
    /// Hash disagreement for a specific input.
    Mismatch { input: String, recorded: String, actual: String },
    /// Named input recorded in manifest but missing on disk now.
    Missing { input: String },
}

/// For each input recorded in the manifest, verify that its current
/// file content still hashes to the recorded value. Returns the first
/// mismatch, or `Ok` if all pass.
pub fn cross_check(
    manifest: &RunManifest,
    current_paths: &BTreeMap<String, &Path>,
) -> CheckResult {
    for (name, recorded) in &manifest.input_hashes {
        let recorded = match recorded {
            Some(h) => h,
            None => continue, // nothing to check against
        };
        let path = match current_paths.get(name) {
            Some(p) => *p,
            None => continue, // caller didn't supply this input for checking
        };
        match crate::util::sha::hash_file(path) {
            None => return CheckResult::Missing { input: name.clone() },
            Some(actual) if &actual != recorded => {
                return CheckResult::Mismatch {
                    input: name.clone(),
                    recorded: recorded.clone(),
                    actual,
                };
            }
            _ => {}
        }
    }
    CheckResult::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_captures_version_and_subcommand() {
        let m = RunManifest::new("synth").with_seed(7);
        assert_eq!(m.kernel_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(m.subcommand, "synth");
        assert_eq!(m.seed, Some(7));
    }

    #[test]
    fn toml_roundtrip() {
        let m = RunManifest::new("calibrate-confidence").with_seed(42);
        let s = m.to_toml();
        let parsed: RunManifest = toml::from_str(&s).unwrap();
        assert_eq!(parsed.seed, Some(42));
        assert_eq!(parsed.subcommand, "calibrate-confidence");
    }

    #[test]
    fn sidecar_save_and_load() {
        let dir = std::env::temp_dir();
        let artifact = dir.join("clob_manifest_test_output.bin");
        std::fs::write(&artifact, b"payload").unwrap();
        let m = RunManifest::new("ingest")
            .with_seed(1)
            .with_input("corpus", &artifact);
        m.save_sidecar(&artifact).unwrap();

        let loaded = RunManifest::load_sidecar(&artifact).expect("manifest should load");
        assert_eq!(loaded.seed, Some(1));
        assert!(loaded.input_hashes.get("corpus").unwrap().is_some());

        let _ = std::fs::remove_file(&artifact);
        let _ = std::fs::remove_file(sidecar_path(&artifact));
    }

    #[test]
    fn cross_check_detects_mismatch() {
        let dir = std::env::temp_dir();
        let artifact = dir.join("clob_manifest_check_test.bin");
        std::fs::write(&artifact, b"original").unwrap();
        let m = RunManifest::new("ingest")
            .with_seed(1)
            .with_input("corpus", &artifact);

        // Same content → Ok.
        let mut paths = BTreeMap::new();
        paths.insert("corpus".to_string(), artifact.as_path());
        let res = cross_check(&m, &paths);
        assert!(matches!(res, CheckResult::Ok));

        // Mutate file → Mismatch.
        std::fs::write(&artifact, b"mutated").unwrap();
        let res = cross_check(&m, &paths);
        assert!(matches!(res, CheckResult::Mismatch { .. }));

        // Remove file → Missing.
        std::fs::remove_file(&artifact).unwrap();
        let res = cross_check(&m, &paths);
        assert!(matches!(res, CheckResult::Missing { .. }));
    }
}
