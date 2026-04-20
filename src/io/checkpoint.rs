//! Checkpoint bundles — the unit of "save everything that matters" for
//! resume-after-crash support on long-running training.
//!
//! A checkpoint is a directory, not a file. Its contents:
//!
//! ```text
//! checkpoint_00042/
//! ├── model.clob              # current model weights
//! ├── confidence.bin          # optional Head C
//! ├── meta.bin                # optional MetaCritic
//! ├── routers.bin             # optional trained routers
//! ├── progress.toml           # { tokens_processed, step, seed, ... }
//! └── manifest.toml           # standard Phase H manifest
//! ```
//!
//! Any individual file may be missing — the training loop decides what's
//! meaningful to snapshot. The progress record is the source of truth for
//! "where in the corpus were we" and must be present.

use crate::config::manifest::RunManifest;
use crate::io::writer::save_model;
use crate::model::stack::CoreModel;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const MODEL_NAME: &str = "model.clob";
const CONFIDENCE_NAME: &str = "confidence.bin";
const META_NAME: &str = "meta.bin";
const ROUTERS_NAME: &str = "routers.bin";
const PROGRESS_NAME: &str = "progress.toml";

/// Serializable training progress marker. Written alongside the model so
/// `--resume-from` can skip already-processed tokens without recomputing
/// them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointProgress {
    /// Total tokens the owning subcommand has processed so far.
    pub tokens_processed: u64,
    /// For training loops: number of optimizer steps completed.
    pub optimizer_steps: u64,
    /// Root seed the owning subcommand was started with.
    pub seed: u64,
    /// Subcommand that produced this checkpoint ("train-router",
    /// "calibrate-confidence", "ingest", ...).
    pub subcommand: String,
    /// Free-form per-subcommand state (e.g., baseline EMA values in
    /// train-router). Caller-defined TOML table, empty by default.
    pub extras: toml::Table,
}

impl CheckpointProgress {
    pub fn new(subcommand: impl Into<String>, seed: u64) -> Self {
        Self {
            tokens_processed: 0,
            optimizer_steps: 0,
            seed,
            subcommand: subcommand.into(),
            extras: toml::Table::new(),
        }
    }
}

/// Numbered checkpoint directory, e.g.
/// `<base_dir>/checkpoint_00042`. Zero-padded to 5 digits so lexical
/// sort matches numerical sort up to 99999.
pub fn checkpoint_dir(base_dir: &Path, index: u64) -> PathBuf {
    base_dir.join(format!("checkpoint_{:05}", index))
}

/// Scan `base_dir` for `checkpoint_<N>` subdirs and return the highest N.
pub fn latest_index(base_dir: &Path) -> Option<u64> {
    let entries = std::fs::read_dir(base_dir).ok()?;
    let mut best: Option<u64> = None;
    for e in entries.flatten() {
        if let Some(name) = e.file_name().to_str() {
            if let Some(num_str) = name.strip_prefix("checkpoint_") {
                if let Ok(n) = num_str.parse::<u64>() {
                    best = Some(best.map_or(n, |b| b.max(n)));
                }
            }
        }
    }
    best
}

/// Save a checkpoint to `<base_dir>/checkpoint_<index>/`.
///
/// Each `Option<_>` is written only if `Some`. The progress record is
/// always written.
pub fn save(
    base_dir: &Path,
    index: u64,
    model: &CoreModel,
    confidence: Option<&[u8]>,
    meta: Option<&[u8]>,
    routers: Option<&[u8]>,
    progress: &CheckpointProgress,
    manifest: &RunManifest,
) -> std::io::Result<PathBuf> {
    let dir = checkpoint_dir(base_dir, index);
    std::fs::create_dir_all(&dir)?;

    save_model(model, &dir.join(MODEL_NAME))?;
    if let Some(bytes) = confidence {
        std::fs::write(dir.join(CONFIDENCE_NAME), bytes)?;
    }
    if let Some(bytes) = meta {
        std::fs::write(dir.join(META_NAME), bytes)?;
    }
    if let Some(bytes) = routers {
        std::fs::write(dir.join(ROUTERS_NAME), bytes)?;
    }

    let progress_toml = toml::to_string_pretty(progress).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    })?;
    std::fs::write(dir.join(PROGRESS_NAME), progress_toml)?;

    // Emit the standard manifest inside the checkpoint dir.
    std::fs::write(dir.join("manifest.toml"), manifest.to_toml())?;

    Ok(dir)
}

/// Load-side view: paths to each file within a checkpoint directory.
/// Files that exist are returned; missing ones are `None`. The progress
/// record is required; its absence is an error.
pub struct CheckpointHandles {
    pub dir: PathBuf,
    pub model: PathBuf,
    pub confidence: Option<PathBuf>,
    pub meta: Option<PathBuf>,
    pub routers: Option<PathBuf>,
    pub progress: CheckpointProgress,
    pub manifest: Option<RunManifest>,
}

pub fn load_handles(dir: &Path) -> std::io::Result<CheckpointHandles> {
    let model_path = dir.join(MODEL_NAME);
    if !model_path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("checkpoint missing model.clob at {:?}", dir),
        ));
    }

    let progress_bytes = std::fs::read_to_string(dir.join(PROGRESS_NAME))?;
    let progress: CheckpointProgress = toml::from_str(&progress_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    let opt = |name: &str| -> Option<PathBuf> {
        let p = dir.join(name);
        if p.exists() { Some(p) } else { None }
    };

    let manifest_path = dir.join("manifest.toml");
    let manifest = if manifest_path.exists() {
        std::fs::read_to_string(&manifest_path).ok()
            .and_then(|s| toml::from_str(&s).ok())
    } else {
        None
    };

    Ok(CheckpointHandles {
        dir: dir.to_path_buf(),
        model: model_path,
        confidence: opt(CONFIDENCE_NAME),
        meta: opt(META_NAME),
        routers: opt(ROUTERS_NAME),
        progress,
        manifest,
    })
}

/// Convenience: resolve `--resume-from <X>` where `X` may be `"latest"` or
/// a specific checkpoint directory path.
pub fn resolve_resume(base_dir: &Path, spec: &str) -> std::io::Result<PathBuf> {
    if spec == "latest" {
        let idx = latest_index(base_dir).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no checkpoints found under {:?}", base_dir),
            )
        })?;
        Ok(checkpoint_dir(base_dir, idx))
    } else {
        let p = PathBuf::from(spec);
        if p.exists() {
            Ok(p)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("--resume-from path not found: {:?}", p),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::loader::load_model;
    use crate::model::config::KernelConfig;
    use rand::SeedableRng;

    #[test]
    fn checkpoint_dir_names_are_zero_padded() {
        let d = checkpoint_dir(Path::new("/tmp/base"), 42);
        assert_eq!(d, PathBuf::from("/tmp/base/checkpoint_00042"));
    }

    #[test]
    fn latest_index_picks_highest() {
        let base = std::env::temp_dir().join("clob_ckpt_latest_test");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(base.join("checkpoint_00001")).unwrap();
        std::fs::create_dir_all(base.join("checkpoint_00042")).unwrap();
        std::fs::create_dir_all(base.join("checkpoint_00003")).unwrap();
        // A noise dir that shouldn't be counted.
        std::fs::create_dir_all(base.join("not_a_checkpoint")).unwrap();
        assert_eq!(latest_index(&base), Some(42));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let base = std::env::temp_dir().join("clob_ckpt_roundtrip_test");
        let _ = std::fs::remove_dir_all(&base);

        let config = KernelConfig::tiny();
        let mut rng = rand::rngs::StdRng::seed_from_u64(55);
        let model = CoreModel::random(config, &mut rng);

        let mut progress = CheckpointProgress::new("train-router", 3);
        progress.tokens_processed = 1234;
        progress.optimizer_steps = 17;

        let manifest = RunManifest::new("train-router").with_seed(3);

        let confidence_bytes = b"pretend this is a conf.bin";
        let routers_bytes = b"pretend this is a routers.bin";

        let dir = save(
            &base, 1, &model,
            Some(confidence_bytes), None, Some(routers_bytes),
            &progress, &manifest,
        ).unwrap();
        assert!(dir.ends_with("checkpoint_00001"));

        let handles = load_handles(&dir).unwrap();
        assert_eq!(handles.progress.tokens_processed, 1234);
        assert_eq!(handles.progress.optimizer_steps, 17);
        assert_eq!(handles.progress.seed, 3);
        assert!(handles.confidence.is_some());
        assert!(handles.meta.is_none());
        assert!(handles.routers.is_some());
        assert!(handles.manifest.is_some());

        // Model round-trips through the same save_model path.
        let restored = load_model(&handles.model).unwrap();
        assert_eq!(restored.config.d_model, 64);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn resolve_resume_handles_latest() {
        let base = std::env::temp_dir().join("clob_ckpt_resolve_test");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(base.join("checkpoint_00007")).unwrap();

        let got = resolve_resume(&base, "latest").unwrap();
        assert_eq!(got, base.join("checkpoint_00007"));

        // Explicit path resolves if it exists.
        let got = resolve_resume(&base, base.join("checkpoint_00007").to_str().unwrap()).unwrap();
        assert_eq!(got, base.join("checkpoint_00007"));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn latest_on_empty_dir_is_none() {
        let base = std::env::temp_dir().join("clob_ckpt_empty_test");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        assert!(latest_index(&base).is_none());
        let _ = std::fs::remove_dir_all(&base);
    }
}
