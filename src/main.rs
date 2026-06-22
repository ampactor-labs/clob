//! The Kernel — CLI entry point and main loop.

use clap::{Parser, Subcommand};
use clob::crystal::detector::NoveltyDetector;
use clob::crystal::engine::{CrystalConfig, CrystallizationEngine};
use clob::crystal::store;
use clob::io;
use clob::memory::episode::Episode;
use clob::memory::ring::EpisodicMemory;
use clob::model::config::KernelConfig;
use clob::model::generate::{self, SamplingConfig};
use clob::model::stack::CoreModel;
use clob::nn::energy::EnergyCritic;
use clob::metrics::EnergyMeter;
use clob::nn::confidence::{AdaptiveConfig, ConfidenceHead, MetaCritic};
use clob::token::bpe::BpeTokenizer;
use rand::SeedableRng;
use rand::Rng;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser)]
#[command(name = "clob")]
#[command(about = "The Kernel — a self-improving ternary recurrent intelligence")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Bootstrap: create seed model, init memory, enter main loop.
    Boot {
        #[arg(long, default_value = "episodes")]
        memory_dir: PathBuf,
        #[arg(long, default_value = "modules")]
        modules_dir: PathBuf,
        #[arg(long, default_value = "seed.clob")]
        model: PathBuf,
        #[arg(long, default_value = "tiny")]
        config: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 10000)]
        memory_capacity: usize,
        /// Tokenizer file (BPE). If missing, uses byte-level.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Listen address for network peers.
        #[arg(long)]
        listen: Option<String>,
        /// Manual peer addresses.
        #[arg(long)]
        peer: Vec<String>,
    },
    /// Interactive REPL.
    Think {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Load crystal modules from this directory.
        #[arg(long)]
        modules_dir: Option<PathBuf>,
    },
    /// Generate a synthetic model.
    Synth {
        #[arg(long, default_value = "seed.clob")]
        output: PathBuf,
        #[arg(long, default_value = "tiny")]
        config: String,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Benchmark throughput.
    Bench {
        #[arg(long, default_value = "tiny")]
        config: String,
        #[arg(long, default_value_t = 32)]
        seq_len: usize,
        #[arg(long, default_value_t = 128)]
        decode_steps: usize,
    },
    /// Show model info.
    Info {
        #[arg(long)]
        model: PathBuf,
    },
    /// Show memory/crystal status.
    Status {
        #[arg(long, default_value = "episodes")]
        memory_dir: PathBuf,
    },
    /// Train a BPE tokenizer from a text file.
    TrainTokenizer {
        /// Input corpus file.
        #[arg(long)]
        corpus: PathBuf,
        /// Output tokenizer file.
        #[arg(long, default_value = "tokenizer.bin")]
        output: PathBuf,
        /// Number of BPE merges.
        #[arg(long, default_value_t = 4000)]
        n_merges: usize,
    },
    /// Encode a text corpus into a cached `.tokens` array (skips BPE on every
    /// later run; consumed by `ingest` when --input ends in `.tokens`).
    Encode {
        /// Input text corpus.
        #[arg(long)]
        input: PathBuf,
        /// Output `.tokens` file.
        #[arg(long)]
        output: PathBuf,
        /// Tokenizer file (BPE). If missing, uses byte-level.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },
    /// Process a file through the kernel (non-interactive).
    /// Streams tokens; records true prediction errors for each next-token prediction.
    Ingest {
        #[arg(long)]
        model: PathBuf,
        /// File to ingest.
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "episodes")]
        memory_dir: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Load existing crystal modules from this directory before ingesting.
        #[arg(long)]
        modules_dir: Option<PathBuf>,
        /// Load a trained energy critic side-file before ingesting.
        #[arg(long)]
        critic: Option<PathBuf>,
        /// Cap on tokens to ingest (0 = unlimited).
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
        /// Write per-window metrics (J/nat, tokens/sec) to a jsonl file.
        #[arg(long)]
        metrics_out: Option<PathBuf>,
        /// Tokens per metrics window (snapshot is emitted every N tokens).
        #[arg(long, default_value_t = 500)]
        metrics_window: u64,
        /// Load a trained confidence head (Head C) side-file before ingesting.
        #[arg(long)]
        confidence_head: Option<PathBuf>,
        /// Load a trained MetaCritic side-file (Phase F). When present, gates
        /// adaptive iteration — suppresses extra steps at hidden states where
        /// Head C is predicted to be unreliable.
        #[arg(long)]
        meta_critic: Option<PathBuf>,
        /// Load a trained routers side-file (Phase F). Replaces MoE router
        /// weights with the trained version before ingestion.
        #[arg(long)]
        routers: Option<PathBuf>,
        /// Enable adaptive test-time compute: when Head C signals low
        /// confidence, the block stack iterates extra times before
        /// unembedding. Requires --confidence-head to be set.
        #[arg(long)]
        adaptive_compute: bool,
        /// Maximum extra SSM iterations per token when adaptive compute fires.
        #[arg(long, default_value_t = 4)]
        adaptive_max_extra: u8,
        /// Z-score threshold on Head C's detector for firing adaptive compute.
        #[arg(long, default_value_t = 1.0)]
        adaptive_z: f32,
        /// MetaCritic prediction threshold: above this, adaptive is suppressed.
        #[arg(long, default_value_t = 2.0)]
        meta_unreliable_threshold: f32,
        /// Save a checkpoint every N tokens under --checkpoint-dir.
        /// 0 (default) disables checkpointing.
        #[arg(long, default_value_t = 0)]
        checkpoint_every: u64,
        /// Base directory for checkpoints (default: `ingest_checkpoints/`).
        #[arg(long)]
        checkpoint_dir: Option<PathBuf>,
        /// Resume from a checkpoint directory. "latest" picks the highest
        /// index under --checkpoint-dir. Fast-forwards past already-
        /// processed tokens before streaming.
        #[arg(long)]
        resume_from: Option<String>,
    },
    /// Active ingest: rotate probe batches among multiple sources, score each
    /// by acquisition `max(N − C, 0) / joules_per_token`, spend the next
    /// commit batch on the winner. Canonical Phase D verification: a noise
    /// source should receive < 20% of total tokens against a real corpus.
    ActiveIngest {
        #[arg(long)]
        model: PathBuf,
        /// Root seed for all RNGs (Phase G reproducibility).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// One or more files to use as sources. Each becomes a `VecSource`
        /// over its tokenized content.
        #[arg(long, value_delimiter = ',')]
        sources: Vec<PathBuf>,
        /// Also include a uniform-noise source as a pathological control.
        #[arg(long)]
        include_noise: bool,
        #[arg(long, default_value = "episodes")]
        memory_dir: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long)]
        critic: Option<PathBuf>,
        #[arg(long)]
        confidence_head: Option<PathBuf>,
        /// Tokens per probe round (each source gets this many tokens to score).
        #[arg(long, default_value_t = 100)]
        probe_tokens: usize,
        /// Tokens to commit from the winning source before the next probe.
        #[arg(long, default_value_t = 400)]
        commit_tokens: usize,
        /// Total token budget across all commit rounds.
        #[arg(long, default_value_t = 3000)]
        max_tokens: usize,
        #[arg(long)]
        metrics_out: Option<PathBuf>,
        #[arg(long, default_value_t = 500)]
        metrics_window: u64,
    },
    /// Run the full ablation bench-suite across {adaptive on/off} ×
    /// {modules present/absent} × {routers pristine/trained} × seeds and
    /// emit a TSV summary. The first real signal that B+C+F earn their
    /// keep on real data.
    BenchSuite {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        eval_corpus: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Seeds to evaluate. One row per (seed, axis-combo).
        #[arg(long, value_delimiter = ',', default_values_t = [1u64])]
        seeds: Vec<u64>,
        #[arg(long)]
        critic: Option<PathBuf>,
        #[arg(long)]
        confidence_head: Option<PathBuf>,
        #[arg(long)]
        meta_critic: Option<PathBuf>,
        /// Trained-router side-files to pair with the pristine router. Each
        /// path contributes one cell per {adaptive, modules} combination.
        #[arg(long, value_delimiter = ',')]
        routers_trained: Vec<PathBuf>,
        /// Directories of crystal modules to pair with the no-module cell.
        #[arg(long, value_delimiter = ',')]
        modules_dirs: Vec<PathBuf>,
        /// Tokens per cell's eval pass.
        #[arg(long, default_value_t = 1500)]
        max_tokens: usize,
        /// Adaptive iteration budget when adaptive=on.
        #[arg(long, default_value_t = 2)]
        adaptive_max_extra: u8,
        /// Adaptive z-threshold when adaptive=on.
        #[arg(long, default_value_t = 1.0)]
        adaptive_z: f32,
        /// Output TSV file.
        #[arg(long)]
        out: PathBuf,
    },
    /// Tail a metrics jsonl file and print the dashboard (J/nat, tokens/sec,
    /// CE, deltas vs. the first window).
    Metrics {
        /// Path to the jsonl file written by `ingest --metrics-out`.
        #[arg(long)]
        path: PathBuf,
        /// Show the most recent N windows (0 = all).
        #[arg(long, default_value_t = 20)]
        tail: usize,
    },
    /// Evaluate: compute per-token cross-entropy / perplexity on a corpus.
    /// The falsifiable test: does adding crystal modules reduce NLL?
    Eval {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Load crystal modules from this directory before evaluating.
        #[arg(long)]
        modules_dir: Option<PathBuf>,
        /// Load a trained energy critic side-file before evaluating.
        #[arg(long)]
        critic: Option<PathBuf>,
        /// Install a trained untied readout sidecar (replaces the tied
        /// unembedding). Produced by `probe-readout --save-readout`.
        #[arg(long)]
        readout: Option<PathBuf>,
        /// Cap on tokens to evaluate (0 = unlimited).
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
    },
    /// Phase F (2/2): Train MoE routers via REINFORCE-style policy gradient.
    /// Streams a corpus through the model in training mode with counterfactual
    /// sampling, computes per-token CE advantage against an EMA baseline, and
    /// applies AdamW to each MoE layer's router weights every N tokens.
    /// Rewrites the model file in place once training completes.
    TrainRouter {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Root seed for all RNGs (Phase G reproducibility).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Counterfactual sampling rate: probability of forcing top-2 over top-1.
        #[arg(long, default_value_t = 0.1)]
        cf_rate: f32,
        /// Baseline EMA decay (closer to 1.0 = slower baseline).
        #[arg(long, default_value_t = 0.98)]
        baseline_decay: f32,
        /// Adam learning rate.
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
        /// Adam weight decay.
        #[arg(long, default_value_t = 0.01)]
        weight_decay: f32,
        /// Gradient clipping (max l2 norm).
        #[arg(long, default_value_t = 1.0)]
        clip_norm: f32,
        /// Tokens per AdamW step (accumulate gradient across this many).
        #[arg(long, default_value_t = 64)]
        batch_size: usize,
        /// Cap on total training tokens (0 = full corpus).
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
        /// Output path for the updated model. Defaults to overwriting input.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Train a confidence head (Head C) online: same shape as Calibrate, but
    /// targets a separate linear probe used by adaptive decode. Keeps heads
    /// N and C independently parameterized to avoid gradient cannibalization.
    CalibrateConfidence {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Output head side-file.
        #[arg(long, default_value = "confidence.bin")]
        output: PathBuf,
        #[arg(long, default_value_t = 1e-4)]
        lr: f32,
        #[arg(long, default_value_t = 1)]
        epochs: usize,
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
        /// Root seed for all RNGs (Phase G reproducibility).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Also train a MetaCritic (Phase F target-network) alongside the
        /// confidence head, saving it to this path. Target refreshes every
        /// `--meta-refresh` steps.
        #[arg(long)]
        meta_out: Option<PathBuf>,
        /// Training steps between target-snapshot refreshes.
        #[arg(long, default_value_t = 500)]
        meta_refresh: u64,
    },
    /// Reservoir probe (IF_FOUND.md fork): fit a dense f32 readout
    /// `logits = W·h + b` over the FROZEN core's hidden states and compare
    /// held-out NLL against the unigram marginal. The gap is the contextual
    /// information the frozen random core linearly exposes. ~0 ⇒ reservoir
    /// dead, train the core (Path B); real gap ⇒ reservoir alive (Path A).
    /// Pure diagnostic — does not touch the on-disk format or the tied table.
    ProbeReadout {
        #[arg(long)]
        model: PathBuf,
        /// Training corpus (.tokens or raw text) — frozen-core features fit here.
        #[arg(long)]
        train: PathBuf,
        /// Held-out corpus (.tokens or raw text) — the deciding measurement.
        #[arg(long)]
        holdout: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Cap on training positions (0 = all). Features cache ~= n·d·4 bytes.
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
        #[arg(long, default_value_t = 6)]
        epochs: usize,
        #[arg(long, default_value_t = 512)]
        batch_size: usize,
        /// Momentum-SGD learning rate (features are standardized first).
        #[arg(long, default_value_t = 0.5)]
        lr: f32,
        /// Comma-separated L2 weight decays to sweep (features once, fit each).
        /// The verdict uses the best held-out result across the sweep, so a
        /// weak signal cannot be hidden behind a single under-regularized fit.
        #[arg(long, default_value = "0.0001,0.001,0.01,0.1")]
        weight_decays: String,
        /// Root seed for the shuffle RNG (Phase G reproducibility).
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Optional JSON report path (a sidecar manifest is written beside it).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Persist the best-fitted readout as an installable sidecar (use with
        /// `ingest`/`eval` via `--readout`). Standardization is folded in.
        #[arg(long)]
        save_readout: Option<PathBuf>,
    },
    /// Train the energy critic online: stream a corpus, SGD the critic on
    /// MSE against per-step NLL, save it to a side file. No episode storage,
    /// no module installation. Separate from ingest by design.
    Calibrate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Output critic side-file.
        #[arg(long, default_value = "critic.bin")]
        output: PathBuf,
        /// Learning rate.
        #[arg(long, default_value_t = 1e-4)]
        lr: f32,
        /// Number of passes through the corpus.
        #[arg(long, default_value_t = 1)]
        epochs: usize,
        /// Cap on tokens per epoch (0 = unlimited).
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
    },
    /// Run one crystallization cycle over stored episodes; write new modules to disk.
    /// Supply --model to use gradient-aligned correction (recommended).
    /// Supply --d-model alone to fall back to hidden-centroid correction (weaker, ignores actual_token).
    Crystal {
        #[arg(long, default_value = "episodes")]
        memory_dir: PathBuf,
        #[arg(long, default_value = "modules")]
        modules_dir: PathBuf,
        /// Model that produced these episodes (enables tied-unembed correction direction).
        #[arg(long)]
        model: Option<PathBuf>,
        /// d_model when `--model` is not provided.
        #[arg(long)]
        d_model: Option<usize>,
        /// Number of clusters (k) for k-means.
        #[arg(long, default_value_t = 8)]
        n_clusters: usize,
    },
    /// Compile crystallized modules to native code (demo: ELF emission).
    Compile {
        #[arg(long, default_value = "episodes")]
        memory_dir: PathBuf,
        #[arg(long, default_value = "modules")]
        modules_dir: PathBuf,
    },
    /// Package a spore for propagation.
    Spore {
        /// Path to the kernel binary.
        #[arg(long, default_value = "target/release/clob")]
        binary: PathBuf,
        /// Path to the seed model.
        #[arg(long, default_value = "seed.clob")]
        model: PathBuf,
        /// Output spore path.
        #[arg(long, default_value = "kernel.spore")]
        output: PathBuf,
    },
    /// Network peer management.
    Net {
        #[command(subcommand)]
        action: NetAction,
    },
}

#[derive(Subcommand)]
enum NetAction {
    /// Discover peers on the local network.
    Discover {
        #[arg(long, default_value_t = 3000)]
        timeout_ms: u64,
    },
    /// Connect to a specific peer.
    Connect {
        #[arg(long)]
        addr: String,
        /// Model whose config fingerprints this kernel in the handshake.
        /// Without it the handshake is anonymous (id/hash = 0).
        #[arg(long)]
        model: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Boot { memory_dir, modules_dir, model, config, seed, memory_capacity, tokenizer, listen, peer } => {
            cmd_boot(&memory_dir, &modules_dir, &model, &config, seed, memory_capacity, &tokenizer, &listen, &peer);
        }
        Commands::Think { model, max_tokens, temperature, tokenizer, modules_dir } => {
            cmd_think(&model, max_tokens, temperature, &tokenizer, &modules_dir);
        }
        Commands::Synth { output, config, seed } => {
            cmd_synth(&output, &config, seed);
        }
        Commands::Bench { config, seq_len, decode_steps } => {
            cmd_bench(&config, seq_len, decode_steps);
        }
        Commands::Info { model } => {
            cmd_info(&model);
        }
        Commands::Status { memory_dir } => {
            cmd_status(&memory_dir);
        }
        Commands::TrainTokenizer { corpus, output, n_merges } => {
            cmd_train_tokenizer(&corpus, &output, n_merges);
        }
        Commands::Encode { input, output, tokenizer } => {
            cmd_encode(&input, &output, &tokenizer);
        }
        Commands::Ingest {
            model, input, memory_dir, tokenizer, modules_dir, critic, max_tokens,
            metrics_out, metrics_window, confidence_head, meta_critic, routers,
            adaptive_compute, adaptive_max_extra, adaptive_z, meta_unreliable_threshold,
            checkpoint_every, checkpoint_dir, resume_from,
        } => {
            let adaptive = AdaptiveConfig {
                enabled: adaptive_compute,
                max_extra_steps: adaptive_max_extra,
                z_threshold: adaptive_z,
                meta_unreliable_threshold,
            };
            cmd_ingest(
                &model, &input, &memory_dir, &tokenizer, &modules_dir, &critic,
                max_tokens, &metrics_out, metrics_window, &confidence_head,
                &meta_critic, &routers, adaptive,
                checkpoint_every, &checkpoint_dir, &resume_from,
            );
        }
        Commands::ActiveIngest {
            model, seed, sources, include_noise, memory_dir, tokenizer, critic,
            confidence_head, probe_tokens, commit_tokens, max_tokens,
            metrics_out, metrics_window,
        } => {
            cmd_active_ingest(
                &model, seed, &sources, include_noise, &memory_dir, &tokenizer,
                &critic, &confidence_head, probe_tokens, commit_tokens,
                max_tokens, &metrics_out, metrics_window,
            );
        }
        Commands::Metrics { path, tail } => {
            cmd_metrics(&path, tail);
        }
        Commands::BenchSuite {
            model, eval_corpus, tokenizer, seeds, critic, confidence_head,
            meta_critic, routers_trained, modules_dirs, max_tokens,
            adaptive_max_extra, adaptive_z, out,
        } => {
            cmd_bench_suite(
                &model, &eval_corpus, &tokenizer, &seeds,
                &critic, &confidence_head, &meta_critic,
                &routers_trained, &modules_dirs,
                max_tokens, adaptive_max_extra, adaptive_z, &out,
            );
        }
        Commands::TrainRouter {
            model, corpus, tokenizer, seed, cf_rate, baseline_decay, lr, weight_decay,
            clip_norm, batch_size, max_tokens, output,
        } => {
            cmd_train_router(
                &model, &corpus, &tokenizer, seed, cf_rate, baseline_decay, lr,
                weight_decay, clip_norm, batch_size, max_tokens, &output,
            );
        }
        Commands::CalibrateConfidence {
            model, corpus, tokenizer, output, lr, epochs, max_tokens, seed, meta_out, meta_refresh,
        } => {
            cmd_calibrate_confidence(
                &model, &corpus, &tokenizer, &output, lr, epochs, max_tokens,
                seed, &meta_out, meta_refresh,
            );
        }
        Commands::ProbeReadout {
            model, train, holdout, tokenizer, max_tokens, epochs, batch_size,
            lr, weight_decays, seed, out, save_readout,
        } => {
            cmd_probe_readout(
                &model, &train, &holdout, &tokenizer, max_tokens, epochs,
                batch_size, lr, &weight_decays, seed, &out, &save_readout,
            );
        }
        Commands::Eval { model, corpus, tokenizer, modules_dir, critic, readout, max_tokens } => {
            cmd_eval(&model, &corpus, &tokenizer, &modules_dir, &critic, &readout, max_tokens);
        }
        Commands::Calibrate { model, corpus, tokenizer, output, lr, epochs, max_tokens } => {
            cmd_calibrate(&model, &corpus, &tokenizer, &output, lr, epochs, max_tokens);
        }
        Commands::Crystal { memory_dir, modules_dir, model, d_model, n_clusters } => {
            match (model, d_model) {
                (Some(m), _) => cmd_crystal_with_model(&memory_dir, &modules_dir, &m, n_clusters),
                (None, Some(d)) => cmd_crystal(&memory_dir, &modules_dir, d, n_clusters),
                (None, None) => {
                    eprintln!("[crystal] provide --model (preferred) or --d-model");
                    std::process::exit(2);
                }
            }
        }
        Commands::Compile { memory_dir: _, modules_dir } => {
            cmd_compile(&modules_dir);
        }
        Commands::Spore { binary, model, output } => {
            cmd_spore(&binary, &model, &output);
        }
        Commands::Net { action } => {
            match action {
                NetAction::Discover { timeout_ms } => cmd_net_discover(timeout_ms),
                NetAction::Connect { addr, model } => cmd_net_connect(&addr, &model),
            }
        }
    }
}

fn parse_config(name: &str) -> KernelConfig {
    match name {
        "tiny" => KernelConfig::tiny(),
        "small" => KernelConfig::small(),
        "seed" => KernelConfig::seed(),
        other => {
            eprintln!("[kernel] Unknown config '{}'. Use 'tiny', 'small', or 'seed'.", other);
            KernelConfig::tiny()
        }
    }
}

fn load_tokenizer(path: &Option<PathBuf>) -> BpeTokenizer {
    match path {
        Some(p) if p.exists() => {
            match BpeTokenizer::load(p) {
                Ok(tok) => {
                    eprintln!("[token] Loaded tokenizer from {:?} (vocab={})", p, tok.vocab_size());
                    tok
                }
                Err(e) => {
                    eprintln!("[token] Failed to load {:?}: {}. Using byte-level.", p, e);
                    BpeTokenizer::byte_level()
                }
            }
        }
        _ => {
            eprintln!("[token] Using byte-level tokenizer (vocab=260)");
            BpeTokenizer::byte_level()
        }
    }
}

fn format_params(count: usize) -> String {
    if count >= 1_000_000_000 { format!("{:.1}B", count as f64 / 1e9) }
    else if count >= 1_000_000 { format!("{:.1}M", count as f64 / 1e6) }
    else if count >= 1_000 { format!("{:.1}K", count as f64 / 1e3) }
    else { format!("{}", count) }
}

fn now_nanos() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

/// Hard-fail if the tokenizer produces ids the model can't represent.
/// No silent clamping — a collapsed vocab is a poisoned training signal.
fn assert_vocab_compatible(tokenizer: &BpeTokenizer, config: &KernelConfig) {
    if tokenizer.vocab_size() > config.vocab_size {
        eprintln!(
            "[error] tokenizer vocab ({}) > model vocab ({}). Retrain the tokenizer with \
             --n-merges that keep vocab ≤ {}, or synth a model with matching vocab.",
            tokenizer.vocab_size(), config.vocab_size, config.vocab_size,
        );
        std::process::exit(2);
    }
}

/// Load crystal modules from a directory into the model. Returns count loaded.
fn load_modules_into(model: &mut CoreModel, dir: &Path) -> usize {
    match store::load_modules(dir) {
        Ok(modules) => {
            let mut ok = 0;
            let mut skipped = 0;
            for m in modules {
                if m.d_model != model.config.d_model {
                    eprintln!(
                        "[modules] skipping id={} (d_model {} != {})",
                        m.id, m.d_model, model.config.d_model,
                    );
                    skipped += 1;
                    continue;
                }
                model.push_crystal_module(m);
                ok += 1;
            }
            if skipped > 0 {
                eprintln!("[modules] loaded {} ({} skipped) from {:?}", ok, skipped, dir);
            } else {
                eprintln!("[modules] loaded {} from {:?}", ok, dir);
            }
            ok
        }
        Err(e) => {
            eprintln!("[modules] {:?}: {}", dir, e);
            0
        }
    }
}

/// Load a trained energy critic from a side file and install it into the model.
/// Hard-fails on dim mismatch. No-op if the file doesn't exist.
fn apply_critic_file(model: &mut CoreModel, path: &Path) -> bool {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[critic] {:?}: {}", path, e);
            return false;
        }
    };
    let critic = match EnergyCritic::from_bytes(&bytes) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[critic] parse {:?}: {}", path, e);
            return false;
        }
    };
    if critic.dim() != model.config.d_model {
        eprintln!("[critic] dim {} != model d_model {}; refusing to install",
            critic.dim(), model.config.d_model);
        std::process::exit(2);
    }
    eprintln!("[critic] loaded {:?} (dim={}, n_trained={}, mse={:.4})",
        path, critic.dim(), critic.n_trained(), critic.train_mse());
    model.replace_energy_critic(critic);
    true
}

fn apply_confidence_head_file(model: &mut CoreModel, path: &Path) -> bool {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[conf] {:?}: {}", path, e);
            return false;
        }
    };
    let head = match ConfidenceHead::from_bytes(&bytes) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[conf] parse {:?}: {}", path, e);
            return false;
        }
    };
    if head.dim() != model.config.d_model {
        eprintln!("[conf] dim {} != model d_model {}; refusing to install",
            head.dim(), model.config.d_model);
        std::process::exit(2);
    }
    eprintln!("[conf] loaded {:?} (dim={}, n_trained={}, mse={:.4})",
        path, head.dim(), head.n_trained(), head.train_mse());
    model.set_confidence_head(head);
    true
}

fn apply_readout_file(model: &mut CoreModel, path: &Path) -> bool {
    use clob::nn::readout::TrainedReadout;
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[readout] {:?}: {}", path, e);
            return false;
        }
    };
    let readout = match TrainedReadout::from_bytes(&bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[readout] parse {:?}: {}", path, e);
            return false;
        }
    };
    if readout.d_model() != model.config.d_model || readout.vocab_size() != model.config.vocab_size {
        eprintln!("[readout] shape {}×{} != model {}×{}; refusing to install",
            readout.vocab_size(), readout.d_model(), model.config.vocab_size, model.config.d_model);
        std::process::exit(2);
    }
    eprintln!("[readout] installed untied output head from {:?} ({}×{})",
        path, readout.vocab_size(), readout.d_model());
    model.set_readout(readout);
    true
}

fn apply_meta_critic_file(model: &mut CoreModel, path: &Path) -> bool {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[meta] {:?}: {}", path, e);
            return false;
        }
    };
    let meta = match MetaCritic::from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[meta] parse {:?}: {}", path, e);
            return false;
        }
    };
    if meta.dim() != model.config.d_model {
        eprintln!("[meta] dim {} != model d_model {}; refusing to install",
            meta.dim(), model.config.d_model);
        std::process::exit(2);
    }
    eprintln!("[meta] loaded {:?} (dim={}, n_trained={}, mse={:.4}, refresh_period={})",
        path, meta.dim(), meta.n_trained(), meta.train_mse(), meta.refresh_period());
    model.set_meta_critic(meta);
    true
}

/// Softmax-probability of a single class from raw logits.
/// Numerically stable (max-subtracted). Returns 0 if token id is out of range.
fn prob_of_token(logits: &[f32], token: u32) -> f32 {
    let tok = token as usize;
    if tok >= logits.len() { return 0.0; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits { sum += (l - max).exp(); }
    if sum <= 0.0 { return 0.0; }
    ((logits[tok] - max).exp()) / sum
}

// ============================================================================
// Commands
// ============================================================================

fn cmd_boot(
    memory_dir: &PathBuf, modules_dir: &PathBuf, model_path: &PathBuf,
    config_str: &str, seed: u64, memory_capacity: usize,
    tokenizer_path: &Option<PathBuf>, _listen: &Option<String>, _peers: &[String],
) {
    let config = parse_config(config_str);
    let tokenizer = load_tokenizer(tokenizer_path);
    assert_vocab_compatible(&tokenizer, &config);

    eprintln!("[kernel] Config: d_model={}, n_layers={}, vocab={}, params=~{}",
        config.d_model, config.n_layers, config.vocab_size, format_params(config.param_count_estimate()));

    let mut model = if model_path.exists() {
        eprintln!("[kernel] Loading model from {:?}", model_path);
        io::loader::load_model(model_path).expect("failed to load model")
    } else {
        eprintln!("[kernel] Creating synthetic seed model");
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let model = CoreModel::random(config.clone(), &mut rng);
        io::synth::generate_synthetic(&config, model_path, &mut rng)
            .expect("failed to write seed model");
        eprintln!("[kernel] Saved seed model to {:?}", model_path);
        model
    };

    if modules_dir.exists() {
        load_modules_into(&mut model, modules_dir);
    }

    let memory = EpisodicMemory::open(memory_dir, memory_capacity)
        .expect("failed to open episodic memory");
    let mut crystal_engine = CrystallizationEngine::new(config.d_model, CrystalConfig::default());

    eprintln!("\n=== THE KERNEL IS ALIVE ===\n");
    eprintln!("Commands: :status, :crystal, :reset, :quit");
    eprintln!("Modules installed: {}\n", model.n_crystal_modules());

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let sampling = SamplingConfig { temperature: 0.7, top_k: 50, top_p: 0.9 };
    let mut context_tokens: Vec<u32> = Vec::new();
    let mut step_count: u64 = 0;

    loop {
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() { break; }
        let input = input.trim();
        if input.is_empty() { continue; }

        match input {
            ":quit" | ":q" => { eprintln!("[kernel] Shutting down."); break; }
            ":status" => {
                let mem_stats = memory.stats();
                eprintln!("[status] Steps: {} | Context: {} tokens", step_count, context_tokens.len());
                eprintln!("[status] Memory: {}/{} ({} unconsumed, {:.2} MB)",
                    mem_stats.total_episodes, mem_stats.capacity,
                    mem_stats.unconsumed, mem_stats.total_bytes as f64 / (1024.0 * 1024.0));
                eprintln!("[status] Crystal (session): {}", crystal_engine.stats());
                eprintln!("[status] Installed modules: {}", model.n_crystal_modules());
                continue;
            }
            ":crystal" => {
                eprintln!("[crystal] Running crystallization cycle...");
                let before = crystal_engine.n_modules();
                let n = {
                    // Separate the embedding borrow from the mutable push below.
                    let embed_snapshot: Vec<f32> = model.embed_table().to_vec();
                    crystal_engine.cycle(&memory, Some(&embed_snapshot), model.config.vocab_size)
                };
                let after = crystal_engine.n_modules();
                let new_modules: Vec<_> = crystal_engine.modules()[before..after].to_vec();
                for module in new_modules {
                    if let Ok(path) = store::save_module(&module, modules_dir) {
                        eprintln!("[crystal] saved {:?}", path);
                    }
                    model.push_crystal_module(module);
                }
                eprintln!("[crystal] {} new modules (session total {}, installed {})",
                    n, crystal_engine.n_modules(), model.n_crystal_modules());
                continue;
            }
            ":reset" => {
                model.reset_state();
                context_tokens.clear();
                eprintln!("[kernel] State reset.");
                continue;
            }
            _ => {}
        }

        let input_tokens: Vec<u32> = tokenizer.encode(input);
        let start = Instant::now();
        let _logits = model.prefill(&input_tokens);
        context_tokens.extend_from_slice(&input_tokens);

        let (energy, is_novel) = model.energy_score_and_detect();

        let generated = generate::generate(&mut model, &[], 64, &sampling, &mut rng);
        let elapsed = start.elapsed();

        let output = tokenizer.decode(&generated);
        context_tokens.extend_from_slice(&generated);
        step_count += 1;

        println!("{}", output);
        eprintln!("  [{:.1}ms | {:.0} tok/s | energy={:.4} {} | modules={}]",
            elapsed.as_secs_f64() * 1000.0,
            generated.len() as f64 / elapsed.as_secs_f64(),
            energy,
            if is_novel { "▲" } else { "○" },
            model.n_crystal_modules());
    }
}

fn cmd_think(
    model_path: &PathBuf, max_tokens: usize, temperature: f32,
    tokenizer_path: &Option<PathBuf>, modules_dir: &Option<PathBuf>,
) {
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[kernel] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model.config);
    if let Some(dir) = modules_dir.as_ref() {
        load_modules_into(&mut model, dir);
    }
    let config = model.config.clone();
    eprintln!("[kernel] d={}, L={}, V={}, modules={}",
        config.d_model, config.n_layers, config.vocab_size, model.n_crystal_modules());

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let sampling = SamplingConfig { temperature, top_k: 50, top_p: 0.9 };

    loop {
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() { break; }
        let input = input.trim();
        if input == ":quit" || input == ":q" { break; }
        if input.is_empty() { continue; }

        let tokens: Vec<u32> = tokenizer.encode(input);
        let start = Instant::now();
        let _logits = model.prefill(&tokens);
        let generated = generate::generate(&mut model, &[], max_tokens, &sampling, &mut rng);
        let elapsed = start.elapsed();

        println!("{}", tokenizer.decode(&generated));
        eprintln!("  [{:.1}ms | {:.0} tok/s | energy={:.4}]",
            elapsed.as_secs_f64() * 1000.0,
            generated.len() as f64 / elapsed.as_secs_f64(),
            model.energy_score());
        model.reset_state();
    }
}

fn cmd_synth(output: &PathBuf, config_str: &str, seed: u64) {
    let config = parse_config(config_str);
    eprintln!("[synth] Config: d={}, L={}, V={}, params=~{}",
        config.d_model, config.n_layers, config.vocab_size, format_params(config.param_count_estimate()));
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    io::synth::generate_synthetic(&config, output, &mut rng).expect("failed to generate");
    let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    eprintln!("[synth] Done: {:?} ({:.2} MB)", output, file_size as f64 / (1024.0 * 1024.0));
    let manifest = clob::config::manifest::RunManifest::new("synth").with_seed(seed);
    if let Err(e) = manifest.save_sidecar(output) {
        eprintln!("[synth] manifest write failed: {}", e);
    }
}

fn cmd_bench(config_str: &str, seq_len: usize, decode_steps: usize) {
    let config = parse_config(config_str);
    eprintln!("[bench] d={}, L={}, V={}, params=~{}",
        config.d_model, config.n_layers, config.vocab_size, format_params(config.param_count_estimate()));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut model = CoreModel::random(config.clone(), &mut rng);
    let prompt: Vec<u32> = (0..seq_len).map(|_| rng.gen_range(0..config.vocab_size) as u32).collect();

    eprintln!("\n--- Prefill ({} tokens) ---", seq_len);
    let start = Instant::now();
    let _ = model.prefill(&prompt);
    let prefill = start.elapsed();
    eprintln!("  TTFT: {:.2} ms ({:.1} tok/s)", prefill.as_secs_f64() * 1000.0, seq_len as f64 / prefill.as_secs_f64());

    eprintln!("\n--- Decode ({} steps) ---", decode_steps);
    let sampling = SamplingConfig::default();
    let start = Instant::now();
    let tokens = generate::generate(&mut model, &[], decode_steps, &sampling, &mut rng);
    let decode = start.elapsed();
    let tok_s = tokens.len() as f64 / decode.as_secs_f64();
    eprintln!("  Throughput: {:.1} tok/s ({:.2} ms/tok)", tok_s, decode.as_secs_f64() * 1000.0 / tokens.len() as f64);

    #[cfg(target_os = "linux")]
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            eprintln!("  Peak RSS: {} MB", ru.ru_maxrss / 1024);
        }
    }
    eprintln!("\n  TTFT: {:.2} ms | Decode: {:.1} tok/s", prefill.as_secs_f64() * 1000.0, tok_s);
}

fn cmd_info(model_path: &PathBuf) {
    let file = std::fs::File::open(model_path).expect("failed to open");
    let mut reader = std::io::BufReader::new(file);
    let (header, header_size) = clob::io::format::read_header(&mut reader).expect("failed to read");
    let file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    let c = &header.config;

    println!("=== The Kernel ===");
    println!("File: {:?} ({:.2} MB)", model_path, file_size as f64 / (1024.0 * 1024.0));
    println!("  d_model={}, n_layers={}, n_heads={}, d_state={}", c.d_model, c.n_layers, c.n_heads, c.d_state);
    println!("  d_inner={}, n_experts={}, n_active={}", c.d_inner, c.n_experts, c.n_active_experts);
    println!("  vocab={}, moe_layers={:?}", c.vocab_size, c.moe_layers);
    println!("  ~{} params, {} weight entries, {:.2} MB tensors",
        format_params(c.param_count_estimate()), header.weight_entries.len(),
        (file_size as usize - header_size) as f64 / (1024.0 * 1024.0));
}

fn cmd_status(memory_dir: &PathBuf) {
    match EpisodicMemory::open(memory_dir, usize::MAX) {
        Ok(memory) => {
            let s = memory.stats();
            println!("=== Episodic Memory ===");
            println!("  Dir: {:?}", memory_dir);
            println!("  Episodes: {} ({} unconsumed, {} consumed)", s.total_episodes, s.unconsumed, s.consumed);
            println!("  Total written: {} | Size: {:.2} MB", s.total_ever_written, s.total_bytes as f64 / (1024.0 * 1024.0));
        }
        Err(e) => eprintln!("[error] {}", e),
    }
}

fn cmd_train_tokenizer(corpus_path: &PathBuf, output: &PathBuf, n_merges: usize) {
    eprintln!("[bpe] Reading corpus from {:?}", corpus_path);
    let corpus = std::fs::read_to_string(corpus_path).expect("failed to read corpus");
    eprintln!("[bpe] Corpus: {} bytes, {} words", corpus.len(), corpus.split_whitespace().count());

    let tok = clob::token::trainer::train_bpe(&corpus, n_merges);
    tok.save(output).expect("failed to save tokenizer");

    let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    eprintln!("[bpe] Saved: {:?} (vocab={}, {:.2} KB)", output, tok.vocab_size(), file_size as f64 / 1024.0);

    // Test encode/decode roundtrip
    let sample = &corpus[..corpus.len().min(100)];
    let encoded = tok.encode(sample);
    let decoded = tok.decode(&encoded);
    let compression = encoded.len() as f64 / sample.len() as f64;
    eprintln!("[bpe] Sample: \"{}...\" → {} tokens (compression={:.2}x)", &sample[..sample.len().min(40)], encoded.len(), 1.0/compression);
    assert_eq!(decoded, sample, "roundtrip failed!");
    eprintln!("[bpe] Roundtrip: ✓");
}

fn cmd_encode(input_path: &PathBuf, output: &PathBuf, tokenizer_path: &Option<PathBuf>) {
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[encode] Reading corpus from {:?}", input_path);
    let text = std::fs::read_to_string(input_path).expect("failed to read corpus");
    let tokens = tokenizer.encode(&text);
    clob::token::tokens_file::save_tokens(output, &tokens, tokenizer.vocab_size() as u32)
        .expect("failed to write .tokens");
    let bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "[encode] {} chars -> {} tokens -> {:?} ({:.1} KB, {:.2} chars/token)",
        text.len(), tokens.len(), output,
        bytes as f64 / 1024.0,
        text.len() as f64 / tokens.len().max(1) as f64,
    );
}

/// Load a token stream for the runtime: a pre-encoded `.tokens` cache when
/// `input_path` ends in `.tokens` (Phase L — skips BPE entirely), otherwise
/// read the file as text and tokenize it on the fly. For a cache, the stamped
/// vocab is checked against the model so ids that can't index the model fail
/// loudly rather than reading past the embedding table.
fn load_or_encode_tokens(
    input_path: &PathBuf,
    tokenizer: &BpeTokenizer,
    config: &KernelConfig,
) -> Vec<u32> {
    if input_path.extension().and_then(|e| e.to_str()) == Some("tokens") {
        eprintln!("[ingest] Loading pre-encoded tokens from {:?}", input_path);
        let tf = clob::token::tokens_file::load_tokens(input_path).expect("failed to read .tokens");
        if tf.vocab_size as usize > config.vocab_size {
            eprintln!(
                "[error] .tokens was encoded with vocab {} but model vocab is {}. \
                 Re-encode against a tokenizer/model that match.",
                tf.vocab_size, config.vocab_size,
            );
            std::process::exit(2);
        }
        tf.tokens
    } else {
        eprintln!("[ingest] Tokenizing {:?}", input_path);
        let text = std::fs::read_to_string(input_path).expect("failed to read corpus");
        tokenizer.encode(&text)
    }
}

fn cmd_ingest(
    model_path: &PathBuf, input_path: &PathBuf, memory_dir: &PathBuf,
    tokenizer_path: &Option<PathBuf>, modules_dir: &Option<PathBuf>,
    critic_path: &Option<PathBuf>, max_tokens: usize,
    metrics_out: &Option<PathBuf>, metrics_window: u64,
    confidence_head_path: &Option<PathBuf>, meta_critic_path: &Option<PathBuf>,
    routers_path: &Option<PathBuf>, adaptive: AdaptiveConfig,
    checkpoint_every: u64, checkpoint_dir: &Option<PathBuf>,
    resume_from: &Option<String>,
) {
    use clob::io::checkpoint;
    let ckpt_base = checkpoint_dir.clone()
        .unwrap_or_else(|| PathBuf::from("ingest_checkpoints"));
    let mut resumed_tokens: u64 = 0;
    let mut ckpt_index: u64 = 1;
    let mut effective_model_path: Option<PathBuf> = None;
    if let Some(spec) = resume_from.as_ref() {
        match checkpoint::resolve_resume(&ckpt_base, spec)
            .and_then(|d| checkpoint::load_handles(&d))
        {
            Ok(handles) => {
                resumed_tokens = handles.progress.tokens_processed;
                // Resume from the next checkpoint index so we don't collide.
                if let Some(name) = handles.dir.file_name().and_then(|n| n.to_str()) {
                    if let Some(n) = name.strip_prefix("checkpoint_").and_then(|s| s.parse::<u64>().ok()) {
                        ckpt_index = n + 1;
                    }
                }
                effective_model_path = Some(handles.model.clone());
                eprintln!(
                    "[ingest] resuming from {:?}: skipping {} already-processed tokens",
                    handles.dir, resumed_tokens,
                );
            }
            Err(e) => {
                eprintln!("[ingest] --resume-from failed: {}", e);
                std::process::exit(2);
            }
        }
    }
    let tokenizer = load_tokenizer(tokenizer_path);
    let load_path = effective_model_path.as_ref().unwrap_or(model_path);
    eprintln!("[ingest] Loading model from {:?}", load_path);
    let mut model = io::loader::load_model(load_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model.config);

    if let Some(p) = critic_path.as_ref() {
        apply_critic_file(&mut model, p);
    }
    if let Some(p) = confidence_head_path.as_ref() {
        apply_confidence_head_file(&mut model, p);
    }
    if let Some(p) = meta_critic_path.as_ref() {
        apply_meta_critic_file(&mut model, p);
    }
    if let Some(p) = routers_path.as_ref() {
        apply_routers_file(&mut model, p);
    }
    model.adaptive_config = adaptive;
    if adaptive.enabled && !model.has_confidence_head() {
        eprintln!("[ingest] WARN: --adaptive-compute requested but no --confidence-head loaded; adaptive will not fire.");
    }
    if let Some(dir) = modules_dir.as_ref() {
        load_modules_into(&mut model, dir);
    }

    let memory = EpisodicMemory::open(memory_dir, 100_000).expect("failed to open memory");
    let mut novelty_detector = NoveltyDetector::new();

    // Per-window metrics sink. Enabled only when --metrics-out is provided.
    let mut metrics = match metrics_out {
        Some(path) => match clob::metrics::JsonlWriter::create(path) {
            Ok(writer) => {
                eprintln!("[metrics] Writing windows of {} tokens to {:?}", metrics_window, path);
                Some(MetricsSink::new(writer, metrics_window))
            }
            Err(e) => {
                eprintln!("[metrics] failed to open {:?}: {} — continuing without metrics", path, e);
                None
            }
        },
        None => None,
    };
    let mut meter = clob::metrics::TdpProxyMeter::t490();

    let tokens = load_or_encode_tokens(input_path, &tokenizer, &model.config);
    let limit = if max_tokens == 0 { tokens.len() } else { tokens.len().min(max_tokens) };
    eprintln!("[ingest] Streaming {} tokens (of {} total)", limit, tokens.len());

    let mut total_nll = 0.0f64;
    let mut n_predicted = 0usize;
    let mut novel_count = 0usize;
    let mut module_activations = 0usize;
    let start = Instant::now();

    // Stream through decode_step one token at a time. Predict token[i+1] from state after token[i].
    // When resuming, skip the first `resumed_tokens` positions — we already
    // processed them in a prior run. The model state is restored from the
    // checkpoint's model file, so the SSM state matches where we left off
    // as far as the saved weights represent it.
    model.reset_state();
    let mut last_tick = Instant::now();
    let start_i = resumed_tokens as usize;
    if start_i > 0 {
        eprintln!("[ingest] fast-forwarding past {} already-processed tokens", start_i);
    }
    for i in start_i..limit.saturating_sub(1) {
        let step_start = last_tick;
        let logits = model.decode_step(tokens[i]);
        let actual = tokens[i + 1];
        let p_actual = prob_of_token(logits.data(), actual);
        let nll = -((p_actual.max(1e-12)) as f64).ln();
        total_nll += nll;
        n_predicted += 1;

        let (energy, _) = model.energy_score_and_detect();
        let novelty = novelty_detector.evaluate(energy);

        if novelty.is_novel {
            // Capture top-k predictions for episode
            let top_k: Vec<(u32, f32)> = {
                let data = logits.data();
                let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = data.iter().map(|l| (l - max).exp()).sum();
                let mut indexed: Vec<(u32, f32)> = data.iter().enumerate()
                    .map(|(i, &l)| (i as u32, (l - max).exp() / sum.max(1e-12)))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.truncate(10);
                indexed
            };
            // Context: up to 32 tokens ending at tokens[i]
            let ctx_start = i.saturating_sub(31);
            let context: Vec<u32> = tokens[ctx_start..=i].to_vec();
            let episode = Episode::new(
                now_nanos(), context,
                model.last_hidden().data().to_vec(),
                energy, top_k, actual,
            );
            if let Err(e) = memory.store(&episode) {
                eprintln!("[ingest] store error: {}", e);
            }
            novel_count += 1;
        }

        if model.n_crystal_modules() > 0 {
            module_activations += model.n_crystal_modules();
        }

        // Metrics: record this step and flush a window if we've crossed the boundary.
        let extra_this_step = model.take_extra_steps();
        let meta_suppressed_this_step = model.take_meta_suppressed();
        if let Some(sink) = metrics.as_mut() {
            let (joules, now) = meter.joules_since(step_start);
            let nanos = now.duration_since(step_start).as_nanos();
            last_tick = now;
            sink.record(nll, joules, nanos, novelty.is_novel);
            for _ in 0..extra_this_step {
                sink.record_extra_step();
            }
            for _ in 0..meta_suppressed_this_step {
                sink.record_meta_suppressed();
            }
            sink.maybe_flush();
        } else {
            last_tick = Instant::now();
        }

        // Checkpoint: every checkpoint_every tokens processed in this run,
        // snapshot the model + progress marker.
        if checkpoint_every > 0 {
            let processed_this_run = (i - start_i + 1) as u64;
            if processed_this_run % checkpoint_every == 0 {
                let mut progress = clob::io::checkpoint::CheckpointProgress::new(
                    "ingest", 0,
                );
                progress.tokens_processed = (i + 1) as u64;
                let manifest = clob::config::manifest::RunManifest::new("ingest")
                    .with_input("corpus", input_path);
                match clob::io::checkpoint::save(
                    &ckpt_base, ckpt_index, &model,
                    None, None, None, &progress, &manifest,
                ) {
                    Ok(dir) => {
                        eprintln!("[ingest] checkpoint {:?} (tokens={})", dir, i + 1);
                        ckpt_index += 1;
                    }
                    Err(e) => eprintln!("[ingest] checkpoint save failed: {}", e),
                }
            }
        }
    }
    if let Some(sink) = metrics.as_mut() {
        sink.flush_final();
    }

    let elapsed = start.elapsed();
    let mean_nll = if n_predicted > 0 { total_nll / n_predicted as f64 } else { 0.0 };
    let perplexity = mean_nll.exp();
    let mem_stats = memory.stats();
    eprintln!(
        "[ingest] Done: {} predictions, mean NLL={:.4}, perplexity={:.2}, {:.1} tok/s",
        n_predicted, mean_nll, perplexity,
        n_predicted as f64 / elapsed.as_secs_f64(),
    );
    eprintln!(
        "[ingest] Novel: {} episodes stored. Memory: {} episodes ({:.2} MB)",
        novel_count, mem_stats.total_episodes,
        mem_stats.total_bytes as f64 / (1024.0 * 1024.0),
    );
    if model.n_crystal_modules() > 0 {
        eprintln!("[ingest] Crystal modules active: {} (total potential activations: {})",
            model.n_crystal_modules(), module_activations);
    }
}

/// Owns the window + jsonl writer and handles boundary flushing. Lives in
/// cmd_ingest today; future phases will share this helper across subcommands.
struct MetricsSink {
    writer: clob::metrics::JsonlWriter,
    window: clob::metrics::MetricsWindow,
    window_size: u64,
    next_idx: u64,
}

impl MetricsSink {
    fn new(writer: clob::metrics::JsonlWriter, window_size: u64) -> Self {
        Self { writer, window: clob::metrics::MetricsWindow::new(0), window_size, next_idx: 1 }
    }

    fn record(&mut self, nll: f64, joules: f64, nanos: u128, novel: bool) {
        self.window.record_step(nll, joules, nanos);
        if novel {
            self.window.record_novel();
        }
    }

    fn record_extra_step(&mut self) {
        self.window.record_extra_step();
    }

    fn record_meta_suppressed(&mut self) {
        self.window.record_meta_suppressed();
    }

    fn maybe_flush(&mut self) {
        if self.window.tokens() >= self.window_size {
            self.flush();
        }
    }

    fn flush(&mut self) {
        let snap = self.window.snapshot_and_reset();
        self.window = clob::metrics::MetricsWindow::new(self.next_idx);
        self.next_idx += 1;
        if let Err(e) = self.writer.append(&snap) {
            eprintln!("[metrics] write error: {}", e);
        }
    }

    fn flush_final(&mut self) {
        if self.window.tokens() > 0 {
            self.flush();
        }
    }
}

fn cmd_active_ingest(
    model_path: &PathBuf, seed: u64, source_paths: &[PathBuf], include_noise: bool,
    memory_dir: &PathBuf, tokenizer_path: &Option<PathBuf>,
    critic_path: &Option<PathBuf>, confidence_head_path: &Option<PathBuf>,
    probe_tokens: usize, commit_tokens: usize, max_tokens: usize,
    metrics_out: &Option<PathBuf>, metrics_window: u64,
) {
    let seed_tree = clob::util::seed::SeedTree::new(seed);
    use clob::perceive::active::{score_probe, DataSource, NoiseSource, VecSource};

    if source_paths.is_empty() && !include_noise {
        eprintln!("[active] error: supply at least one --sources path or --include-noise");
        std::process::exit(2);
    }

    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[active] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model.config);

    if let Some(p) = critic_path.as_ref() {
        apply_critic_file(&mut model, p);
    }
    if let Some(p) = confidence_head_path.as_ref() {
        apply_confidence_head_file(&mut model, p);
    }

    let memory = EpisodicMemory::open(memory_dir, 100_000).expect("failed to open memory");

    // Build sources from file paths + optional noise.
    let mut sources: Vec<Box<dyn DataSource>> = Vec::new();
    for path in source_paths {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[active] skipping {:?}: {}", path, e);
                continue;
            }
        };
        let tokens = tokenizer.encode(&text);
        let id = format!("file:{}", path.display());
        eprintln!("[active] source {} → {} tokens", id, tokens.len());
        sources.push(Box::new(VecSource::new(id, tokens)));
    }
    if include_noise {
        eprintln!("[active] source noise:uniform (vocab={})", model.config.vocab_size);
        let noise_seed = seed_tree.child_seed("active_ingest_noise");
        sources.push(Box::new(NoiseSource::new_seeded(
            model.config.vocab_size as u32, noise_seed,
        )));
    }
    if sources.is_empty() {
        eprintln!("[active] no usable sources, aborting");
        std::process::exit(2);
    }

    // Metrics sink is shared across probe and commit phases; all tokens that
    // actually run through the model are billed to the window.
    let mut metrics = match metrics_out {
        Some(path) => match clob::metrics::JsonlWriter::create(path) {
            Ok(writer) => {
                eprintln!("[metrics] Writing windows of {} tokens to {:?}", metrics_window, path);
                Some(MetricsSink::new(writer, metrics_window))
            }
            Err(e) => {
                eprintln!("[metrics] failed to open {:?}: {} — continuing without metrics", path, e);
                None
            }
        },
        None => None,
    };
    let mut meter = clob::metrics::TdpProxyMeter::t490();

    // Per-source time accounting for the verification histogram.
    let mut commit_tokens_by_source: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    model.reset_state();
    let mut total_committed = 0usize;
    let mut round = 0u64;

    while total_committed < max_tokens && !sources.is_empty() {
        // ── Probe phase: run each source's next probe batch, score it. ──────
        let mut probe_results: Vec<(usize, clob::perceive::active::ProbeStats)> = Vec::new();
        let mut to_remove: Vec<usize> = Vec::new();
        for (i, src) in sources.iter_mut().enumerate() {
            let tokens = src.next_batch(probe_tokens);
            if tokens.is_empty() {
                to_remove.push(i);
                continue;
            }
            let probe_start = Instant::now();
            let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(tokens.len().saturating_sub(1));
            let mut actual_nlls: Vec<f32> = Vec::with_capacity(tokens.len().saturating_sub(1));
            for t in 0..tokens.len().saturating_sub(1) {
                let logits = model.decode_step(tokens[t]);
                let p = prob_of_token(logits.data(), tokens[t + 1]);
                let nll = -((p.max(1e-12)) as f64).ln() as f32;
                hidden_states.push(model.last_hidden().data().to_vec());
                actual_nlls.push(nll);

                // Probe tokens are billed to the metrics window — joules
                // spent here count against J/nat just like commit tokens.
                let extra = model.take_extra_steps();
                if let Some(sink) = metrics.as_mut() {
                    let (joules, _) = meter.joules_since(probe_start);
                    sink.record(nll as f64, joules / tokens.len() as f64, 0, false);
                    for _ in 0..extra { sink.record_extra_step(); }
                    sink.maybe_flush();
                }
            }
            let elapsed = probe_start.elapsed().as_secs_f64();
            let (joules, _) = meter.joules_since(probe_start);
            let stats = score_probe(
                &hidden_states, &actual_nlls,
                model.energy_critic(),
                model.confidence_head(),
                elapsed, joules,
            );
            probe_results.push((i, stats));
            total_committed += tokens.len();
        }
        // Remove exhausted sources in reverse order to keep indices stable.
        for &i in to_remove.iter().rev() {
            sources.remove(i);
        }
        if probe_results.is_empty() {
            break;
        }

        // ── Scoring + selection ─────────────────────────────────────────────
        // Primary: acquisition score (descending). Secondary tiebreak:
        // lower mean_true_nll (ascending) — when N and C are uninformative
        // (untrained critic/head), still prefer the source where the model
        // actually achieved lower cross-entropy. This protects Phase D's
        // verification from becoming a sort-stability test.
        probe_results.sort_by(|a, b| {
            b.1.acquisition_score()
                .partial_cmp(&a.1.acquisition_score())
                .unwrap()
                .then(
                    a.1.mean_true_nll
                        .partial_cmp(&b.1.mean_true_nll)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        eprintln!(
            "[active] round {} probe scores:{}",
            round,
            probe_results.iter().map(|(i, s)| {
                format!(" {}={:.3}", sources.get(*i).map(|s| s.id()).unwrap_or("?"), s.acquisition_score())
            }).collect::<String>()
        );
        let (winner_idx, _) = probe_results[0];

        // ── Commit phase: spend commit_tokens on the winner. ────────────────
        let winner_tokens = sources[winner_idx].next_batch(commit_tokens);
        if winner_tokens.is_empty() {
            // Exhausted mid-round; continue with remaining sources.
            sources.remove(winner_idx);
            round += 1;
            continue;
        }
        let winner_id = sources[winner_idx].id().to_string();
        *commit_tokens_by_source.entry(winner_id.clone()).or_insert(0) += winner_tokens.len();

        let commit_start = Instant::now();
        for t in 0..winner_tokens.len().saturating_sub(1) {
            let step_start = Instant::now();
            let logits = model.decode_step(winner_tokens[t]);
            let p = prob_of_token(logits.data(), winner_tokens[t + 1]);
            let nll = -((p.max(1e-12)) as f64).ln();

            // Commit phase stores episodes when novel (reuses ingest logic).
            let (energy, _) = model.energy_score_and_detect();
            // Use the critic's internal baseline rule for now; Phase B+C's
            // detector wrapping was on Head C only.
            let is_novel = energy > model.energy_baseline() * 1.5;
            if is_novel {
                let top_k: Vec<(u32, f32)> = {
                    let data = logits.data();
                    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = data.iter().map(|l| (l - max).exp()).sum();
                    let mut indexed: Vec<(u32, f32)> = data.iter().enumerate()
                        .map(|(i, &l)| (i as u32, (l - max).exp() / sum.max(1e-12)))
                        .collect();
                    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    indexed.truncate(10);
                    indexed
                };
                let ctx_start = t.saturating_sub(31);
                let context: Vec<u32> = winner_tokens[ctx_start..=t].to_vec();
                let episode = Episode::new(
                    now_nanos(), context,
                    model.last_hidden().data().to_vec(),
                    energy, top_k, winner_tokens[t + 1],
                );
                let _ = memory.store(&episode);
            }

            let extra = model.take_extra_steps();
            if let Some(sink) = metrics.as_mut() {
                let (joules, now) = meter.joules_since(step_start);
                let nanos = now.duration_since(step_start).as_nanos();
                sink.record(nll, joules, nanos, is_novel);
                for _ in 0..extra { sink.record_extra_step(); }
                sink.maybe_flush();
            }
        }
        total_committed += winner_tokens.len();
        let commit_elapsed = commit_start.elapsed().as_secs_f64();
        eprintln!(
            "[active] round {} commit: {} tokens from {} ({:.1} tok/s)",
            round, winner_tokens.len(), winner_id,
            winner_tokens.len() as f64 / commit_elapsed,
        );
        round += 1;
    }

    if let Some(sink) = metrics.as_mut() {
        sink.flush_final();
    }

    eprintln!("[active] Done: {} total tokens across {} rounds", total_committed, round);
    eprintln!("[active] commit histogram:");
    let mut by_source: Vec<(String, usize)> = commit_tokens_by_source.into_iter().collect();
    by_source.sort_by(|a, b| b.1.cmp(&a.1));
    let total_commit: usize = by_source.iter().map(|(_, n)| n).sum();
    for (id, n) in &by_source {
        let pct = if total_commit > 0 { *n as f64 / total_commit as f64 * 100.0 } else { 0.0 };
        eprintln!("  {}: {} tokens ({:.1}%)", id, n, pct);
    }
}

fn cmd_bench_suite(
    model_path: &PathBuf, eval_corpus: &PathBuf,
    tokenizer_path: &Option<PathBuf>, seeds: &[u64],
    critic_path: &Option<PathBuf>, confidence_head_path: &Option<PathBuf>,
    meta_critic_path: &Option<PathBuf>,
    routers_trained: &[PathBuf], modules_dirs: &[PathBuf],
    max_tokens: usize, adaptive_max_extra: u8, adaptive_z: f32,
    out: &PathBuf,
) {
    use clob::eval::bench_suite::{self, BenchInputs, BenchMatrix};

    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[bench-suite] Loading model from {:?}", model_path);
    let model_for_vocab = io::loader::load_model(model_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model_for_vocab.config);
    drop(model_for_vocab); // each cell loads fresh.

    let text = std::fs::read_to_string(eval_corpus).expect("failed to read eval corpus");
    let holdout_tokens = tokenizer.encode(&text);
    eprintln!("[bench-suite] Holdout tokens: {}", holdout_tokens.len());

    let critic = critic_path.as_ref().map(|p| {
        let b = std::fs::read(p).expect("read critic");
        EnergyCritic::from_bytes(&b).expect("parse critic")
    });
    let confidence_head = confidence_head_path.as_ref().map(|p| {
        let b = std::fs::read(p).expect("read confidence head");
        ConfidenceHead::from_bytes(&b).expect("parse confidence head")
    });
    let meta_critic = meta_critic_path.as_ref().map(|p| {
        let b = std::fs::read(p).expect("read meta critic");
        MetaCritic::from_bytes(&b).expect("parse meta critic")
    });

    let module_axis = bench_suite::cartesian_option_paths(modules_dirs);
    let routers_axis = bench_suite::cartesian_option_paths(routers_trained);

    let matrix = BenchMatrix {
        seeds: seeds.to_vec(),
        adaptive: vec![false, true],
        modules_dir: module_axis,
        routers_path: routers_axis,
        max_tokens,
        adaptive_max_extra,
        adaptive_z,
    };

    let inputs = BenchInputs {
        model_path,
        tokenizer: &tokenizer,
        holdout_tokens: &holdout_tokens,
        critic: critic.as_ref(),
        confidence_head: confidence_head.as_ref(),
        meta_critic: meta_critic.as_ref(),
    };

    eprintln!("[bench-suite] Running {} cells", matrix.seeds.len()
        * matrix.adaptive.len() * matrix.modules_dir.len() * matrix.routers_path.len());
    let cells = bench_suite::run_matrix(&matrix, &inputs).expect("bench matrix failed");
    bench_suite::emit_tsv(&cells, out).expect("emit tsv");
    eprintln!("[bench-suite] Wrote {} cells to {:?}", cells.len(), out);
    eprintln!("\n[bench-suite] Summary (axis-pivot):\n{}",
        bench_suite::summarize_by_axis(&cells));

    // Sidecar manifest for the TSV.
    let mut manifest = clob::config::manifest::RunManifest::new("bench-suite")
        .with_input("model", model_path)
        .with_input("eval_corpus", eval_corpus);
    if let Some(p) = confidence_head_path.as_ref() {
        manifest = manifest.with_input("confidence_head", p);
    }
    if let Some(p) = meta_critic_path.as_ref() {
        manifest = manifest.with_input("meta_critic", p);
    }
    if let Err(e) = manifest.save_sidecar(out) {
        eprintln!("[bench-suite] manifest write failed: {}", e);
    }
}

fn cmd_metrics(path: &PathBuf, tail: usize) {
    let n = if tail == 0 { usize::MAX } else { tail };
    let snaps = match clob::metrics::read_tail(path, n) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[metrics] cannot read {:?}: {}", path, e);
            std::process::exit(2);
        }
    };
    if snaps.is_empty() {
        eprintln!("[metrics] no windows in {:?}", path);
        return;
    }
    let baseline_jpn = snaps[0].j_per_nat;
    let baseline_ce = snaps[0].mean_nll;
    println!(
        "{:>4} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "idx", "tokens", "sec", "J", "J/nat", "ΔJ/nat%", "mean_nll", "Δmean_nll", "tok/s",
    );
    for s in &snaps {
        let d_jpn = if baseline_jpn > 0.0 {
            (s.j_per_nat - baseline_jpn) / baseline_jpn * 100.0
        } else {
            0.0
        };
        let d_ce = s.mean_nll - baseline_ce;
        println!(
            "{:>4} {:>8} {:>8.2} {:>10.3} {:>10.4} {:>+9.2}% {:>10.4} {:>+10.4} {:>8.1}",
            s.window_idx, s.tokens, s.seconds,
            s.joules, s.j_per_nat, d_jpn,
            s.mean_nll, d_ce,
            s.tokens_per_sec,
        );
    }
    // Summary line
    let last = snaps.last().unwrap();
    println!();
    println!(
        "baseline J/nat = {:.4} | latest J/nat = {:.4} | ΔCE vs baseline = {:+.4}",
        baseline_jpn, last.j_per_nat, last.mean_nll - baseline_ce,
    );
}

fn cmd_eval(
    model_path: &PathBuf, corpus_path: &PathBuf,
    tokenizer_path: &Option<PathBuf>, modules_dir: &Option<PathBuf>,
    critic_path: &Option<PathBuf>, readout_path: &Option<PathBuf>, max_tokens: usize,
) {
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[eval] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model.config);

    if let Some(p) = critic_path.as_ref() {
        apply_critic_file(&mut model, p);
    }
    if let Some(p) = readout_path.as_ref() {
        apply_readout_file(&mut model, p);
    }
    let n_modules = if let Some(dir) = modules_dir.as_ref() {
        load_modules_into(&mut model, dir)
    } else { 0 };

    eprintln!("[eval] Tokenizing {:?}", corpus_path);
    let text = std::fs::read_to_string(corpus_path).expect("failed to read corpus");
    let tokens = tokenizer.encode(&text);
    let limit = if max_tokens == 0 { tokens.len() } else { tokens.len().min(max_tokens) };
    eprintln!("[eval] Scoring {} tokens (of {} total) with {} modules",
        limit, tokens.len(), n_modules);

    let mut total_nll = 0.0f64;
    let mut n_predicted = 0usize;
    let start = Instant::now();

    model.reset_state();
    for i in 0..limit.saturating_sub(1) {
        let logits = model.decode_step(tokens[i]);
        let p = prob_of_token(logits.data(), tokens[i + 1]);
        total_nll += -((p.max(1e-12)) as f64).ln();
        n_predicted += 1;
    }

    let elapsed = start.elapsed();
    let mean_nll = if n_predicted > 0 { total_nll / n_predicted as f64 } else { 0.0 };
    let perplexity = mean_nll.exp();
    let uniform_nll = (model.config.vocab_size as f64).ln();
    eprintln!("[eval] n_tokens={}, mean_nll={:.6}, perplexity={:.4}", n_predicted, mean_nll, perplexity);
    eprintln!("[eval] (uniform baseline NLL over vocab={} is {:.4})", model.config.vocab_size, uniform_nll);
    eprintln!("[eval] speed: {:.1} tok/s ({:.2}s total)",
        n_predicted as f64 / elapsed.as_secs_f64(), elapsed.as_secs_f64());

    // Machine-readable last line for scripting.
    println!("{{\"n\":{},\"mean_nll\":{:.8},\"perplexity\":{:.6},\"n_modules\":{}}}",
        n_predicted, mean_nll, perplexity, n_modules);
}

fn cmd_calibrate(
    model_path: &PathBuf, corpus_path: &PathBuf,
    tokenizer_path: &Option<PathBuf>, output: &PathBuf,
    lr: f32, epochs: usize, max_tokens: usize,
) {
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[calibrate] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model.config);

    eprintln!("[calibrate] Tokenizing {:?}", corpus_path);
    let text = std::fs::read_to_string(corpus_path).expect("failed to read corpus");
    let tokens = tokenizer.encode(&text);
    let per_epoch = if max_tokens == 0 { tokens.len() } else { tokens.len().min(max_tokens) };
    eprintln!("[calibrate] {} tokens × {} epochs, lr={}", per_epoch, epochs, lr);

    let d_model = model.config.d_model;
    let mut hidden_scratch: Vec<f32> = vec![0.0; d_model];
    let start = Instant::now();

    for epoch in 0..epochs {
        model.reset_state();
        let mut total_nll = 0.0f64;
        let mut total_sse = 0.0f64;
        let mut n = 0usize;

        for i in 0..per_epoch.saturating_sub(1) {
            let logits = model.decode_step(tokens[i]);
            let p = prob_of_token(logits.data(), tokens[i + 1]);
            let nll = -(p.max(1e-12) as f64).ln();
            total_nll += nll;

            // Snapshot current hidden before training to avoid aliasing.
            hidden_scratch.copy_from_slice(model.last_hidden().data());
            let sse = model.energy_critic_mut()
                .train_step(&hidden_scratch, nll as f32, lr);
            total_sse += sse as f64;
            n += 1;
        }

        let mean_nll = if n > 0 { total_nll / n as f64 } else { 0.0 };
        let mean_sse = if n > 0 { total_sse / n as f64 } else { 0.0 };
        eprintln!("[calibrate] epoch {}: n={}, mean_nll={:.4}, mean_critic_se={:.4}, cumulative_mse={:.4}",
            epoch + 1, n, mean_nll, mean_sse, model.energy_critic().train_mse());
    }

    let bytes = model.energy_critic().to_bytes();
    std::fs::write(output, &bytes).expect("failed to write critic file");
    let elapsed = start.elapsed();
    eprintln!("[calibrate] wrote {:?} ({} bytes), {:.2}s total, n_trained={}",
        output, bytes.len(), elapsed.as_secs_f64(), model.energy_critic().n_trained());
    let manifest = clob::config::manifest::RunManifest::new("calibrate")
        .with_input("model", model_path)
        .with_input("corpus", corpus_path);
    if let Err(e) = manifest.save_sidecar(output) {
        eprintln!("[calibrate] manifest write failed: {}", e);
    }

    // Diagnostic: correlation between critic output and NLL on a single fresh pass.
    model.reset_state();
    let sample_n = per_epoch.min(2000).saturating_sub(1);
    let mut xs = Vec::with_capacity(sample_n);
    let mut ys = Vec::with_capacity(sample_n);
    for i in 0..sample_n {
        let logits = model.decode_step(tokens[i]);
        let p = prob_of_token(logits.data(), tokens[i + 1]);
        let nll = -(p.max(1e-12) as f64).ln();
        let pred = model.energy_critic().predict(model.last_hidden().data()) as f64;
        xs.push(pred); ys.push(nll);
    }
    if !xs.is_empty() {
        let n = xs.len() as f64;
        let mx: f64 = xs.iter().sum::<f64>() / n;
        let my: f64 = ys.iter().sum::<f64>() / n;
        let mut num = 0.0f64; let mut sx = 0.0f64; let mut sy = 0.0f64;
        for i in 0..xs.len() {
            let dx = xs[i] - mx; let dy = ys[i] - my;
            num += dx * dy; sx += dx * dx; sy += dy * dy;
        }
        let r = if sx > 0.0 && sy > 0.0 { num / (sx.sqrt() * sy.sqrt()) } else { 0.0 };
        eprintln!("[calibrate] diagnostic (first {} tokens): pearson(critic, nll) = {:.4}", xs.len(), r);
        println!("{{\"n_trained\":{},\"mse\":{:.6},\"pearson\":{:.6}}}",
            model.energy_critic().n_trained(), model.energy_critic().train_mse(), r);
    }
}

fn cmd_train_router(
    model_path: &PathBuf, corpus_path: &PathBuf,
    tokenizer_path: &Option<PathBuf>, seed: u64,
    cf_rate: f32, baseline_decay: f32, lr: f32, weight_decay: f32,
    clip_norm: f32, batch_size: usize, max_tokens: usize,
    output: &Option<PathBuf>,
) {
    let seed_tree = clob::util::seed::SeedTree::new(seed);
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[router] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model.config);

    let moe_layers = model.moe_layer_indices();
    if moe_layers.is_empty() {
        eprintln!("[router] model has no MoE layers — nothing to train");
        return;
    }
    eprintln!("[router] {} MoE layers: {:?}", moe_layers.len(), moe_layers);
    eprintln!("[router] cf_rate={}, lr={}, batch_size={}, baseline_decay={}",
        cf_rate, lr, batch_size, baseline_decay);

    // Snapshot original weights per layer for l2-drift reporting.
    let mut orig_weights: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
    for &li in &moe_layers {
        if let Some(r) = model.router_mut(li) {
            orig_weights.insert(li, r.weights().to_vec());
            r.reset_training_state();
        }
    }

    let text = std::fs::read_to_string(corpus_path).expect("failed to read corpus");
    let tokens = tokenizer.encode(&text);
    let limit = if max_tokens == 0 { tokens.len() } else { tokens.len().min(max_tokens) };
    eprintln!("[router] training on {} tokens", limit);

    let mut rng = seed_tree.child("train_router_cf");
    let mut baseline: f64 = 0.0;
    let mut baseline_initialized = false;
    let mut n_cf = 0u64;
    let mut total_steps = 0u64;

    model.reset_state();
    for i in 0..limit.saturating_sub(1) {
        let (logits, captures) = model.decode_step_training(tokens[i], &mut rng, cf_rate);
        let p = prob_of_token(logits.data(), tokens[i + 1]);
        let nll = -(p.max(1e-12) as f64).ln();

        // Initialize baseline to first observed NLL.
        if !baseline_initialized {
            baseline = nll;
            baseline_initialized = true;
        }
        let advantage = (baseline - nll) as f32;
        let bd = baseline_decay as f64;
        baseline = bd * baseline + (1.0 - bd) * nll;

        // Update each MoE layer's router with its own hidden/route result.
        for (layer_idx, capture) in moe_layers.iter().zip(captures.iter().filter_map(|c| c.as_ref())) {
            let (hidden_at_router, route) = capture;
            if route.counterfactual {
                n_cf += 1;
            }
            if let Some(router) = model.router_mut(*layer_idx) {
                router.accumulate_policy_gradient(
                    hidden_at_router,
                    &route.expert_indices,
                    &route.full_probs,
                    advantage,
                );
                if route.counterfactual {
                    router.record_counterfactual();
                }
            }
        }

        // AdamW step every batch_size tokens, for each layer independently.
        if (i + 1) % batch_size == 0 {
            let mut max_update = 0.0f32;
            for &li in &moe_layers {
                if let Some(router) = model.router_mut(li) {
                    let u = router.apply_adam(lr, weight_decay, 0.9, 0.999, 1e-8, clip_norm);
                    max_update = max_update.max(u);
                }
            }
            total_steps += 1;
            if total_steps % 10 == 0 {
                eprintln!("[router] step {} (tokens={}): baseline_nll={:.4}, last_nll={:.4}, advantage={:+.4}, cf={}, max_update_norm={:.6}",
                    total_steps, i + 1, baseline, nll, advantage, n_cf, max_update);
            }
        }
    }

    // Final AdamW step on partial batch.
    for &li in &moe_layers {
        if let Some(router) = model.router_mut(li) {
            if router.n_accumulated() > 0 {
                router.apply_adam(lr, weight_decay, 0.9, 0.999, 1e-8, clip_norm);
            }
        }
    }

    // Report per-layer L2 drift + save to side-file.
    let mut side_file: Vec<(u32, Vec<f32>)> = Vec::new();
    eprintln!("[router] final per-layer diagnostics:");
    for &li in &moe_layers {
        let orig = orig_weights.get(&li).unwrap();
        let router = model.router_mut(li).unwrap();
        let new_w = router.weights();
        let delta_sq: f32 = orig.iter().zip(new_w.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let drift = delta_sq.sqrt();
        let orig_norm: f32 = orig.iter().map(|w| w * w).sum::<f32>().sqrt();
        eprintln!("  layer {}: steps={}, counterfactuals={}, l2_drift={:.5} (original_norm={:.4}, relative={:.2}%)",
            li, router.step_count(), router.counterfactual_count(), drift, orig_norm,
            100.0 * drift / orig_norm.max(1e-8));
        side_file.push((li as u32, new_w.to_vec()));
    }

    let out_path = output.clone().unwrap_or_else(|| {
        let mut p = model_path.clone();
        let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
        p.set_file_name(format!("{}_routers.bin", stem));
        p
    });
    let bytes = bincode::serialize(&side_file).expect("serialize routers");
    std::fs::write(&out_path, &bytes).expect("write routers file");
    eprintln!("[router] wrote {} routers to {:?} ({} bytes, cf_count={}, total_steps={})",
        side_file.len(), out_path, bytes.len(), n_cf, total_steps);
    let manifest = clob::config::manifest::RunManifest::new("train-router")
        .with_seed(seed)
        .with_input("model", model_path)
        .with_input("corpus", corpus_path);
    if let Err(e) = manifest.save_sidecar(&out_path) {
        eprintln!("[router] manifest write failed: {}", e);
    }
}

fn apply_routers_file(model: &mut CoreModel, path: &Path) -> bool {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => { eprintln!("[routers] {:?}: {}", path, e); return false; }
    };
    let loaded: Vec<(u32, Vec<f32>)> = match bincode::deserialize(&bytes) {
        Ok(v) => v,
        Err(e) => { eprintln!("[routers] parse {:?}: {}", path, e); return false; }
    };
    let mut applied = 0;
    for (layer_idx, new_w) in loaded {
        if let Some(router) = model.router_mut(layer_idx as usize) {
            if router.weights().len() != new_w.len() {
                eprintln!("[routers] layer {} weight-size mismatch ({} vs {}); skipping",
                    layer_idx, router.weights().len(), new_w.len());
                continue;
            }
            router.weights_mut().copy_from_slice(&new_w);
            applied += 1;
        } else {
            eprintln!("[routers] layer {} is not MoE in the loaded model; skipping", layer_idx);
        }
    }
    eprintln!("[routers] applied {} routers from {:?}", applied, path);
    applied > 0
}

fn cmd_calibrate_confidence(
    model_path: &PathBuf, corpus_path: &PathBuf,
    tokenizer_path: &Option<PathBuf>, output: &PathBuf,
    lr: f32, epochs: usize, max_tokens: usize, seed: u64,
    meta_out: &Option<PathBuf>, meta_refresh: u64,
) {
    let seed_tree = clob::util::seed::SeedTree::new(seed);
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[conf-cal] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model.config);

    let d_model = model.config.d_model;
    let mut head_rng = seed_tree.child("confidence_head_init");
    let mut head = ConfidenceHead::random(d_model, &mut head_rng);
    let mut meta: Option<MetaCritic> = meta_out.as_ref().map(|_| {
        let mut meta_rng = seed_tree.child("meta_critic_init");
        MetaCritic::new(d_model, &mut meta_rng, head.clone(), meta_refresh)
    });

    eprintln!("[conf-cal] Tokenizing {:?}", corpus_path);
    let text = std::fs::read_to_string(corpus_path).expect("failed to read corpus");
    let tokens = tokenizer.encode(&text);
    let per_epoch = if max_tokens == 0 { tokens.len() } else { tokens.len().min(max_tokens) };
    eprintln!("[conf-cal] {} tokens × {} epochs, lr={}{}",
        per_epoch, epochs, lr,
        if meta.is_some() { format!(", meta-critic refresh every {} steps", meta_refresh) } else { "".into() });

    let mut hidden_scratch: Vec<f32> = vec![0.0; d_model];
    let start = Instant::now();

    for epoch in 0..epochs {
        model.reset_state();
        let mut total_nll = 0.0f64;
        let mut total_sse = 0.0f64;
        let mut total_meta_sse = 0.0f64;
        let mut n = 0usize;

        for i in 0..per_epoch.saturating_sub(1) {
            let logits = model.decode_step(tokens[i]);
            let p = prob_of_token(logits.data(), tokens[i + 1]);
            let nll = -(p.max(1e-12) as f64).ln();
            total_nll += nll;

            hidden_scratch.copy_from_slice(model.last_hidden().data());
            // Train the meta-critic FIRST with the head state as it was at
            // this step — before the head updates on this sample. This is
            // the clean ordering for target-network training: label is a
            // property of the current live head, not of its next update.
            if let Some(m) = meta.as_mut() {
                let mse = m.train_step(&hidden_scratch, nll as f32, &head, lr);
                total_meta_sse += mse as f64;
            }
            let sse = head.train_step(&hidden_scratch, nll as f32, lr);
            total_sse += sse as f64;
            n += 1;
        }

        let mean_nll = if n > 0 { total_nll / n as f64 } else { 0.0 };
        let mean_sse = if n > 0 { total_sse / n as f64 } else { 0.0 };
        let meta_info = match meta.as_ref() {
            Some(m) => format!(", meta_se={:.4}, meta_mse_cum={:.4}, steps_since_refresh={}",
                if n > 0 { total_meta_sse / n as f64 } else { 0.0 },
                m.train_mse(), m.steps_since_refresh()),
            None => String::new(),
        };
        eprintln!("[conf-cal] epoch {}: n={}, mean_nll={:.4}, mean_head_se={:.4}, cumulative_mse={:.4}{}",
            epoch + 1, n, mean_nll, mean_sse, head.train_mse(), meta_info);
    }

    let bytes = head.to_bytes();
    std::fs::write(output, &bytes).expect("failed to write confidence head file");
    let elapsed = start.elapsed();
    eprintln!("[conf-cal] wrote {:?} ({} bytes), {:.2}s total, n_trained={}",
        output, bytes.len(), elapsed.as_secs_f64(), head.n_trained());
    let head_manifest = clob::config::manifest::RunManifest::new("calibrate-confidence")
        .with_seed(seed)
        .with_input("model", model_path)
        .with_input("corpus", corpus_path);
    if let Err(e) = head_manifest.save_sidecar(output) {
        eprintln!("[conf-cal] head manifest write failed: {}", e);
    }

    if let (Some(path), Some(m)) = (meta_out.as_ref(), meta.as_ref()) {
        let meta_bytes = m.to_bytes();
        std::fs::write(path, &meta_bytes).expect("failed to write meta critic file");
        eprintln!("[conf-cal] wrote meta {:?} ({} bytes), n_trained={}, mse={:.4}",
            path, meta_bytes.len(), m.n_trained(), m.train_mse());
        let meta_manifest = clob::config::manifest::RunManifest::new("calibrate-confidence:meta")
            .with_seed(seed)
            .with_input("model", model_path)
            .with_input("corpus", corpus_path)
            .with_input("confidence_head", output);
        if let Err(e) = meta_manifest.save_sidecar(path) {
            eprintln!("[conf-cal] meta manifest write failed: {}", e);
        }
    }

    // Diagnostic: correlation between head output and NLL on a fresh pass.
    model.reset_state();
    let sample_n = per_epoch.min(2000).saturating_sub(1);
    let mut xs = Vec::with_capacity(sample_n);
    let mut ys = Vec::with_capacity(sample_n);
    for i in 0..sample_n {
        let logits = model.decode_step(tokens[i]);
        let p = prob_of_token(logits.data(), tokens[i + 1]);
        let nll = -(p.max(1e-12) as f64).ln();
        let pred = head.predict_raw(model.last_hidden().data()) as f64;
        xs.push(pred); ys.push(nll);
    }
    if !xs.is_empty() {
        let n = xs.len() as f64;
        let mx: f64 = xs.iter().sum::<f64>() / n;
        let my: f64 = ys.iter().sum::<f64>() / n;
        let mut num = 0.0f64; let mut sx = 0.0f64; let mut sy = 0.0f64;
        for i in 0..xs.len() {
            let dx = xs[i] - mx; let dy = ys[i] - my;
            num += dx * dy; sx += dx * dx; sy += dy * dy;
        }
        let r = if sx > 0.0 && sy > 0.0 { num / (sx.sqrt() * sy.sqrt()) } else { 0.0 };
        eprintln!("[conf-cal] diagnostic (first {} tokens): pearson(head, nll) = {:.4}", xs.len(), r);
        println!("{{\"n_trained\":{},\"mse\":{:.6},\"pearson\":{:.6}}}",
            head.n_trained(), head.train_mse(), r);
    }
}

/// Reservoir probe — fit a trainable f32 readout over the FROZEN core's
/// hidden states and measure how much held-out NLL it recovers beyond the
/// unigram marginal. See `clob::learn::readout` and `IF_FOUND.md`'s fork.
#[allow(clippy::too_many_arguments)]
fn cmd_probe_readout(
    model_path: &PathBuf, train_path: &PathBuf, holdout_path: &PathBuf,
    tokenizer_path: &Option<PathBuf>, max_tokens: usize, epochs: usize,
    batch_size: usize, lr: f32, weight_decays: &str, seed: u64,
    out: &Option<PathBuf>, save_readout: &Option<PathBuf>,
) {
    use clob::learn::readout::{fixed_logprob_nll, unigram_log_bias, FitConfig, LinearReadout};
    use clob::nn::readout::TrainedReadout;

    let seed_tree = clob::util::seed::SeedTree::new(seed);
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[probe] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    assert_vocab_compatible(&tokenizer, &model.config);

    let d = model.config.d_model;
    let vocab = model.config.vocab_size;
    eprintln!("[probe] frozen core: d_model={}, vocab={} — no head, no modules", d, vocab);

    // ── Extract frozen-core features. The hidden state AFTER consuming
    // token[i] is the predictor of token[i+1]; it is exactly what the tied
    // unembedding reads, so the trained readout sees the same features the
    // model's own (random) readout does. The SSM is recurrent — stream in
    // order, resetting state at the start of each corpus. ──────────────────
    let extract = |model: &mut CoreModel, tokens: &[u32], cap: usize, tag: &str|
        -> (Vec<f32>, Vec<u32>, f64) {
        let limit = if cap == 0 { tokens.len() } else { tokens.len().min(cap) };
        let n = limit.saturating_sub(1);
        let mut feats = Vec::with_capacity(n * d);
        let mut targets = Vec::with_capacity(n);
        let mut tied_nll = 0.0f64; // the model's own tied-random readout, for reference
        model.reset_state();
        for i in 0..n {
            let logits = model.decode_step(tokens[i]);
            let actual = tokens[i + 1];
            tied_nll += -(prob_of_token(logits.data(), actual).max(1e-12) as f64).ln();
            feats.extend_from_slice(model.last_hidden().data());
            targets.push(actual);
            if (i + 1) % 50_000 == 0 {
                eprintln!("[probe] {}: extracted {}/{} features", tag, i + 1, n);
            }
        }
        let tied_mean = if n > 0 { tied_nll / n as f64 } else { 0.0 };
        (feats, targets, tied_mean)
    };

    let train_tokens = load_or_encode_tokens(train_path, &tokenizer, &model.config);
    eprintln!("[probe] extracting TRAIN features from {:?}", train_path);
    let (mut tf, tt, _tied_train) = extract(&mut model, &train_tokens, max_tokens, "train");

    let holdout_tokens = load_or_encode_tokens(holdout_path, &tokenizer, &model.config);
    eprintln!("[probe] extracting HOLDOUT features from {:?}", holdout_path);
    let (mut hf, ht, tied_holdout) = extract(&mut model, &holdout_tokens, 0, "holdout");

    eprintln!("[probe] train positions={}, holdout positions={}", tt.len(), ht.len());

    // ── Standardize features: per-dim zero-mean / unit-variance from TRAIN
    // statistics, applied to both splits. This is an information-preserving
    // linear conditioning (a linear readout over standardized features is
    // equivalent to one over raw features); it only makes the momentum-SGD
    // well-scaled so the lr is not at the mercy of the random core's per-dim
    // output magnitudes. The (mean, inv_std) are retained so they can be
    // folded into a saved readout (which then applies to raw hidden states).
    let std_stats: (Vec<f32>, Vec<f32>) = {
        let n = tt.len();
        let mut mean = vec![0.0f64; d];
        for e in 0..n {
            let h = &tf[e * d..e * d + d];
            for k in 0..d {
                mean[k] += h[k] as f64;
            }
        }
        for m in mean.iter_mut() {
            *m /= n.max(1) as f64;
        }
        let mut var = vec![0.0f64; d];
        for e in 0..n {
            let h = &tf[e * d..e * d + d];
            for k in 0..d {
                let dv = h[k] as f64 - mean[k];
                var[k] += dv * dv;
            }
        }
        let meanf: Vec<f32> = mean.iter().map(|&m| m as f32).collect();
        let inv_std: Vec<f32> = var
            .iter()
            .map(|&s| (1.0 / (s / n.max(1) as f64).sqrt().max(1e-6)) as f32)
            .collect();
        for feats in [&mut tf, &mut hf] {
            for h in feats.chunks_mut(d) {
                for k in 0..d {
                    h[k] = (h[k] - meanf[k]) * inv_std[k];
                }
            }
        }
        (meanf, inv_std)
    };
    let (std_mean, std_inv) = std_stats;

    // ── Baselines ──────────────────────────────────────────────────────────
    let floor = (vocab as f64).ln(); // uniform over the model vocab
    let bias = unigram_log_bias(&tt, vocab);
    let marginal_holdout = fixed_logprob_nll(&bias, &ht); // Laplace unigram (fixed)

    // ── Trained-bias-only null (the proper "no context" baseline) ───────────
    // W frozen at 0, only the bias trained under the identical objective. This
    // re-fits the token marginal as well as a linear readout possibly can
    // WITHOUT using the core's hidden state, so the gap between a full fit and
    // this null is purely the contextual (W·h) contribution. Comparing instead
    // against the hand-set Laplace marginal would miscredit bias-fitting as if
    // it were context.
    let bias_only_holdout = {
        let cfg = FitConfig {
            epochs, batch_size, lr, weight_decay: 0.0, freeze_w: true,
            ..FitConfig::default()
        };
        let mut null = LinearReadout::with_bias(vocab, d, bias.clone());
        let mut nrng = seed_tree.child("readout_bias_only");
        let ntrace = null.fit(&tf, &tt, &hf, &ht, &cfg, &mut nrng);
        let bo = ntrace.iter().map(|s| s.holdout_nll).fold(marginal_holdout, f64::min);
        eprintln!("[probe] bias-only null: holdout NLL = {:.4} (Laplace marginal = {:.4})",
            bo, marginal_holdout);
        bo
    };

    // ── Fit the trainable readout across a weight-decay sweep ───────────────
    // Features are extracted once; each wd gets a fresh readout warm-started
    // at the marginal. The verdict uses the best held-out NLL across the whole
    // sweep, so a weak-but-real signal cannot be hidden behind a single
    // under- or over-regularized fit.
    let wds: Vec<f32> = weight_decays
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    let wds = if wds.is_empty() { vec![1e-3] } else { wds };
    eprintln!("[probe] fitting readout: epochs={}, batch_size={}, lr={}, wd sweep={:?}",
        epochs, batch_size, lr, wds);

    let start = Instant::now();
    // Per-wd: (wd, best_holdout_fitted, final_holdout, train_first, train_last, trace).
    let mut sweep: Vec<(f32, f64, f64, f64, f64, Vec<clob::learn::readout::EpochStat>)> = Vec::new();
    // Fitted readouts retained in parallel so the best can be persisted.
    let mut readouts: Vec<LinearReadout> = Vec::new();
    for &wd in &wds {
        let cfg = FitConfig { epochs, batch_size, lr, weight_decay: wd, ..FitConfig::default() };
        let mut readout = LinearReadout::with_bias(vocab, d, bias.clone());
        // Independent, deterministic shuffle stream per wd.
        let mut fit_rng = seed_tree.child(&format!("readout_shuffle_wd{}", wd));
        let trace = readout.fit(&tf, &tt, &hf, &ht, &cfg, &mut fit_rng);
        let bf = trace.iter().map(|s| s.holdout_nll).fold(f64::INFINITY, f64::min);
        let fin = trace.last().map(|s| s.holdout_nll).unwrap_or(f64::NAN);
        let tfirst = trace.first().map(|s| s.train_nll).unwrap_or(f64::NAN);
        let tlast = trace.last().map(|s| s.train_nll).unwrap_or(f64::NAN);
        eprintln!("[probe] wd={:<8}: best_holdout={:.4} final={:.4} (train {:.4}→{:.4})",
            wd, bf, fin, tfirst, tlast);
        sweep.push((wd, bf, fin, tfirst, tlast, trace));
        readouts.push(readout);
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Best wd = lowest held-out NLL across the sweep.
    let best_idx = sweep
        .iter()
        .enumerate()
        .min_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let (best_wd, best_fitted, final_holdout, train_first, train_last, trace) = {
        let s = &sweep[best_idx];
        (s.0, s.1, s.2, s.3, s.4, s.5.clone())
    };
    // The readout can always fall back to the bias-only null (W=0), so the
    // best *achievable* holdout NLL is floored there.
    let best_holdout = best_fitted.min(bias_only_holdout);

    // Convergence guard: if the best-performing fit still ran its training loss
    // away from the near-optimal warm start AND never beat the null, the
    // optimization failed — not evidence about the reservoir. Refuse to call
    // it dead in that case.
    let diverged = train_last > train_first + 0.1 && best_fitted > bias_only_holdout;

    // ── Verdict. The deciding quantity is how far the full readout falls below
    // the trained-bias-only null: that gap is purely the frozen core's
    // contextual (W·h) contribution, with the token-marginal advantage divided
    // out. (Heuristic bands for orientation — NOT the pre-registered Part V bet
    // thresholds.) ─────────────────────────────────────────────────────────
    let gain_abs = bias_only_holdout - best_holdout;
    let gain_rel = if bias_only_holdout > 0.0 { gain_abs / bias_only_holdout } else { 0.0 };
    let verdict = if diverged {
        "FIT DID NOT CONVERGE — training loss rose from the warm start and no epoch beat the bias-only null; lower --lr and re-run. Result INVALID, not a reservoir verdict."
    } else if gain_rel >= 0.02 {
        "RESERVOIR ALIVE — full readout beats the trained-bias-only null; the frozen core exposes linearly-readable context. Path A (reservoir) viable, crystallization has signal to compress."
    } else if gain_rel >= 0.005 {
        "WEAK SIGNAL — full readout barely beats the bias-only null; inconclusive, lean Path B."
    } else {
        "RESERVOIR DEAD — full readout matches the trained-bias-only null; the core's hidden state adds no usable context beyond the token marginal. Path B (train the core) is mandatory."
    };

    eprintln!();
    eprintln!("[probe] ===== reservoir probe summary =====");
    eprintln!("[probe]  uniform floor      ln(vocab) = {:.4}", floor);
    eprintln!("[probe]  tied-random readout (model's own) holdout NLL = {:.4}", tied_holdout);
    eprintln!("[probe]  Laplace unigram marginal     holdout NLL = {:.4}", marginal_holdout);
    eprintln!("[probe]  trained-bias-only NULL       holdout NLL = {:.4}", bias_only_holdout);
    eprintln!("[probe]  --- weight-decay sweep (best holdout per wd) ---");
    for (wd, bf, _fin, _tf, _tl, _tr) in &sweep {
        let mark = if (*wd - best_wd).abs() < f32::EPSILON { " <- best" } else { "" };
        eprintln!("[probe]    wd={:<8} best_holdout={:.4}{}", wd, bf, mark);
    }
    eprintln!("[probe]  TRAINED readout (best wd={})  holdout NLL = {:.4}", best_wd, best_fitted);
    eprintln!("[probe]  TRAINED readout (floored)    holdout NLL = {:.4}", best_holdout);
    eprintln!("[probe]  TRAINED readout (final)      holdout NLL = {:.4}", final_holdout);
    eprintln!("[probe]  contextual gain over bias-only null = {:.4} nats ({:+.2}%)",
        gain_abs, gain_rel * 100.0);
    eprintln!("[probe]  VERDICT: {}", verdict);
    eprintln!("[probe]  ({:.1}s, {} wd × {} epochs)", elapsed, wds.len(), epochs);

    // Machine-readable line on stdout (mirrors other subcommands).
    println!(
        "{{\"floor\":{:.6},\"tied_holdout_nll\":{:.6},\"marginal_holdout_nll\":{:.6},\"bias_only_null_nll\":{:.6},\"best_wd\":{},\"trained_holdout_nll_best\":{:.6},\"trained_holdout_nll_final\":{:.6},\"contextual_gain_nats\":{:.6},\"contextual_gain_rel\":{:.6},\"train_positions\":{},\"holdout_positions\":{}}}",
        floor, tied_holdout, marginal_holdout, bias_only_holdout, best_wd, best_holdout, final_holdout,
        gain_abs, gain_rel, tt.len(), ht.len()
    );

    if let Some(out_path) = out {
        let sweep_json: Vec<String> = sweep.iter().map(|(wd, bf, fin, tfi, tla, _)|
            format!("{{\"wd\":{},\"best_holdout_nll\":{:.6},\"final_holdout_nll\":{:.6},\"train_first\":{:.6},\"train_last\":{:.6}}}",
                wd, bf, fin, tfi, tla)).collect();
        let epochs_json: Vec<String> = trace.iter().map(|s|
            format!("{{\"epoch\":{},\"train_nll\":{:.6},\"holdout_nll\":{:.6}}}",
                s.epoch, s.train_nll, s.holdout_nll)).collect();
        let report = format!(
            "{{\n  \"floor\": {:.6},\n  \"tied_holdout_nll\": {:.6},\n  \"marginal_holdout_nll\": {:.6},\n  \"bias_only_null_nll\": {:.6},\n  \"best_wd\": {},\n  \"trained_holdout_nll_best\": {:.6},\n  \"trained_holdout_nll_final\": {:.6},\n  \"contextual_gain_nats\": {:.6},\n  \"contextual_gain_rel\": {:.6},\n  \"verdict\": \"{}\",\n  \"train_positions\": {},\n  \"holdout_positions\": {},\n  \"wd_sweep\": [{}],\n  \"best_wd_epochs\": [{}]\n}}\n",
            floor, tied_holdout, marginal_holdout, bias_only_holdout, best_wd, best_holdout, final_holdout,
            gain_abs, gain_rel,
            verdict.split(" —").next().unwrap_or(verdict),
            tt.len(), ht.len(), sweep_json.join(", "), epochs_json.join(", ")
        );
        if let Err(e) = std::fs::write(out_path, report) {
            eprintln!("[probe] report write failed: {}", e);
        } else {
            eprintln!("[probe] wrote report {:?}", out_path);
            let manifest = clob::config::manifest::RunManifest::new("probe-readout")
                .with_seed(seed)
                .with_input("model", model_path)
                .with_input("train", train_path)
                .with_input("holdout", holdout_path);
            if let Err(e) = manifest.save_sidecar(out_path) {
                eprintln!("[probe] manifest write failed: {}", e);
            }
        }
    }

    // ── Persist the best-fitted readout as an installable sidecar ───────────
    // Fold standardization (x' = (x - mean)·inv_std) into (W, b) so the saved
    // readout applies directly to raw hidden states:
    //   logits = W·x' + b = (W·diag(inv_std))·x + (b - W·diag(inv_std)·mean).
    if let Some(save_path) = save_readout {
        let src = &readouts[best_idx]; // final-epoch weights of the best wd
        let mut w_folded = vec![0.0f32; vocab * d];
        let mut b_folded = src.b.clone();
        for v in 0..vocab {
            let mut shift = 0.0f32;
            for k in 0..d {
                let wp = src.w[v * d + k] * std_inv[k];
                w_folded[v * d + k] = wp;
                shift += wp * std_mean[k];
            }
            b_folded[v] -= shift;
        }
        let readout = TrainedReadout::new(vocab, d, w_folded, b_folded);
        match std::fs::write(save_path, readout.to_bytes()) {
            Ok(()) => {
                eprintln!("[probe] saved readout sidecar (wd={}, final-epoch) → {:?}",
                    best_wd, save_path);
                let manifest = clob::config::manifest::RunManifest::new("probe-readout:readout")
                    .with_seed(seed)
                    .with_input("model", model_path)
                    .with_input("train", train_path);
                if let Err(e) = manifest.save_sidecar(save_path) {
                    eprintln!("[probe] readout manifest write failed: {}", e);
                }
            }
            Err(e) => eprintln!("[probe] readout save failed: {}", e),
        }
    }
}

fn cmd_crystal(memory_dir: &PathBuf, modules_dir: &PathBuf, d_model: usize, n_clusters: usize) {
    eprintln!("[crystal] Opening memory {:?}", memory_dir);
    let memory = EpisodicMemory::open(memory_dir, usize::MAX).expect("failed to open memory");
    let mut engine = CrystallizationEngine::new(d_model, CrystalConfig {
        n_clusters, ..CrystalConfig::default()
    });
    engine.set_next_id(next_module_id(modules_dir));
    let before = engine.n_modules();
    // No embedding supplied here; distillation will use the weaker hidden-centroid fallback.
    // Prefer `crystal` with --model when tied-unembed directions are wanted.
    let _n = engine.cycle(&memory, None, 0);
    let after = engine.n_modules();

    let mut saved = 0usize;
    for module in &engine.modules()[before..after] {
        match store::save_module(module, modules_dir) {
            Ok(path) => { eprintln!("[crystal] saved {:?}", path); saved += 1; }
            Err(e) => eprintln!("[crystal] save error for id={}: {}", module.id, e),
        }
    }
    eprintln!("[crystal] {} new modules saved to {:?}", saved, modules_dir);
}

fn cmd_crystal_with_model(memory_dir: &PathBuf, modules_dir: &PathBuf, model_path: &PathBuf, n_clusters: usize) {
    eprintln!("[crystal] Loading model {:?}", model_path);
    let model = io::loader::load_model(model_path).expect("failed to load model");
    eprintln!("[crystal] Opening memory {:?}", memory_dir);
    let memory = EpisodicMemory::open(memory_dir, usize::MAX).expect("failed to open memory");

    let d_model = model.config.d_model;
    let vocab = model.config.vocab_size;
    let embed = model.embed_table();

    let mut engine = CrystallizationEngine::new(d_model, CrystalConfig {
        n_clusters, ..CrystalConfig::default()
    });
    // Continue numbering from the highest existing module id on disk, so cycles
    // accumulate rather than overwrite.
    let next_id = next_module_id(modules_dir);
    engine.set_next_id(next_id);

    let before = engine.n_modules();
    let _n = engine.cycle(&memory, Some(embed), vocab);
    let after = engine.n_modules();

    let mut saved = 0usize;
    for module in &engine.modules()[before..after] {
        match store::save_module(module, modules_dir) {
            Ok(path) => { eprintln!("[crystal] saved {:?}", path); saved += 1; }
            Err(e) => eprintln!("[crystal] save error for id={}: {}", module.id, e),
        }
    }
    eprintln!("[crystal] {} new modules saved to {:?} (next_id was {})",
        saved, modules_dir, next_id);
}

/// Return the next crystal module id that won't collide with any existing
/// `module_<id>.mod` file in `dir`. Returns 0 if the directory is empty/missing.
fn next_module_id(dir: &Path) -> u64 {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut max_id: Option<u64> = None;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(stem) = name.strip_prefix("module_").and_then(|s| s.strip_suffix(".mod")) {
            if let Ok(id) = stem.parse::<u64>() {
                max_id = Some(max_id.map_or(id, |m| m.max(id)));
            }
        }
    }
    max_id.map(|m| m + 1).unwrap_or(0)
}

fn cmd_compile(modules_dir: &PathBuf) {
    eprintln!("[compile] Compilation of crystallized modules to native x86-64.");
    eprintln!("[compile] Modules dir: {:?}", modules_dir);
    // This would scan for crystal modules and compile them.
    // For now, demonstrate the pipeline with a synthetic module.
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mat = clob::tensor::ternary::TernaryMatrix::random(64, 64, &mut rng);

    let ir = clob::compile::lower::lower_ternary(&mat);
    eprintln!("[compile] {}", ir.stats());

    let code = clob::compile::x86::emit_x86(&ir);
    eprintln!("[compile] Emitted {} bytes of x86-64 machine code", code.text_size);

    std::fs::create_dir_all(modules_dir).ok();
    let so_path = modules_dir.join("test_module.so");
    clob::compile::elf::write_elf(&code, &so_path).expect("failed to write ELF");
    let file_size = std::fs::metadata(&so_path).map(|m| m.len()).unwrap_or(0);
    eprintln!("[compile] Wrote {:?} ({} bytes)", so_path, file_size);
}

fn cmd_spore(binary: &PathBuf, model: &PathBuf, output: &PathBuf) {
    eprintln!("[spore] Packaging kernel spore...");
    eprintln!("[spore] Binary: {:?}", binary);
    eprintln!("[spore] Model: {:?}", model);

    match clob::net::spore::package_spore(binary, model, output) {
        Ok(info) => {
            eprintln!("[spore] Done: {:?} ({})", output, info.summary());
        }
        Err(e) => {
            eprintln!("[spore] Failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_net_discover(timeout_ms: u64) {
    eprintln!("[net] Discovering peers via mDNS (timeout={}ms)...", timeout_ms);
    let peers = clob::net::discovery::discover_mdns("_clob._tcp.local", timeout_ms);
    if peers.is_empty() {
        eprintln!("[net] No peers found.");
    } else {
        for p in &peers {
            eprintln!("[net] Found: {} ({:?})", p.addr, p.source);
        }
    }
}

fn cmd_net_connect(addr: &str, model: &Option<PathBuf>) {
    eprintln!("[net] Connecting to {}...", addr);
    // Identify this kernel by its config fingerprint when a model is given, so
    // the peer can detect shape-incompatible module exchange. Anonymous (0/0)
    // otherwise — this is a connectivity check, not a module-exchange session.
    let (kernel_id, config_hash) = match model {
        Some(p) => match io::loader::load_model(p) {
            Ok(m) => {
                let h = m.config.config_hash();
                (h, h)
            }
            Err(e) => {
                eprintln!("[net] failed to load --model {:?}: {} — connecting anonymously", p, e);
                (0, 0)
            }
        },
        None => (0, 0),
    };
    match addr.parse::<std::net::SocketAddr>() {
        Ok(socket_addr) => {
            match clob::net::peer::Peer::connect(socket_addr, kernel_id, config_hash, 0) {
                Ok(peer) => {
                    eprintln!("[net] Connected to kernel #{}", peer.kernel_id);
                }
                Err(e) => {
                    eprintln!("[net] Failed: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("[net] Invalid address '{}': {}", addr, e);
        }
    }
}
