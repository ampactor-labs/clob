//! The Kernel — CLI entry point and main loop.

use clap::{Parser, Subcommand};
use clob::crystal::detector::NoveltyDetector;
use clob::crystal::engine::{CrystalConfig, CrystallizationEngine};
use clob::io;
use clob::memory::episode::Episode;
use clob::memory::ring::EpisodicMemory;
use clob::model::config::KernelConfig;
use clob::model::generate::{self, SamplingConfig};
use clob::model::stack::CoreModel;
use clob::token::bpe::BpeTokenizer;
use rand::SeedableRng;
use rand::Rng;
use std::path::PathBuf;
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
    /// Process a file through the kernel (non-interactive).
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
    },
    /// Compile crystallized modules to native code.
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
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Boot { memory_dir, modules_dir, model, config, seed, memory_capacity, tokenizer, listen, peer } => {
            cmd_boot(&memory_dir, &modules_dir, &model, &config, seed, memory_capacity, &tokenizer, &listen, &peer);
        }
        Commands::Think { model, max_tokens, temperature, tokenizer } => {
            cmd_think(&model, max_tokens, temperature, &tokenizer);
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
        Commands::Ingest { model, input, memory_dir, tokenizer } => {
            cmd_ingest(&model, &input, &memory_dir, &tokenizer);
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
                NetAction::Connect { addr } => cmd_net_connect(&addr),
            }
        }
    }
}

fn parse_config(name: &str) -> KernelConfig {
    match name {
        "tiny" => KernelConfig::tiny(),
        "seed" => KernelConfig::seed(),
        other => {
            eprintln!("[kernel] Unknown config '{}'. Use 'tiny' or 'seed'.", other);
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

// ============================================================================
// Commands
// ============================================================================

fn cmd_boot(
    memory_dir: &PathBuf, _modules_dir: &PathBuf, model_path: &PathBuf,
    config_str: &str, seed: u64, memory_capacity: usize,
    tokenizer_path: &Option<PathBuf>, _listen: &Option<String>, _peers: &[String],
) {
    let config = parse_config(config_str);
    let tokenizer = load_tokenizer(tokenizer_path);

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

    let memory = EpisodicMemory::open(memory_dir, memory_capacity)
        .expect("failed to open episodic memory");
    let mut crystal_engine = CrystallizationEngine::new(config.d_model, CrystalConfig::default());
    let mut novelty_detector = NoveltyDetector::new();

    eprintln!("\n=== THE KERNEL IS ALIVE ===\n");
    eprintln!("Commands: :status, :crystal, :reset, :quit\n");

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
                eprintln!("[status] Crystal: {}", crystal_engine.stats());
                eprintln!("[status] Novelty baseline: {:.4}", novelty_detector.baseline());
                continue;
            }
            ":crystal" => {
                eprintln!("[crystal] Running crystallization cycle...");
                let n = crystal_engine.cycle(&memory);
                eprintln!("[crystal] Crystallized {} new modules. {}", n, crystal_engine.stats());
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

        // Tokenize
        let input_tokens: Vec<u32> = tokenizer.encode(input).iter()
            .map(|&t| t.min(config.vocab_size as u32 - 1))
            .collect();

        let start = Instant::now();
        let logits = model.prefill(&input_tokens);
        context_tokens.extend_from_slice(&input_tokens);

        let (energy, is_novel) = model.energy_score_and_detect();
        let novelty = novelty_detector.evaluate(energy);

        if novelty.is_novel {
            let mut probs = logits.clone();
            probs.softmax_();
            let top_k: Vec<(u32, f32)> = {
                let mut indexed: Vec<(u32, f32)> = probs.data().iter().enumerate()
                    .map(|(i, &p)| (i as u32, p)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.truncate(10);
                indexed
            };
            let episode = Episode::new(
                now_nanos(), context_tokens.clone(),
                model.last_hidden().data().to_vec(),
                energy, top_k, 0,
            );
            if let Err(e) = memory.store(&episode) {
                eprintln!("[memory] store error: {}", e);
            }
        }

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
            crystal_engine.n_modules());
    }
}

fn cmd_think(model_path: &PathBuf, max_tokens: usize, temperature: f32, tokenizer_path: &Option<PathBuf>) {
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[kernel] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    let config = model.config.clone();
    eprintln!("[kernel] d={}, L={}, V={}", config.d_model, config.n_layers, config.vocab_size);

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let sampling = SamplingConfig { temperature, top_k: 50, top_p: 0.9 };

    loop {
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() { break; }
        let input = input.trim();
        if input == ":quit" || input == ":q" { break; }
        if input.is_empty() { continue; }

        let tokens: Vec<u32> = tokenizer.encode(input).iter()
            .map(|&t| t.min(config.vocab_size as u32 - 1)).collect();
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

fn cmd_ingest(model_path: &PathBuf, input_path: &PathBuf, memory_dir: &PathBuf, tokenizer_path: &Option<PathBuf>) {
    let tokenizer = load_tokenizer(tokenizer_path);
    eprintln!("[ingest] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    let config = model.config.clone();

    let memory = EpisodicMemory::open(memory_dir, 100_000).expect("failed to open memory");
    let mut novelty_detector = NoveltyDetector::new();

    eprintln!("[ingest] Processing {:?}", input_path);
    let mut stream = clob::perceive::FileStream::open(input_path).expect("failed to open");
    let mut total_tokens = 0usize;
    let mut novel_count = 0usize;
    let start = Instant::now();

    while let Some(percept) = stream.next_percept(&tokenizer) {
        if percept.tokens.is_empty() { continue; }
        let clamped: Vec<u32> = percept.tokens.iter()
            .map(|&t| t.min(config.vocab_size as u32 - 1)).collect();
        let _logits = model.prefill(&clamped);
        total_tokens += clamped.len();

        let (energy, _) = model.energy_score_and_detect();
        let novelty = novelty_detector.evaluate(energy);

        if novelty.is_novel {
            let episode = Episode::new(
                now_nanos(), clamped,
                model.last_hidden().data().to_vec(),
                energy, vec![], 0,
            );
            let _ = memory.store(&episode);
            novel_count += 1;
        }
        model.reset_state();
    }

    let elapsed = start.elapsed();
    let mem_stats = memory.stats();
    eprintln!("[ingest] Done: {} tokens, {} novel episodes, {:.1} tok/s",
        total_tokens, novel_count, total_tokens as f64 / elapsed.as_secs_f64());
    eprintln!("[ingest] Memory: {} episodes ({:.2} MB)",
        mem_stats.total_episodes, mem_stats.total_bytes as f64 / (1024.0 * 1024.0));
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

fn cmd_net_connect(addr: &str) {
    eprintln!("[net] Connecting to {}...", addr);
    match addr.parse::<std::net::SocketAddr>() {
        Ok(socket_addr) => {
            match clob::net::peer::Peer::connect(socket_addr, 0, 0) {
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
