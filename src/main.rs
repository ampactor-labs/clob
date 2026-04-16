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
    /// Bootstrap: create a seed model, initialize memory, enter the main loop.
    Boot {
        /// Directory for episodic memory.
        #[arg(long, default_value = "episodes")]
        memory_dir: PathBuf,
        /// Directory for crystallized modules.
        #[arg(long, default_value = "modules")]
        modules_dir: PathBuf,
        /// Model file to load (or create if missing).
        #[arg(long, default_value = "seed.clob")]
        model: PathBuf,
        /// Config preset: "tiny" or "seed".
        #[arg(long, default_value = "tiny")]
        config: String,
        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Episodic memory capacity.
        #[arg(long, default_value_t = 10000)]
        memory_capacity: usize,
    },
    /// Interactive REPL: type input, observe output + energy + modules.
    Think {
        /// Model file to load.
        #[arg(long)]
        model: PathBuf,
        /// Max tokens to generate per turn.
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        /// Temperature.
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
    },
    /// Generate a synthetic model for testing.
    Synth {
        /// Output path.
        #[arg(long, default_value = "seed.clob")]
        output: PathBuf,
        /// Config: "tiny" or "seed".
        #[arg(long, default_value = "tiny")]
        config: String,
        /// Random seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Benchmark throughput.
    Bench {
        /// Config: "tiny" or "seed".
        #[arg(long, default_value = "tiny")]
        config: String,
        /// Prefill length.
        #[arg(long, default_value_t = 32)]
        seq_len: usize,
        /// Decode steps.
        #[arg(long, default_value_t = 128)]
        decode_steps: usize,
    },
    /// Show model info.
    Info {
        #[arg(long)]
        model: PathBuf,
    },
    /// Show crystallization status.
    Status {
        #[arg(long, default_value = "episodes")]
        memory_dir: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Boot { memory_dir, modules_dir, model, config, seed, memory_capacity } => {
            cmd_boot(&memory_dir, &modules_dir, &model, &config, seed, memory_capacity);
        }
        Commands::Think { model, max_tokens, temperature } => {
            cmd_think(&model, max_tokens, temperature);
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

fn format_params(count: usize) -> String {
    if count >= 1_000_000_000 { format!("{:.1}B", count as f64 / 1e9) }
    else if count >= 1_000_000 { format!("{:.1}M", count as f64 / 1e6) }
    else if count >= 1_000 { format!("{:.1}K", count as f64 / 1e3) }
    else { format!("{}", count) }
}

fn now_nanos() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

fn cmd_boot(memory_dir: &PathBuf, _modules_dir: &PathBuf, model_path: &PathBuf, config_str: &str, seed: u64, memory_capacity: usize) {
    let config = parse_config(config_str);
    eprintln!("[kernel] Config: d_model={}, n_layers={}, vocab={}, params=~{}",
        config.d_model, config.n_layers, config.vocab_size, format_params(config.param_count_estimate()));

    // Create or load model
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

    // Initialize episodic memory
    let memory = EpisodicMemory::open(memory_dir, memory_capacity)
        .expect("failed to open episodic memory");
    eprintln!("[kernel] Episodic memory: {:?} (capacity={})", memory_dir, memory_capacity);

    // Initialize crystallization engine
    let mut crystal_engine = CrystallizationEngine::new(config.d_model, CrystalConfig::default());
    let mut novelty_detector = NoveltyDetector::new();

    eprintln!("\n=== THE KERNEL IS ALIVE ===\n");
    eprintln!("Entering main loop. Type input and press Enter.");
    eprintln!("Commands: :status, :crystal, :reset, :quit\n");

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let sampling = SamplingConfig { temperature: 0.7, top_k: 50, top_p: 0.9 };
    let mut context_tokens: Vec<u32> = Vec::new();
    let mut step_count: u64 = 0;

    // Main loop
    loop {
        // Read input
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() { break; }
        let input = input.trim();

        if input.is_empty() { continue; }

        match input {
            ":quit" | ":q" => {
                eprintln!("[kernel] Shutting down.");
                break;
            }
            ":status" => {
                let mem_stats = memory.stats();
                eprintln!("[status] Steps: {}", step_count);
                eprintln!("[status] Memory: {}/{} episodes ({} unconsumed, {} consumed, {:.2} MB)",
                    mem_stats.total_episodes, mem_stats.capacity,
                    mem_stats.unconsumed, mem_stats.consumed,
                    mem_stats.total_bytes as f64 / (1024.0 * 1024.0));
                eprintln!("[status] Crystal: {}", crystal_engine.stats());
                eprintln!("[status] Novelty baseline: {:.4}", novelty_detector.baseline());
                eprintln!("[status] Context length: {} tokens", context_tokens.len());
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

        // Tokenize input (character-level for now)
        let input_tokens: Vec<u32> = input.bytes()
            .map(|b| (b as u32).min(config.vocab_size as u32 - 1))
            .collect();

        // Prefill with input
        let start = Instant::now();
        let logits = model.prefill(&input_tokens);
        context_tokens.extend_from_slice(&input_tokens);

        // Check energy/novelty for this input
        let (energy, is_novel) = model.energy_score_and_detect();
        let novelty = novelty_detector.evaluate(energy);

        if novelty.is_novel {
            // Store episode
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
                now_nanos(),
                context_tokens.clone(),
                model.last_hidden().data().to_vec(),
                energy,
                top_k,
                0, // actual_token unknown at this point
            );

            if let Err(e) = memory.store(&episode) {
                eprintln!("[memory] Failed to store episode: {}", e);
            }
        }

        // Apply crystallized modules to hidden state
        // (this modifies the hidden state before generation)
        // Note: in a full implementation, this would be integrated into the
        // forward pass. For now, we apply modules post-hoc.

        // Generate response
        let generated = generate::generate(&mut model, &[], 64, &sampling, &mut rng);
        let elapsed = start.elapsed();

        // Decode output (character-level)
        let output: String = generated.iter()
            .map(|&t| if t < 128 { t as u8 as char } else { '?' })
            .collect();

        context_tokens.extend_from_slice(&generated);
        step_count += 1;

        // Display
        println!("{}", output);
        eprintln!("  [{:.1}ms | {:.0} tok/s | energy={:.4} {} | baseline={:.4} | modules={}]",
            elapsed.as_secs_f64() * 1000.0,
            generated.len() as f64 / elapsed.as_secs_f64(),
            energy,
            if is_novel { "▲NOVEL" } else { "○" },
            novelty_detector.baseline(),
            crystal_engine.n_modules());
    }
}

fn cmd_think(model_path: &PathBuf, max_tokens: usize, temperature: f32) {
    eprintln!("[kernel] Loading model from {:?}", model_path);
    let mut model = io::loader::load_model(model_path).expect("failed to load");
    let config = model.config.clone();

    eprintln!("[kernel] d_model={}, n_layers={}, vocab={}", config.d_model, config.n_layers, config.vocab_size);
    eprintln!("Type input, press Enter. :quit to exit.\n");

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let sampling = SamplingConfig { temperature, top_k: 50, top_p: 0.9 };

    loop {
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() { break; }
        let input = input.trim();
        if input == ":quit" || input == ":q" { break; }
        if input.is_empty() { continue; }

        let tokens: Vec<u32> = input.bytes().map(|b| (b as u32).min(config.vocab_size as u32 - 1)).collect();
        let start = Instant::now();
        let _logits = model.prefill(&tokens);
        let generated = generate::generate(&mut model, &[], max_tokens, &sampling, &mut rng);
        let elapsed = start.elapsed();

        let output: String = generated.iter().map(|&t| if t < 128 { t as u8 as char } else { '?' }).collect();
        println!("{}", output);
        eprintln!("  [{:.1}ms | {:.0} tok/s | energy={:.4}]",
            elapsed.as_secs_f64() * 1000.0,
            generated.len() as f64 / elapsed.as_secs_f64(),
            model.energy_score());

        model.reset_state();
    }
}

fn cmd_synth(output: &PathBuf, config_str: &str, seed: u64) {
    let config = parse_config(config_str);
    eprintln!("[synth] Config: d_model={}, n_layers={}, vocab={}, params=~{}",
        config.d_model, config.n_layers, config.vocab_size, format_params(config.param_count_estimate()));

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    io::synth::generate_synthetic(&config, output, &mut rng).expect("failed to generate");
    let file_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    eprintln!("[synth] Done: {:?} ({:.2} MB)", output, file_size as f64 / (1024.0 * 1024.0));
}

fn cmd_bench(config_str: &str, seq_len: usize, decode_steps: usize) {
    let config = parse_config(config_str);
    eprintln!("[bench] Config: d_model={}, n_layers={}, vocab={}, params=~{}",
        config.d_model, config.n_layers, config.vocab_size, format_params(config.param_count_estimate()));

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut model = CoreModel::random(config.clone(), &mut rng);

    // Random input
    let prompt: Vec<u32> = (0..seq_len).map(|_| rng.gen_range(0..config.vocab_size) as u32).collect();

    // Prefill
    eprintln!("\n--- Prefill ({} tokens) ---", seq_len);
    let start = Instant::now();
    let _ = model.prefill(&prompt);
    let prefill = start.elapsed();
    eprintln!("  TTFT: {:.2} ms", prefill.as_secs_f64() * 1000.0);
    eprintln!("  Throughput: {:.1} tok/s", seq_len as f64 / prefill.as_secs_f64());

    // Decode
    eprintln!("\n--- Decode ({} steps) ---", decode_steps);
    let sampling = SamplingConfig::default();
    let start = Instant::now();
    let tokens = generate::generate(&mut model, &[], decode_steps, &sampling, &mut rng);
    let decode = start.elapsed();
    let tok_s = tokens.len() as f64 / decode.as_secs_f64();
    eprintln!("  Generated: {} tokens", tokens.len());
    eprintln!("  Throughput: {:.1} tok/s", tok_s);
    eprintln!("  Latency: {:.2} ms/tok", decode.as_secs_f64() * 1000.0 / tokens.len() as f64);
    eprintln!("  Energy: {:.4}", model.energy_score());

    // Peak RSS
    #[cfg(target_os = "linux")]
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            eprintln!("  Peak RSS: {} MB", ru.ru_maxrss / 1024);
        }
    }

    eprintln!("\n--- Summary ---");
    eprintln!("  TTFT: {:.2} ms | Decode: {:.1} tok/s", prefill.as_secs_f64() * 1000.0, tok_s);
}

fn cmd_info(model_path: &PathBuf) {
    let file = std::fs::File::open(model_path).expect("failed to open");
    let mut reader = std::io::BufReader::new(file);
    let (header, header_size) = clob::io::format::read_header(&mut reader).expect("failed to read header");
    let file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);

    let config = &header.config;
    println!("=== The Kernel ===");
    println!("File: {:?}  ({:.2} MB)", model_path, file_size as f64 / (1024.0 * 1024.0));
    println!("Header: {} bytes", header_size);
    println!();
    println!("--- Configuration ---");
    println!("  d_model:     {}", config.d_model);
    println!("  n_layers:    {}", config.n_layers);
    println!("  d_state:     {}", config.d_state);
    println!("  n_heads:     {}", config.n_heads);
    println!("  d_inner:     {}", config.d_inner);
    println!("  n_experts:   {}", config.n_experts);
    println!("  n_active:    {}", config.n_active_experts);
    println!("  vocab_size:  {}", config.vocab_size);
    println!("  moe_layers:  {:?}", config.moe_layers);
    println!();
    println!("--- Estimates ---");
    println!("  Parameters:     ~{}", format_params(config.param_count_estimate()));
    println!("  Weight entries: {}", header.weight_entries.len());
    println!("  Tensor data:    {:.2} MB", (file_size as usize - header_size) as f64 / (1024.0 * 1024.0));
}

fn cmd_status(memory_dir: &PathBuf) {
    match EpisodicMemory::open(memory_dir, usize::MAX) {
        Ok(memory) => {
            let stats = memory.stats();
            println!("=== Episodic Memory ===");
            println!("  Directory:      {:?}", memory_dir);
            println!("  Episodes:       {}", stats.total_episodes);
            println!("  Unconsumed:     {}", stats.unconsumed);
            println!("  Consumed:       {}", stats.consumed);
            println!("  Total written:  {}", stats.total_ever_written);
            println!("  Size on disk:   {:.2} MB", stats.total_bytes as f64 / (1024.0 * 1024.0));
        }
        Err(e) => {
            eprintln!("[status] Failed to open episodic memory: {}", e);
        }
    }
}
