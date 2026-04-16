//! Synthetic model generator — creates .clob files with random weights.

use crate::io::format::{self, WeightDtype, WeightEntry};
use crate::model::config::KernelConfig;
use crate::tensor::ternary::TernaryMatrix;
use rand::Rng;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Accumulator for weight entries and raw tensor data.
struct WeightAccum {
    entries: Vec<WeightEntry>,
    data: Vec<u8>,
    offset: u64,
}

impl WeightAccum {
    fn new() -> Self {
        Self { entries: Vec::new(), data: Vec::new(), offset: 0 }
    }

    fn push_f32(&mut self, name: &str, values: &[f32], shape: Vec<usize>) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let size = bytes.len() as u64;
        self.entries.push(WeightEntry {
            name: name.to_string(), offset: self.offset, size,
            dtype: WeightDtype::F32, shape,
        });
        self.data.extend_from_slice(&bytes);
        self.offset += size;
    }

    fn push_ternary<R: Rng>(&mut self, name: &str, rows: usize, cols: usize, rng: &mut R) {
        let mat = TernaryMatrix::random(rows, cols, rng);

        let packed = mat.packed_data();
        self.entries.push(WeightEntry {
            name: format!("{}.packed", name), offset: self.offset,
            size: packed.len() as u64,
            dtype: WeightDtype::TernaryPacked,
            shape: vec![rows, cols],
        });
        self.data.extend_from_slice(packed);
        self.offset += packed.len() as u64;

        let scales_bytes: Vec<u8> = mat.scales().iter().flat_map(|v| v.to_le_bytes()).collect();
        self.entries.push(WeightEntry {
            name: format!("{}.scales", name), offset: self.offset,
            size: scales_bytes.len() as u64,
            dtype: WeightDtype::F32,
            shape: vec![rows],
        });
        self.data.extend_from_slice(&scales_bytes);
        self.offset += scales_bytes.len() as u64;
    }

    fn push_ternary_linear<R: Rng>(&mut self, prefix: &str, in_f: usize, out_f: usize, with_bias: bool, rng: &mut R) {
        self.push_ternary(prefix, out_f, in_f, rng);
        if with_bias {
            let bias: Vec<f32> = (0..out_f).map(|_| rng.gen_range(-0.1..0.1)).collect();
            self.push_f32(&format!("{}.bias", prefix), &bias, vec![out_f]);
        }
    }
}



/// Generate a synthetic .clob model file.
pub fn generate_synthetic(config: &KernelConfig, path: &Path, rng: &mut impl Rng) -> std::io::Result<()> {
    let mut acc = WeightAccum::new();
    let d = config.d_model;
    let scale = 1.0 / (d as f32).sqrt();

    // Embedding
    let embed: Vec<f32> = (0..config.vocab_size * d).map(|_| rng.gen_range(-scale..scale)).collect();
    acc.push_f32("embed.table", &embed, vec![config.vocab_size, d]);

    // Blocks
    for i in 0..config.n_layers {
        let prefix = format!("block.{}", i);
        let n_heads = config.n_heads;
        let d_state = config.d_state;
        let d_inner = d;
        let x_proj_size = n_heads + 2 * n_heads * d_state;

        // norm1
        let ones = vec![1.0f32; d];
        acc.push_f32(&format!("{}.norm1.weight", prefix), &ones, vec![d]);

        // SSM
        acc.push_ternary_linear(&format!("{}.ssm.in_proj", prefix), d, d_inner, false, rng);
        acc.push_ternary_linear(&format!("{}.ssm.x_proj", prefix), d_inner, x_proj_size, false, rng);
        acc.push_ternary_linear(&format!("{}.ssm.dt_proj", prefix), n_heads, d_inner, false, rng);
        acc.push_ternary_linear(&format!("{}.ssm.out_proj", prefix), d_inner, d, false, rng);

        let a_log: Vec<f32> = (0..n_heads * d_state).map(|_| rng.gen_range(-2.0..-0.5)).collect();
        acc.push_f32(&format!("{}.ssm.a_log", prefix), &a_log, vec![n_heads, d_state]);
        let d_param: Vec<f32> = (0..n_heads).map(|_| rng.gen_range(0.5..1.5)).collect();
        acc.push_f32(&format!("{}.ssm.d_param", prefix), &d_param, vec![n_heads]);
        let dt_bias: Vec<f32> = (0..n_heads).map(|_| rng.gen_range(-0.5..0.5)).collect();
        acc.push_f32(&format!("{}.ssm.dt_bias", prefix), &dt_bias, vec![n_heads]);

        // norm2
        acc.push_f32(&format!("{}.norm2.weight", prefix), &ones, vec![d]);

        // Channel mixer
        if config.is_moe_layer(i) {
            let router_w: Vec<f32> = (0..config.n_experts * d).map(|_| rng.gen_range(-scale..scale)).collect();
            acc.push_f32(&format!("{}.router.weights", prefix), &router_w, vec![config.n_experts, d]);

            for e in 0..config.n_experts {
                let ep = format!("{}.expert.{}", prefix, e);
                acc.push_ternary_linear(&format!("{}.gate", ep), d, config.d_inner, false, rng);
                acc.push_ternary_linear(&format!("{}.up", ep), d, config.d_inner, false, rng);
                acc.push_ternary_linear(&format!("{}.down", ep), config.d_inner, d, false, rng);
            }
        } else {
            acc.push_ternary_linear(&format!("{}.mlgru.w_f", prefix), d, d, true, rng);
            acc.push_ternary_linear(&format!("{}.mlgru.w_c", prefix), d, d, true, rng);
            acc.push_ternary_linear(&format!("{}.mlgru.w_o", prefix), d, d, true, rng);
            acc.push_ternary_linear(&format!("{}.glu.gate", prefix), d, config.d_inner, false, rng);
            acc.push_ternary_linear(&format!("{}.glu.up", prefix), d, config.d_inner, false, rng);
            acc.push_ternary_linear(&format!("{}.glu.down", prefix), config.d_inner, d, false, rng);
        }
    }

    // Final norm
    let ones = vec![1.0f32; d];
    acc.push_f32("final_norm.weight", &ones, vec![d]);

    // Energy critic
    let energy_w: Vec<f32> = (0..d).map(|_| rng.gen_range(-scale..scale)).collect();
    acc.push_f32("energy.weights", &energy_w, vec![d]);
    acc.push_f32("energy.bias", &[0.0f32], vec![1]);

    // Write file
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    format::write_header(&mut writer, config, &acc.entries)?;
    writer.write_all(&acc.data)?;
    writer.flush()?;

    eprintln!("[synth] Wrote {} weight entries, {:.2} MB tensor data",
        acc.entries.len(), acc.data.len() as f64 / (1024.0 * 1024.0));

    Ok(())
}
