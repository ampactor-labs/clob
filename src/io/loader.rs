//! Weight hydration: .clob bytes → model structs.

use crate::io::format;
use crate::model::block::{Block, ChannelMixer};
use crate::model::router::ExpertRouter;
use crate::model::stack::CoreModel;
use crate::nn::embed::Embedding;
use crate::nn::energy::EnergyCritic;
use crate::nn::glu::TernaryGLU;
use crate::nn::mlgru::MLGRU;
use crate::nn::rmsnorm::RMSNorm;
use crate::nn::ssm::SelectiveSSM;
use crate::nn::ternary_linear::TernaryLinear;
use crate::tensor::ternary::TernaryMatrix;
use std::io::BufReader;
use std::path::Path;

/// Load a complete model from a .clob file.
pub fn load_model(path: &Path) -> std::io::Result<CoreModel> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let (header, _header_size) = format::read_header(&mut reader)?;

    // Read all tensor data
    let mut tensor_data = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut tensor_data)?;

    let config = header.config;
    let entries = &header.weight_entries;

    // Helper to extract f32 data
    let get_f32 = |name: &str| -> Vec<f32> {
        let entry = entries.iter().find(|e| e.name == name)
            .unwrap_or_else(|| panic!("missing weight: {}", name));
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        let bytes = &tensor_data[start..end];
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };

    // Helper to extract ternary matrix
    let get_ternary = |name: &str, rows: usize, cols: usize| -> TernaryMatrix {
        let packed_name = format!("{}.packed", name);
        let scales_name = format!("{}.scales", name);

        let packed_entry = entries.iter().find(|e| e.name == packed_name)
            .unwrap_or_else(|| panic!("missing weight: {}", packed_name));
        let start = packed_entry.offset as usize;
        let end = start + packed_entry.size as usize;
        let packed = tensor_data[start..end].to_vec();

        let scales = get_f32(&scales_name);
        TernaryMatrix::from_raw(packed, scales, rows, cols)
    };

    let get_ternary_linear = |prefix: &str, in_f: usize, out_f: usize, with_bias: bool| -> TernaryLinear {
        let weight = get_ternary(prefix, out_f, in_f);
        let bias = if with_bias {
            Some(get_f32(&format!("{}.bias", prefix)))
        } else {
            None
        };
        TernaryLinear::new(weight, bias)
    };

    // Embedding
    let embed_table = get_f32("embed.table");
    let embedding = Embedding::new(embed_table, config.vocab_size, config.d_model);

    // Blocks
    let mut blocks = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let prefix = format!("block.{}", i);

        let norm1_w = get_f32(&format!("{}.norm1.weight", prefix));
        let norm1 = RMSNorm::new(norm1_w, 1e-6);

        let d = config.d_model;
        let n_heads = config.n_heads;
        let d_state = config.d_state;
        let d_inner = d; // SSM d_inner = d_model
        let d_head = d / n_heads;
        let x_proj_size = n_heads + 2 * n_heads * d_state;

        let ssm = SelectiveSSM::from_weights(
            get_ternary_linear(&format!("{}.ssm.in_proj", prefix), d, d_inner, false),
            get_ternary_linear(&format!("{}.ssm.x_proj", prefix), d_inner, x_proj_size, false),
            get_ternary_linear(&format!("{}.ssm.dt_proj", prefix), n_heads, d_inner, false),
            get_ternary_linear(&format!("{}.ssm.out_proj", prefix), d_inner, d, false),
            get_f32(&format!("{}.ssm.a_log", prefix)),
            get_f32(&format!("{}.ssm.d_param", prefix)),
            get_f32(&format!("{}.ssm.dt_bias", prefix)),
            n_heads, d_state, d_inner, d_head,
        );

        let norm2_w = get_f32(&format!("{}.norm2.weight", prefix));
        let norm2 = RMSNorm::new(norm2_w, 1e-6);

        let channel_mixer = if config.is_moe_layer(i) {
            let router_w = get_f32(&format!("{}.router.weights", prefix));
            let router = ExpertRouter::from_weights(
                router_w, config.d_model, config.n_experts, config.n_active_experts,
            );
            let experts: Vec<TernaryGLU> = (0..config.n_experts).map(|e| {
                let ep = format!("{}.expert.{}", prefix, e);
                TernaryGLU::from_weights(
                    get_ternary_linear(&format!("{}.gate", ep), d, config.d_inner, false),
                    get_ternary_linear(&format!("{}.up", ep), d, config.d_inner, false),
                    get_ternary_linear(&format!("{}.down", ep), config.d_inner, d, false),
                    d, config.d_inner,
                )
            }).collect();
            ChannelMixer::MoE { router, experts }
        } else {
            let mlgru = MLGRU::from_weights(
                get_ternary_linear(&format!("{}.mlgru.w_f", prefix), d, d, true),
                get_ternary_linear(&format!("{}.mlgru.w_c", prefix), d, d, true),
                get_ternary_linear(&format!("{}.mlgru.w_o", prefix), d, d, true),
                d,
            );
            let glu = TernaryGLU::from_weights(
                get_ternary_linear(&format!("{}.glu.gate", prefix), d, config.d_inner, false),
                get_ternary_linear(&format!("{}.glu.up", prefix), d, config.d_inner, false),
                get_ternary_linear(&format!("{}.glu.down", prefix), config.d_inner, d, false),
                d, config.d_inner,
            );
            ChannelMixer::Dense { mlgru, glu }
        };

        blocks.push(Block::from_parts(norm1, ssm, norm2, channel_mixer, i, d));
    }

    let final_norm_w = get_f32("final_norm.weight");
    let final_norm = RMSNorm::new(final_norm_w, 1e-6);

    let energy_critic = if let Some(entry) = entries.iter().find(|e| e.name == "energy.weights") {
        let _ = entry;
        let w = get_f32("energy.weights");
        let b = entries.iter().find(|e| e.name == "energy.bias")
            .map(|_| get_f32("energy.bias")[0])
            .unwrap_or(0.0);
        EnergyCritic::new(w, b)
    } else {
        let mut rng = rand::thread_rng();
        EnergyCritic::random(config.d_model, &mut rng)
    };

    Ok(CoreModel::from_parts(config, embedding, blocks, final_norm, energy_critic))
}
