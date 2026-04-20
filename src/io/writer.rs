//! Model save path — mirror of `src/io/synth.rs::generate_synthetic` that
//! serializes an existing `CoreModel` rather than building one with random
//! weights.
//!
//! Drift between save and load is the single most expensive bug this code
//! could introduce: if the names or order here disagree with
//! `src/io/loader.rs::load_model`, a saved model becomes unloadable. The
//! `save_load_round_trip` test guards against it.

use crate::io::format::{self, WeightDtype, WeightEntry};
use crate::model::block::ChannelMixer;
use crate::model::stack::CoreModel;
use crate::tensor::ternary::TernaryMatrix;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Accumulator for weight entries and raw tensor data. Public in this
/// module so tests (and future phases that want to add new tensors to an
/// existing bundle) can reuse it.
pub struct WeightAccum {
    pub entries: Vec<WeightEntry>,
    pub data: Vec<u8>,
    pub offset: u64,
}

impl WeightAccum {
    pub fn new() -> Self {
        Self { entries: Vec::new(), data: Vec::new(), offset: 0 }
    }

    pub fn push_f32(&mut self, name: &str, values: &[f32], shape: Vec<usize>) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let size = bytes.len() as u64;
        self.entries.push(WeightEntry {
            name: name.to_string(), offset: self.offset, size,
            dtype: WeightDtype::F32, shape,
        });
        self.data.extend_from_slice(&bytes);
        self.offset += size;
    }

    pub fn push_ternary_matrix(&mut self, name: &str, mat: &TernaryMatrix) {
        let packed = mat.packed_data();
        self.entries.push(WeightEntry {
            name: format!("{}.packed", name),
            offset: self.offset,
            size: packed.len() as u64,
            dtype: WeightDtype::TernaryPacked,
            shape: vec![mat.rows(), mat.cols()],
        });
        self.data.extend_from_slice(packed);
        self.offset += packed.len() as u64;

        let scales_bytes: Vec<u8> = mat.scales().iter().flat_map(|v| v.to_le_bytes()).collect();
        self.entries.push(WeightEntry {
            name: format!("{}.scales", name),
            offset: self.offset,
            size: scales_bytes.len() as u64,
            dtype: WeightDtype::F32,
            shape: vec![mat.rows()],
        });
        self.data.extend_from_slice(&scales_bytes);
        self.offset += scales_bytes.len() as u64;
    }

    /// Push a TernaryLinear (weight + optional bias) using the naming
    /// convention the loader expects: `{prefix}.packed`, `{prefix}.scales`,
    /// and (optionally) `{prefix}.bias`.
    pub fn push_ternary_linear(
        &mut self,
        prefix: &str,
        linear: &crate::nn::ternary_linear::TernaryLinear,
        with_bias: bool,
    ) {
        self.push_ternary_matrix(prefix, linear.weight_mat());
        if with_bias {
            let bias = linear.bias_ref().unwrap_or_else(|| {
                panic!("expected bias on {} but TernaryLinear has none", prefix)
            });
            self.push_f32(&format!("{}.bias", prefix), bias, vec![bias.len()]);
        }
    }
}

impl Default for WeightAccum {
    fn default() -> Self { Self::new() }
}

/// Save `model` to the given path as a .clob file. Byte-identical
/// serialization for two models whose every component is pairwise equal.
pub fn save_model(model: &CoreModel, path: &Path) -> std::io::Result<()> {
    let mut acc = WeightAccum::new();
    let config = &model.config;
    let d = config.d_model;

    // Embedding — mirrors synth.rs's order exactly.
    acc.push_f32(
        "embed.table",
        model.embedding().table(),
        vec![config.vocab_size, d],
    );

    // Blocks
    for (i, block) in model.blocks().iter().enumerate() {
        let prefix = format!("block.{}", i);

        acc.push_f32(
            &format!("{}.norm1.weight", prefix),
            block.norm1().weight_slice(),
            vec![d],
        );

        let ssm = block.ssm();
        acc.push_ternary_linear(
            &format!("{}.ssm.in_proj", prefix), ssm.in_proj(), false,
        );
        acc.push_ternary_linear(
            &format!("{}.ssm.x_proj", prefix), ssm.x_proj(), false,
        );
        acc.push_ternary_linear(
            &format!("{}.ssm.dt_proj", prefix), ssm.dt_proj(), false,
        );
        acc.push_ternary_linear(
            &format!("{}.ssm.out_proj", prefix), ssm.out_proj(), false,
        );
        acc.push_f32(
            &format!("{}.ssm.a_log", prefix),
            ssm.a_log(),
            vec![ssm.n_heads(), ssm.d_state()],
        );
        acc.push_f32(
            &format!("{}.ssm.d_param", prefix),
            ssm.d_param(),
            vec![ssm.n_heads()],
        );
        acc.push_f32(
            &format!("{}.ssm.dt_bias", prefix),
            ssm.dt_bias(),
            vec![ssm.n_heads()],
        );

        acc.push_f32(
            &format!("{}.norm2.weight", prefix),
            block.norm2().weight_slice(),
            vec![d],
        );

        match block.channel_mixer() {
            ChannelMixer::MoE { router, experts } => {
                acc.push_f32(
                    &format!("{}.router.weights", prefix),
                    router.weights(),
                    vec![config.n_experts, d],
                );
                for (e, expert) in experts.iter().enumerate() {
                    let ep = format!("{}.expert.{}", prefix, e);
                    acc.push_ternary_linear(
                        &format!("{}.gate", ep), expert.w_gate(), false,
                    );
                    acc.push_ternary_linear(
                        &format!("{}.up", ep), expert.w_up(), false,
                    );
                    acc.push_ternary_linear(
                        &format!("{}.down", ep), expert.w_down(), false,
                    );
                }
            }
            ChannelMixer::Dense { mlgru, glu } => {
                acc.push_ternary_linear(
                    &format!("{}.mlgru.w_f", prefix), mlgru.w_f(), true,
                );
                acc.push_ternary_linear(
                    &format!("{}.mlgru.w_c", prefix), mlgru.w_c(), true,
                );
                acc.push_ternary_linear(
                    &format!("{}.mlgru.w_o", prefix), mlgru.w_o(), true,
                );
                acc.push_ternary_linear(
                    &format!("{}.glu.gate", prefix), glu.w_gate(), false,
                );
                acc.push_ternary_linear(
                    &format!("{}.glu.up", prefix), glu.w_up(), false,
                );
                acc.push_ternary_linear(
                    &format!("{}.glu.down", prefix), glu.w_down(), false,
                );
            }
        }
    }

    acc.push_f32(
        "final_norm.weight",
        model.final_norm().weight_slice(),
        vec![d],
    );

    let critic = model.energy_critic();
    acc.push_f32("energy.weights", critic.weights(), vec![d]);
    acc.push_f32("energy.bias", &[critic.bias()], vec![1]);

    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    format::write_header(&mut writer, config, &acc.entries)?;
    writer.write_all(&acc.data)?;
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::loader::load_model;
    use crate::model::config::KernelConfig;
    use rand::SeedableRng;

    /// Compare two f32 slices exactly, bit-for-bit.
    fn eq_f32(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn eq_ternary(a: &TernaryMatrix, b: &TernaryMatrix) -> bool {
        a.rows() == b.rows()
            && a.cols() == b.cols()
            && a.unpack() == b.unpack()
            && eq_f32(a.scales(), b.scales())
    }

    fn eq_ternary_linear(
        a: &crate::nn::ternary_linear::TernaryLinear,
        b: &crate::nn::ternary_linear::TernaryLinear,
    ) -> bool {
        eq_ternary(a.weight_mat(), b.weight_mat())
            && match (a.bias_ref(), b.bias_ref()) {
                (None, None) => true,
                (Some(x), Some(y)) => eq_f32(x, y),
                _ => false,
            }
    }

    #[test]
    fn save_load_round_trip_tiny() {
        let config = KernelConfig::tiny();
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let original = CoreModel::random(config.clone(), &mut rng);

        let tmp = std::env::temp_dir().join("clob_writer_round_trip_tiny.clob");
        save_model(&original, &tmp).expect("save failed");
        let restored = load_model(&tmp).expect("load failed");

        assert_eq!(restored.config.d_model, original.config.d_model);
        assert_eq!(restored.config.n_layers, original.config.n_layers);
        assert_eq!(restored.config.vocab_size, original.config.vocab_size);

        assert!(eq_f32(restored.embedding().table(), original.embedding().table()));
        assert!(eq_f32(
            restored.final_norm().weight_slice(),
            original.final_norm().weight_slice(),
        ));
        assert!(eq_f32(
            restored.energy_critic().weights(),
            original.energy_critic().weights(),
        ));
        assert_eq!(
            restored.energy_critic().bias().to_bits(),
            original.energy_critic().bias().to_bits(),
        );
        assert_eq!(restored.blocks().len(), original.blocks().len());

        for (a, b) in restored.blocks().iter().zip(original.blocks().iter()) {
            assert!(eq_f32(a.norm1().weight_slice(), b.norm1().weight_slice()));
            assert!(eq_f32(a.norm2().weight_slice(), b.norm2().weight_slice()));
            assert!(eq_ternary_linear(a.ssm().in_proj(), b.ssm().in_proj()));
            assert!(eq_ternary_linear(a.ssm().x_proj(), b.ssm().x_proj()));
            assert!(eq_ternary_linear(a.ssm().dt_proj(), b.ssm().dt_proj()));
            assert!(eq_ternary_linear(a.ssm().out_proj(), b.ssm().out_proj()));
            assert!(eq_f32(a.ssm().a_log(), b.ssm().a_log()));
            assert!(eq_f32(a.ssm().d_param(), b.ssm().d_param()));
            assert!(eq_f32(a.ssm().dt_bias(), b.ssm().dt_bias()));

            match (a.channel_mixer(), b.channel_mixer()) {
                (
                    ChannelMixer::MoE { router: ra, experts: ea },
                    ChannelMixer::MoE { router: rb, experts: eb },
                ) => {
                    assert!(eq_f32(ra.weights(), rb.weights()));
                    assert_eq!(ea.len(), eb.len());
                    for (xa, xb) in ea.iter().zip(eb.iter()) {
                        assert!(eq_ternary_linear(xa.w_gate(), xb.w_gate()));
                        assert!(eq_ternary_linear(xa.w_up(), xb.w_up()));
                        assert!(eq_ternary_linear(xa.w_down(), xb.w_down()));
                    }
                }
                (
                    ChannelMixer::Dense { mlgru: ma, glu: ga },
                    ChannelMixer::Dense { mlgru: mb, glu: gb },
                ) => {
                    assert!(eq_ternary_linear(ma.w_f(), mb.w_f()));
                    assert!(eq_ternary_linear(ma.w_c(), mb.w_c()));
                    assert!(eq_ternary_linear(ma.w_o(), mb.w_o()));
                    assert!(eq_ternary_linear(ga.w_gate(), gb.w_gate()));
                    assert!(eq_ternary_linear(ga.w_up(), gb.w_up()));
                    assert!(eq_ternary_linear(ga.w_down(), gb.w_down()));
                }
                _ => panic!("channel mixer variant mismatch"),
            }
        }

        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn two_saves_are_byte_identical() {
        let config = KernelConfig::tiny();
        let mut rng = rand::rngs::StdRng::seed_from_u64(17);
        let model = CoreModel::random(config, &mut rng);

        let a_path = std::env::temp_dir().join("clob_writer_bytes_a.clob");
        let b_path = std::env::temp_dir().join("clob_writer_bytes_b.clob");
        save_model(&model, &a_path).unwrap();
        save_model(&model, &b_path).unwrap();

        let a = std::fs::read(&a_path).unwrap();
        let b = std::fs::read(&b_path).unwrap();
        assert_eq!(a.len(), b.len());
        assert_eq!(a, b, "two saves of the same model must be byte-identical");

        let _ = std::fs::remove_file(a_path);
        let _ = std::fs::remove_file(b_path);
    }
}
