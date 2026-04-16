//! Block: SSM (token mixer) + channel mixer with pre-norm + residual.

use crate::model::config::KernelConfig;
use crate::model::router::{ExpertRouter, RouteResult};
use crate::nn::glu::TernaryGLU;
use crate::nn::mlgru::MLGRU;
use crate::nn::rmsnorm::RMSNorm;
use crate::nn::ssm::SelectiveSSM;
use crate::simd::KernelDispatch;
use crate::tensor::Tensor;
use rand::Rng;

/// Channel mixer variant.
pub enum ChannelMixer {
    Dense { mlgru: MLGRU, glu: TernaryGLU },
    MoE { router: ExpertRouter, experts: Vec<TernaryGLU> },
}

/// A single block: SSM + channel mixer.
pub struct Block {
    norm1: RMSNorm,
    ssm: SelectiveSSM,
    norm2: RMSNorm,
    pub channel_mixer: ChannelMixer,
    pub layer_idx: usize,
    buf_residual: Tensor,
    buf_ssm_out: Tensor,
    buf_mixer_out: Tensor,
    buf_gru_out: Option<Tensor>,
}

impl Block {
    pub fn random(config: &KernelConfig, layer_idx: usize, rng: &mut impl Rng) -> Self {
        let is_moe = config.is_moe_layer(layer_idx);

        let channel_mixer = if is_moe {
            let experts: Vec<TernaryGLU> = (0..config.n_experts)
                .map(|_| TernaryGLU::random(config.d_model, config.d_inner, rng))
                .collect();
            ChannelMixer::MoE {
                router: ExpertRouter::random(config.d_model, config.n_experts, config.n_active_experts, rng),
                experts,
            }
        } else {
            ChannelMixer::Dense {
                mlgru: MLGRU::random(config.d_model, rng),
                glu: TernaryGLU::random(config.d_model, config.d_inner, rng),
            }
        };

        Self {
            norm1: RMSNorm::ones(config.d_model),
            ssm: SelectiveSSM::random(config.d_model, config.n_heads, config.d_state, rng),
            norm2: RMSNorm::ones(config.d_model),
            channel_mixer,
            layer_idx,
            buf_residual: Tensor::zeros(&[config.d_model]),
            buf_ssm_out: Tensor::zeros(&[config.d_model]),
            buf_mixer_out: Tensor::zeros(&[config.d_model]),
            buf_gru_out: if !is_moe { Some(Tensor::zeros(&[config.d_model])) } else { None },
        }
    }

    pub fn from_parts(
        norm1: RMSNorm, ssm: SelectiveSSM, norm2: RMSNorm,
        channel_mixer: ChannelMixer, layer_idx: usize, d_model: usize,
    ) -> Self {
        let is_moe = matches!(channel_mixer, ChannelMixer::MoE { .. });
        Self {
            norm1, ssm, norm2, channel_mixer, layer_idx,
            buf_residual: Tensor::zeros(&[d_model]),
            buf_ssm_out: Tensor::zeros(&[d_model]),
            buf_mixer_out: Tensor::zeros(&[d_model]),
            buf_gru_out: if !is_moe { Some(Tensor::zeros(&[d_model])) } else { None },
        }
    }

    /// Forward: single token step. Zero heap allocations.
    pub fn forward(&mut self, x: &mut Tensor, dispatch: &KernelDispatch) -> Option<RouteResult> {
        // Save residual
        self.buf_residual.copy_from(x);

        // Token mixer: RMSNorm → SSM → + residual
        self.norm1.forward(x, dispatch);
        self.ssm.forward(x, &mut self.buf_ssm_out, dispatch);
        x.copy_from(&self.buf_residual);
        x.add_(&self.buf_ssm_out);

        // Save residual
        self.buf_residual.copy_from(x);

        // Channel mixer: RMSNorm → MoE/Dense → + residual
        self.norm2.forward(x, dispatch);

        let route_result = match &mut self.channel_mixer {
            ChannelMixer::Dense { mlgru, glu } => {
                let gru_out = self.buf_gru_out.as_mut().unwrap();
                mlgru.forward(x, gru_out, dispatch);
                glu.forward(gru_out, &mut self.buf_mixer_out, dispatch);
                None
            }
            ChannelMixer::MoE { router, experts } => {
                let result = router.route(x);
                // Weighted combination of active experts
                self.buf_mixer_out.zero_();
                let mut expert_out = Tensor::zeros(&[x.len()]);
                for (&idx, &weight) in result.expert_indices.iter().zip(result.expert_weights.iter()) {
                    experts[idx].forward(x, &mut expert_out, dispatch);
                    for (o, e) in self.buf_mixer_out.data_mut().iter_mut().zip(expert_out.data().iter()) {
                        *o += weight * e;
                    }
                }
                Some(result)
            }
        };

        x.copy_from(&self.buf_residual);
        x.add_(&self.buf_mixer_out);

        route_result
    }

    pub fn reset_state(&mut self) {
        self.ssm.reset_state();
        if let ChannelMixer::Dense { mlgru, .. } = &mut self.channel_mixer {
            mlgru.reset_state();
        }
    }

    pub fn is_moe(&self) -> bool {
        matches!(self.channel_mixer, ChannelMixer::MoE { .. })
    }
}
