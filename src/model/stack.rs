//! CoreModel — the full ternary recurrent stack.
//!
//! Embed → [Block × L] → FinalNorm → LM Head.
//! O(1) decode memory. Zero heap allocations on the hot path.

use crate::crystal::detector::NoveltyDetector;
use crate::crystal::module::CrystalModule;
use crate::model::block::Block;
use crate::model::config::KernelConfig;
use crate::nn::confidence::{AdaptiveConfig, ConfidenceHead, MetaCritic};
use crate::nn::embed::Embedding;
use crate::nn::energy::EnergyCritic;
use crate::nn::readout::TrainedReadout;
use crate::nn::rmsnorm::RMSNorm;
use crate::simd::KernelDispatch;
use crate::tensor::Tensor;
use rand::Rng;

/// The complete ternary recurrent core.
pub struct CoreModel {
    pub config: KernelConfig,
    embedding: Embedding,
    blocks: Vec<Block>,
    final_norm: RMSNorm,
    energy_critic: EnergyCritic,
    dispatch: KernelDispatch,
    last_hidden: Tensor,
    buf_x: Tensor,
    buf_logits: Tensor,
    /// Phase L+: trained untied output head. When present, it replaces the
    /// tied embedding-table unembedding at readout. Installed as a sidecar
    /// (`--readout`), produced by `probe-readout --save-readout`. The frozen
    /// tied readout predicts worse than chance; this is the fix.
    readout: Option<TrainedReadout>,
    /// Crystallized knowledge modules, applied after final_norm.
    /// Additive, scaled by cosine similarity to each module's domain signature.
    crystal_modules: Vec<CrystalModule>,
    /// Minimum cosine similarity for a module to activate.
    pub crystal_activation_threshold: f32,
    /// Scratch buffer for module outputs (d_model).
    buf_module_out: Tensor,
    /// Phase B+C: Head C — predicts per-step NLL. When `adaptive_config.enabled`
    /// is true and this head fires above its detector threshold, the block
    /// stack iterates extra times before unembedding.
    confidence_head: Option<ConfidenceHead>,
    /// Per-head adaptive threshold for Head C.
    confidence_detector: NoveltyDetector,
    /// Phase F: MetaCritic gating Head C's adaptive signal. When its
    /// predicted absolute error for Head C exceeds
    /// `adaptive_config.meta_unreliable_threshold`, we SKIP adaptive
    /// iteration at this hidden state — Head C's uncertainty signal is
    /// itself unreliable.
    meta_critic: Option<MetaCritic>,
    /// Adaptive-decode configuration.
    pub adaptive_config: AdaptiveConfig,
    /// Counter: number of extra SSM iterations taken since last reset. Read
    /// by callers who want to feed it into the metrics window.
    extra_steps_counter: u64,
    /// Counter: number of tokens where adaptive WOULD have fired but
    /// MetaCritic suppressed it. Diagnostic signal for the target-network
    /// stability probe.
    meta_suppressed_counter: u64,
}

impl CoreModel {
    /// Create with random weights for testing.
    pub fn random(config: KernelConfig, rng: &mut impl Rng) -> Self {
        let embedding = Embedding::random(config.vocab_size, config.d_model, rng);
        let blocks: Vec<Block> = (0..config.n_layers)
            .map(|i| Block::random(&config, i, rng))
            .collect();
        let energy_critic = EnergyCritic::random(config.d_model, rng);
        let dispatch = KernelDispatch::new();
        let last_hidden = Tensor::zeros(&[config.d_model]);
        let buf_x = Tensor::zeros(&[config.d_model]);
        let buf_logits = Tensor::zeros(&[config.vocab_size]);

        let buf_module_out = Tensor::zeros(&[config.d_model]);
        Self {
            config: config.clone(),
            embedding,
            blocks,
            final_norm: RMSNorm::ones(config.d_model),
            energy_critic,
            dispatch,
            last_hidden,
            buf_x,
            buf_logits,
            crystal_modules: Vec::new(),
            crystal_activation_threshold: 0.3,
            buf_module_out,
            confidence_head: None,
            readout: None,
            confidence_detector: NoveltyDetector::new(),
            meta_critic: None,
            adaptive_config: AdaptiveConfig::default(),
            extra_steps_counter: 0,
            meta_suppressed_counter: 0,
        }
    }

    /// Create from pre-loaded components.
    pub fn from_parts(
        config: KernelConfig,
        embedding: Embedding,
        blocks: Vec<Block>,
        final_norm: RMSNorm,
        energy_critic: EnergyCritic,
    ) -> Self {
        let dispatch = KernelDispatch::new();
        let last_hidden = Tensor::zeros(&[config.d_model]);
        let buf_x = Tensor::zeros(&[config.d_model]);
        let buf_logits = Tensor::zeros(&[config.vocab_size]);
        let buf_module_out = Tensor::zeros(&[config.d_model]);
        Self {
            config: config.clone(), embedding, blocks, final_norm,
            energy_critic, dispatch, last_hidden, buf_x, buf_logits,
            crystal_modules: Vec::new(),
            crystal_activation_threshold: 0.3,
            buf_module_out,
            confidence_head: None,
            readout: None,
            confidence_detector: NoveltyDetector::new(),
            meta_critic: None,
            adaptive_config: AdaptiveConfig::default(),
            extra_steps_counter: 0,
            meta_suppressed_counter: 0,
        }
    }

    /// Install a trained untied readout. It replaces the tied embedding-table
    /// unembedding at the output. Dimensions must match the model.
    pub fn set_readout(&mut self, readout: TrainedReadout) {
        assert_eq!(
            readout.d_model(), self.config.d_model,
            "readout d_model {} != model d_model {}", readout.d_model(), self.config.d_model,
        );
        assert_eq!(
            readout.vocab_size(), self.config.vocab_size,
            "readout vocab {} != model vocab {}", readout.vocab_size(), self.config.vocab_size,
        );
        self.readout = Some(readout);
    }

    pub fn has_readout(&self) -> bool {
        self.readout.is_some()
    }

    /// Install a pre-trained Head C. Its `dim()` must equal `config.d_model`.
    pub fn set_confidence_head(&mut self, head: ConfidenceHead) {
        assert_eq!(
            head.dim(), self.config.d_model,
            "confidence head dim {} != d_model {}",
            head.dim(), self.config.d_model,
        );
        self.confidence_head = Some(head);
    }

    pub fn confidence_head(&self) -> Option<&ConfidenceHead> {
        self.confidence_head.as_ref()
    }

    pub fn confidence_head_mut(&mut self) -> Option<&mut ConfidenceHead> {
        self.confidence_head.as_mut()
    }

    pub fn has_confidence_head(&self) -> bool {
        self.confidence_head.is_some()
    }

    /// Number of extra adaptive iterations taken since `reset_extra_steps()`.
    /// Callers drain this into the metrics window per token.
    pub fn take_extra_steps(&mut self) -> u64 {
        std::mem::take(&mut self.extra_steps_counter)
    }

    /// Install a MetaCritic. Its `dim()` must equal `config.d_model`.
    pub fn set_meta_critic(&mut self, meta: MetaCritic) {
        assert_eq!(
            meta.dim(), self.config.d_model,
            "meta critic dim {} != d_model {}", meta.dim(), self.config.d_model,
        );
        self.meta_critic = Some(meta);
    }

    pub fn meta_critic(&self) -> Option<&MetaCritic> { self.meta_critic.as_ref() }
    pub fn has_meta_critic(&self) -> bool { self.meta_critic.is_some() }

    /// Number of tokens where the MetaCritic suppressed an adaptive step.
    /// Drained by the caller into metrics; a non-zero value here with a
    /// stable Head-C MSE is the Phase F "target-network is doing its job"
    /// signal.
    pub fn take_meta_suppressed(&mut self) -> u64 {
        std::mem::take(&mut self.meta_suppressed_counter)
    }

    /// Install a crystallized knowledge module.
    /// Module's d_model must match this model's d_model.
    pub fn push_crystal_module(&mut self, module: CrystalModule) {
        assert_eq!(
            module.d_model, self.config.d_model,
            "crystal module d_model {} != core d_model {}",
            module.d_model, self.config.d_model,
        );
        self.crystal_modules.push(module);
    }

    /// Number of installed crystal modules.
    pub fn n_crystal_modules(&self) -> usize {
        self.crystal_modules.len()
    }

    /// Remove all installed modules (for A/B evaluation).
    pub fn clear_crystal_modules(&mut self) {
        self.crystal_modules.clear();
    }

    /// Apply all matching crystal modules to the current `buf_x` (post-final-norm
    /// hidden state). Additive, routed by cosine similarity to each module's
    /// domain signature. Returns the number of modules that activated.
    fn apply_crystal_modules(&mut self) -> usize {
        if self.crystal_modules.is_empty() { return 0; }
        let threshold = self.crystal_activation_threshold;
        let mut activated = 0;
        for module in &mut self.crystal_modules {
            let similarity = module.domain_match_slice(self.buf_x.data());
            if similarity > threshold {
                self.buf_module_out.zero_();
                module.apply(&self.buf_x, &mut self.buf_module_out, &self.dispatch);
                let scale = similarity.min(1.0);
                for (h, m) in self.buf_x.data_mut().iter_mut()
                    .zip(self.buf_module_out.data().iter())
                {
                    *h += scale * m;
                }
                activated += 1;
            }
        }
        activated
    }

    /// Process a single token. Returns logits.
    ///
    /// Phase B+C: when `adaptive_config.enabled` is true and a confidence
    /// head is installed, Head C is evaluated on the post-block hidden
    /// state. If its z-score against the detector's running baseline
    /// exceeds `z_threshold`, the block stack is iterated up to
    /// `max_extra_steps` additional times on the current buffer before
    /// unembedding. Each extra iteration increments
    /// `extra_steps_counter`, which callers drain via `take_extra_steps()`
    /// into the metrics window.
    pub fn decode_step(&mut self, token: u32) -> Tensor {
        self.embedding.embed(token, &mut self.buf_x);

        for block in self.blocks.iter_mut() {
            let _ = block.forward(&mut self.buf_x, &self.dispatch);
        }

        self.final_norm.forward(&mut self.buf_x, &self.dispatch);
        self.apply_crystal_modules();

        // Adaptive pondering: iterate the block stack extra times when
        // Head C signals uncertainty. Each extra pass refines the SSM
        // state on the *current* hidden without re-embedding. The
        // MetaCritic can suppress adaptive iteration at hidden states
        // where Head C's uncertainty signal is itself unreliable.
        if self.adaptive_config.enabled {
            if let Some(head) = self.confidence_head.as_ref() {
                let predicted_nll = head.predict(self.buf_x.data());
                let novelty = self.confidence_detector.evaluate(predicted_nll);
                if novelty.is_novel && novelty.z >= self.adaptive_config.z_threshold {
                    let meta_suppress = match self.meta_critic.as_ref() {
                        Some(meta) => {
                            meta.predict(self.buf_x.data())
                                >= self.adaptive_config.meta_unreliable_threshold
                        }
                        None => false,
                    };
                    if meta_suppress {
                        self.meta_suppressed_counter += 1;
                    } else {
                        let budget = self.adaptive_config.max_extra_steps as usize;
                        for _ in 0..budget {
                            for block in self.blocks.iter_mut() {
                                let _ = block.forward(&mut self.buf_x, &self.dispatch);
                            }
                            self.final_norm.forward(&mut self.buf_x, &self.dispatch);
                            self.apply_crystal_modules();
                            self.extra_steps_counter += 1;
                        }
                    }
                }
            }
        }

        self.last_hidden.copy_from(&self.buf_x);
        match &self.readout {
            Some(r) => r.unembed(&self.buf_x, &mut self.buf_logits),
            None => self.embedding.unembed(&self.buf_x, &mut self.buf_logits),
        }

        Tensor::from_vec(self.buf_logits.data().to_vec(), &[self.config.vocab_size])
    }

    /// Prefill: process tokens sequentially, return final logits.
    pub fn prefill(&mut self, tokens: &[u32]) -> Tensor {
        let mut logits = Tensor::zeros(&[self.config.vocab_size]);
        for &token in tokens {
            logits = self.decode_step(token);
        }
        logits
    }

    /// Score + detect novelty of current hidden state.
    pub fn energy_score_and_detect(&mut self) -> (f32, bool) {
        self.energy_critic.score_and_detect(&self.last_hidden)
    }

    /// Raw energy score (no detection).
    pub fn energy_score(&self) -> f32 {
        self.energy_critic.score(&self.last_hidden)
    }

    /// Energy baseline.
    pub fn energy_baseline(&self) -> f32 {
        self.energy_critic.baseline()
    }

    /// Last hidden state (for episodic memory).
    pub fn last_hidden(&self) -> &Tensor {
        &self.last_hidden
    }

    /// Training-mode decode that captures per-MoE-block hidden states and
    /// router results. For each MoE block, the tuple is
    /// `(hidden_at_router, RouteResult)`. Non-MoE blocks contribute
    /// `None`. Caller is responsible for the policy-gradient update.
    pub fn decode_step_training(
        &mut self,
        token: u32,
        rng: &mut impl rand::Rng,
        cf_rate: f32,
    ) -> (Tensor, Vec<Option<(Tensor, crate::model::router::RouteResult)>>) {
        self.embedding.embed(token, &mut self.buf_x);
        let mut route_captures = Vec::with_capacity(self.blocks.len());
        for block in self.blocks.iter_mut() {
            let cap = block.forward_training(&mut self.buf_x, &self.dispatch, rng, cf_rate);
            route_captures.push(cap);
        }
        self.final_norm.forward(&mut self.buf_x, &self.dispatch);
        self.apply_crystal_modules();
        self.last_hidden.copy_from(&self.buf_x);
        match &self.readout {
            Some(r) => r.unembed(&self.buf_x, &mut self.buf_logits),
            None => self.embedding.unembed(&self.buf_x, &mut self.buf_logits),
        }
        let logits = Tensor::from_vec(self.buf_logits.data().to_vec(), &[self.config.vocab_size]);
        (logits, route_captures)
    }

    /// Access the router of MoE block `layer_idx`, if any. Returns None when
    /// the layer is Dense.
    pub fn router_mut(&mut self, layer_idx: usize) -> Option<&mut crate::model::router::ExpertRouter> {
        let block = self.blocks.get_mut(layer_idx)?;
        match block.channel_mixer_mut() {
            crate::model::block::ChannelMixer::MoE { router, .. } => Some(router),
            crate::model::block::ChannelMixer::Dense { .. } => None,
        }
    }

    /// Borrow the embedding table layer.
    pub fn embedding(&self) -> &Embedding { &self.embedding }
    /// Borrow the final normalization layer.
    pub fn final_norm(&self) -> &RMSNorm { &self.final_norm }
    /// Borrow the block stack.
    pub fn blocks(&self) -> &[Block] { &self.blocks }

    /// Indices of MoE layers.
    pub fn moe_layer_indices(&self) -> Vec<usize> {
        self.blocks.iter().enumerate()
            .filter_map(|(i, b)| if b.is_moe() { Some(i) } else { None })
            .collect()
    }

    /// Reset all recurrent states.
    pub fn reset_state(&mut self) {
        for block in self.blocks.iter_mut() {
            block.reset_state();
        }
        self.last_hidden.zero_();
    }

    /// Reference to dispatch.
    pub fn dispatch(&self) -> &KernelDispatch {
        &self.dispatch
    }

    /// Tied embed/unembed table, row-major [vocab_size × d_model].
    pub fn embed_table(&self) -> &[f32] {
        self.embedding.table()
    }

    /// Replace the energy critic with a (typically trained) one of the same d_model.
    pub fn replace_energy_critic(&mut self, critic: EnergyCritic) {
        assert_eq!(critic.dim(), self.config.d_model,
            "critic d_model {} != core d_model {}", critic.dim(), self.config.d_model);
        self.energy_critic = critic;
    }

    /// Mutable access to the energy critic (for online training).
    pub fn energy_critic_mut(&mut self) -> &mut EnergyCritic {
        &mut self.energy_critic
    }

    /// Immutable access.
    pub fn energy_critic(&self) -> &EnergyCritic {
        &self.energy_critic
    }
}
