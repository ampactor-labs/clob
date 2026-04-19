//! The crystallization engine — orchestrates the full loop.
//!
//! Runs as a background operation. Scans episodic memory for mature
//! clusters, distills patterns, crystallizes into ternary modules,
//! and registers them for routing.

use crate::crystal::cluster;
use crate::crystal::crystallize;
use crate::crystal::distill;
use crate::crystal::module::CrystalModule;
use crate::memory::ring::EpisodicMemory;
use crate::simd::KernelDispatch;
use crate::tensor::Tensor;


/// Configuration for the crystallization engine.
pub struct CrystalConfig {
    /// Minimum episodes before attempting crystallization.
    pub min_episodes: usize,
    /// Maximum episodes to process per cycle.
    pub batch_size: usize,
    /// Number of clusters for k-means.
    pub n_clusters: usize,
    /// Maximum k-means iterations.
    pub max_kmeans_iters: usize,
    /// Activation threshold for module routing (cosine similarity).
    pub activation_threshold: f32,
}

impl Default for CrystalConfig {
    fn default() -> Self {
        Self {
            min_episodes: 50,
            batch_size: 500,
            n_clusters: 8,
            max_kmeans_iters: 20,
            activation_threshold: 0.3,
        }
    }
}

/// The crystallization engine.
pub struct CrystallizationEngine {
    /// Crystallized modules.
    modules: Vec<CrystalModule>,
    /// Next module ID.
    next_id: u64,
    /// Configuration.
    config: CrystalConfig,
    /// Dispatch for module application.
    dispatch: KernelDispatch,
    /// Scratch buffer for module output.
    buf_module_out: Tensor,
}

impl CrystallizationEngine {
    pub fn new(d_model: usize, config: CrystalConfig) -> Self {
        Self {
            modules: Vec::new(),
            next_id: 0,
            config,
            dispatch: KernelDispatch::new(),
            buf_module_out: Tensor::zeros(&[d_model]),
        }
    }

    /// Set the starting id for newly crystallized modules. Call after loading
    /// existing modules so IDs don't collide with on-disk files.
    pub fn set_next_id(&mut self, id: u64) {
        self.next_id = id;
    }

    /// Run one crystallization cycle.
    ///
    /// `embed_table` is the model's tied embed/unembed matrix, row-major
    /// `[vocab × d_model]`. Without it, the distiller falls back to a weaker
    /// hidden-centroid correction that ignores `actual_token`.
    ///
    /// Returns the number of new modules crystallized.
    pub fn cycle(
        &mut self,
        memory: &EpisodicMemory,
        embed_table: Option<&[f32]>,
        vocab_size: usize,
    ) -> usize {
        let stats = memory.stats();
        if stats.unconsumed < self.config.min_episodes {
            return 0;
        }

        // Read unconsumed episodes
        let episodes = memory.read_unconsumed(self.config.batch_size);
        if episodes.len() < self.config.min_episodes {
            return 0;
        }

        eprintln!("[crystal] Processing {} unconsumed episodes", episodes.len());

        // Cluster
        let clusters = cluster::cluster_episodes(
            &episodes,
            self.config.n_clusters,
            self.config.max_kmeans_iters,
        );

        eprintln!("[crystal] Found {} clusters", clusters.len());

        let mut new_modules = 0;
        let mut consumed_timestamps = Vec::new();

        for cl in &clusters {
            // Attempt distillation
            if let Some(pattern) = distill::distill(cl, &episodes, embed_table, vocab_size) {
                eprintln!("[crystal] Distilled pattern: {} episodes, avg_error={:.4}, mdl_ratio={:.4}",
                    pattern.n_episodes, pattern.avg_error, pattern.mdl_ratio);

                // Crystallize
                let module = crystallize::crystallize(&pattern, self.next_id);
                self.next_id += 1;

                let sym = match module.symbolic_hint.as_ref() {
                    Some(p) => format!(", symbolic={}ops ({:.1}% of matrix)", p.ops.len(),
                        module.symbolic_compression_ratio() * 100.0),
                    None => String::new(),
                };
                eprintln!("[crystal] Crystallized module #{}: sparsity={:.1}%, weight_bytes={}{}",
                    module.id, module.sparsity * 100.0, module.weight_bytes(), sym);

                consumed_timestamps.extend_from_slice(&pattern.source_timestamps);
                self.modules.push(module);
                new_modules += 1;
            }
        }

        // Mark consumed episodes
        if !consumed_timestamps.is_empty() {
            memory.mark_consumed(&consumed_timestamps);
            eprintln!("[crystal] Marked {} episodes as consumed", consumed_timestamps.len());
        }

        new_modules
    }

    /// Apply all matching modules to a hidden state.
    ///
    /// Modules whose domain signature closely matches the input are activated.
    /// Returns the modified hidden state and the number of modules activated.
    pub fn apply_modules(&mut self, hidden: &mut Tensor) -> usize {
        let mut activated = 0;

        for module in self.modules.iter_mut() {
            let similarity = module.domain_match(hidden);
            if similarity > self.config.activation_threshold {
                // Apply module: hidden += module(hidden)
                self.buf_module_out.zero_();
                module.apply(hidden, &mut self.buf_module_out, &self.dispatch);

                // Scale by similarity (soft routing)
                let scale = similarity.min(1.0);
                for (h, m) in hidden.data_mut().iter_mut().zip(self.buf_module_out.data().iter()) {
                    *h += scale * m;
                }

                activated += 1;
            }
        }

        activated
    }

    /// Number of crystallized modules.
    pub fn n_modules(&self) -> usize { self.modules.len() }

    /// All crystallized modules.
    pub fn modules(&self) -> &[CrystalModule] { &self.modules }

    /// Total activation count across all modules.
    pub fn total_activations(&self) -> u64 {
        self.modules.iter().map(|m| m.activation_count).sum()
    }

    /// Number of modules with a symbolic hint attached.
    pub fn n_symbolic(&self) -> usize {
        self.modules.iter().filter(|m| m.is_symbolic()).count()
    }

    /// Stats string.
    pub fn stats(&self) -> String {
        format!(
            "modules={}, total_activations={}, avg_sparsity={:.1}%, symbolic={}",
            self.modules.len(),
            self.total_activations(),
            if self.modules.is_empty() { 0.0 }
            else { self.modules.iter().map(|m| m.sparsity).sum::<f32>() / self.modules.len() as f32 * 100.0 },
            self.n_symbolic(),
        )
    }
}
