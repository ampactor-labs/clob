//! Module exchange logic — knowledge barter between kernels.

use crate::crystal::module::{CrystalModule, ModuleMeta};
use crate::net::protocol::{Message, ModuleOffer};
use crate::tensor::ternary::TernaryMatrix;

/// Decide which modules to request from a peer's offer.
/// Prefers modules from domains the local kernel is weak in.
pub fn select_modules(
    offers: &[ModuleOffer],
    local_domains: &[Vec<f32>],
    max_requests: usize,
) -> Vec<u64> {
    // Score each offer: higher = more novel to us
    let mut scored: Vec<(u64, f32)> = offers.iter().map(|offer| {
        // Check overlap with local modules
        let min_overlap = if local_domains.is_empty() {
            0.0 // We have nothing — everything is novel
        } else {
            // Use domain signature hash as a proxy for similarity
            // In practice, we'd compare full domain signatures
            let overlap = local_domains.iter()
                .map(|_| 0.0f32) // Simplified: no overlap check without full signatures
                .fold(0.0f32, f32::max);
            overlap
        };

        // Novelty score: prefer high activation count (proven useful)
        // and low overlap with existing modules
        let novelty = (1.0 - min_overlap)
            * (offer.activation_count as f32 + 1.0).ln()
            * (1.0 - offer.sparsity * 0.3); // Slight penalty for very sparse modules

        (offer.module_id, novelty)
    }).collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(max_requests);
    scored.into_iter().map(|(id, _)| id).collect()
}

/// Build module offers from local crystal modules.
pub fn build_offers(modules: &[CrystalModule]) -> Vec<ModuleOffer> {
    modules.iter().map(|m| {
        // Hash the domain signature for compact comparison
        let hash = hash_domain(&m.domain_signature);
        ModuleOffer {
            module_id: m.id,
            domain_signature_hash: hash,
            n_source_episodes: m.n_source_episodes,
            sparsity: m.sparsity,
            activation_count: m.activation_count,
        }
    }).collect()
}

/// Simple hash of a domain signature vector.
fn hash_domain(sig: &[f32]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &v in sig {
        let bits = v.to_bits() as u64;
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

/// Pack a crystal module for network transfer.
pub fn pack_module(module: &CrystalModule) -> Message {
    let metadata = bincode::serialize(&module.metadata()).unwrap_or_default();
    Message::Transfer {
        module_id: module.id,
        d_model: module.d_model,
        packed_weights: module.weight.packed_data().to_vec(),
        scales: module.weight.scales().to_vec(),
        domain_signature: module.domain_signature.clone(),
        metadata,
    }
}

/// Unpack a transferred module.
pub fn unpack_module(msg: &Message) -> Option<CrystalModule> {
    if let Message::Transfer {
        module_id, d_model, packed_weights, scales, domain_signature, metadata,
    } = msg {
        let weight = TernaryMatrix::from_raw(
            packed_weights.clone(), scales.clone(), *d_model, *d_model,
        );

        let meta: ModuleMeta = bincode::deserialize(metadata).ok()?;
        let n_nonzero = weight.unpack().iter().filter(|&&t| t != 0).count();
        let sparsity = 1.0 - (n_nonzero as f32 / (d_model * d_model) as f32);

        Some(CrystalModule {
            id: *module_id,
            weight,
            domain_signature: domain_signature.clone(),
            d_model: *d_model,
            n_source_episodes: meta.n_source_episodes,
            avg_error_before: meta.avg_error_before,
            mdl_ratio: meta.mdl_ratio,
            sparsity,
            activation_count: 0, // Reset for local tracking
            symbolic_hint: None, // Network-received modules never carry Phase E hints today
        })
    } else {
        None
    }
}
