//! Deterministic seed tree for reproducibility.
//!
//! Every random draw in a `clob` run should be traceable to one top-level
//! `--seed` value. This module provides `SeedTree`: call `.child(label)`
//! to get a `StdRng` derived from the root seed XOR the label's 64-bit
//! hash. Children are deterministic across runs on the same commit given
//! the same root seed — which is the reproducibility guarantee Phase G
//! delivers.
//!
//! The labels are stable strings (`"critic"`, `"router"`, `"synth"`,
//! `"head_c"`, etc.) chosen per-site. Renaming a label changes that
//! subsystem's RNG stream, so labels are a stable part of the
//! reproducibility contract — don't casually rename.

use rand::rngs::StdRng;
use rand::SeedableRng;

/// A deterministic tree of RNGs rooted at a single u64.
#[derive(Debug, Clone, Copy)]
pub struct SeedTree {
    root: u64,
}

impl SeedTree {
    /// Create a new seed tree from a root seed.
    pub fn new(root: u64) -> Self {
        Self { root }
    }

    /// Derive a child RNG from a stable string label. The same (root, label)
    /// pair always yields the same RNG stream.
    pub fn child(&self, label: &str) -> StdRng {
        StdRng::seed_from_u64(self.mix(label))
    }

    /// Derive a child seed (u64) from a label, for places that want the raw
    /// seed rather than a constructed RNG.
    pub fn child_seed(&self, label: &str) -> u64 {
        self.mix(label)
    }

    /// Root seed this tree was constructed with.
    pub fn root(&self) -> u64 {
        self.root
    }

    fn mix(&self, label: &str) -> u64 {
        // SplitMix64-style finalizer over (root XOR fnv1a(label)). Not
        // cryptographic — the goal is deterministic de-correlation, not
        // adversarial resistance. Behaves well with StdRng's entropy
        // requirements.
        let mut h = 0xcbf29ce484222325u64;
        for b in label.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let mut x = self.root ^ h;
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn same_root_same_label_identical_stream() {
        let a = SeedTree::new(42).child("critic");
        let b = SeedTree::new(42).child("critic");
        let mut a = a;
        let mut b = b;
        let va: [u64; 8] = std::array::from_fn(|_| a.gen());
        let vb: [u64; 8] = std::array::from_fn(|_| b.gen());
        assert_eq!(va, vb);
    }

    #[test]
    fn different_labels_decorrelate() {
        let mut a = SeedTree::new(42).child("critic");
        let mut b = SeedTree::new(42).child("router");
        let va: u64 = a.gen();
        let vb: u64 = b.gen();
        assert_ne!(va, vb);
    }

    #[test]
    fn different_roots_decorrelate() {
        let mut a = SeedTree::new(42).child("critic");
        let mut b = SeedTree::new(43).child("critic");
        let va: u64 = a.gen();
        let vb: u64 = b.gen();
        assert_ne!(va, vb);
    }

    #[test]
    fn child_seed_matches_child_rng_first_draw() {
        // child_seed(label) produces the same u64 used internally, so
        // constructing StdRng::seed_from_u64(child_seed(label)) is
        // equivalent to child(label).
        let tree = SeedTree::new(7);
        let mut direct = StdRng::seed_from_u64(tree.child_seed("x"));
        let mut via_child = tree.child("x");
        let a: u64 = direct.gen();
        let b: u64 = via_child.gen();
        assert_eq!(a, b);
    }
}
