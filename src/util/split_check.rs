//! Train/holdout split enforcement (Phase L).
//!
//! A held-out evaluation set is worthless if its sentences also appear in the
//! training corpus — the model has then memorized, not generalized, and every
//! downstream J/nat number is contaminated. This module hashes every
//! (normalized, non-trivial) sentence of each side and reports any holdout
//! sentence that also occurs in train. `tests/data_split.rs` uses it to gate
//! the real corpus before a run.

use std::collections::HashSet;

/// Sentences shorter than this many words are ignored. Short boilerplate
/// ("He nodded.", "Chapter 3.") legitimately recurs across any split and would
/// only generate noise; contamination that matters is whole reused sentences.
pub const MIN_WORDS: usize = 5;

/// A detected train/holdout contamination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overlap {
    /// Number of distinct holdout sentences that also appear in train.
    pub count: usize,
    /// Up to five example offending sentences (normalized), for diagnostics.
    pub examples: Vec<String>,
}

/// Normalize a sentence to a canonical form (collapse whitespace, lowercase).
/// Returns `None` for sentences below [`MIN_WORDS`].
fn normalize(sentence: &str) -> Option<String> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.len() < MIN_WORDS {
        return None;
    }
    Some(words.join(" ").to_lowercase())
}

/// Split text into sentences on `.`, `!`, `?`, and newlines. Crude but
/// deterministic — the goal is contamination detection, not linguistics.
fn sentences(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| matches!(c, '.' | '!' | '?' | '\n'))
}

/// Build the normalized sentence set for one side of a split.
fn sentence_set(text: &str) -> HashSet<String> {
    sentences(text).filter_map(normalize).collect()
}

/// Return `Some(Overlap)` if any non-trivial holdout sentence also appears in
/// train, else `None`. Order-independent and idempotent.
pub fn find_overlap(train: &str, holdout: &str) -> Option<Overlap> {
    let train_set = sentence_set(train);
    let mut offending: Vec<String> = sentence_set(holdout)
        .into_iter()
        .filter(|s| train_set.contains(s))
        .collect();
    if offending.is_empty() {
        return None;
    }
    offending.sort();
    let count = offending.len();
    offending.truncate(5);
    Some(Overlap { count, examples: offending })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_corpora_have_no_overlap() {
        let train = "The kernel compiles experience into compiled ternary code. \
                     Ternary weights cost less to invoke than the episodes they replace.";
        let holdout = "Crystallization releases the raw episodes once a pattern is captured. \
                       Energy spent per nat of prediction should fall, not rise.";
        assert!(find_overlap(train, holdout).is_none());
    }

    #[test]
    fn injected_overlap_is_detected() {
        let shared = "The crystallization loop turns prediction errors into compiled modules.";
        let train = format!("Some preamble here about nothing in particular. {shared} More text follows.");
        let holdout = format!("An unrelated holdout opening sentence with plenty of words. {shared}");
        let overlap = find_overlap(&train, &holdout).expect("overlap must be detected");
        assert_eq!(overlap.count, 1);
        assert!(overlap.examples[0].contains("crystallization loop turns prediction errors"));
    }

    #[test]
    fn short_boilerplate_is_ignored() {
        // "He nodded." recurring on both sides must NOT count as contamination.
        let train = "He nodded. The long technical explanation spans many distinct words here.";
        let holdout = "He nodded. A wholly different holdout sentence with enough words to count.";
        assert!(find_overlap(train, holdout).is_none());
    }

    #[test]
    fn normalization_ignores_whitespace_and_case() {
        let train = "Compression   efficiency  per JOULE is the figure of merit here.";
        let holdout = "compression efficiency per joule is the figure of merit here.";
        assert!(find_overlap(train, holdout).is_some());
    }
}
