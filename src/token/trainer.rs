//! BPE vocabulary trainer.
//!
//! Given a text corpus, compute merge rules via frequency counting.

use crate::token::bpe::BpeTokenizer;
use std::collections::HashMap;

/// Train a BPE tokenizer from a corpus.
pub fn train_bpe(corpus: &str, n_merges: usize) -> BpeTokenizer {
    // Start with byte-level vocabulary
    let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let mut merges: Vec<(u32, u32, u32)> = Vec::new();

    // Tokenize corpus into byte sequences (one per word, split on whitespace)
    let words: Vec<Vec<u32>> = corpus.split_whitespace()
        .map(|w| w.bytes().map(|b| b as u32).collect())
        .collect();

    // Word frequencies
    let mut word_freqs: HashMap<Vec<u32>, usize> = HashMap::new();
    for word in &words {
        *word_freqs.entry(word.clone()).or_insert(0) += 1;
    }

    for _merge_idx in 0..n_merges {
        // Count pair frequencies
        let mut pair_freqs: HashMap<(u32, u32), usize> = HashMap::new();
        for (word, &freq) in &word_freqs {
            for i in 0..word.len().saturating_sub(1) {
                *pair_freqs.entry((word[i], word[i + 1])).or_insert(0) += freq;
            }
        }

        if pair_freqs.is_empty() { break; }

        // Find most frequent pair
        let &best_pair = pair_freqs.iter()
            .max_by_key(|&(_, &count)| count)
            .map(|(pair, _)| pair)
            .unwrap();

        let (a, b) = best_pair;
        let new_id = vocab.len() as u32;

        // Create merged token bytes
        let mut merged_bytes = vocab[a as usize].clone();
        merged_bytes.extend_from_slice(&vocab[b as usize]);
        vocab.push(merged_bytes);
        merges.push((a, b, new_id));

        // Apply merge to all words
        let mut new_word_freqs: HashMap<Vec<u32>, usize> = HashMap::new();
        for (word, freq) in word_freqs {
            let mut new_word = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == a && word[i + 1] == b {
                    new_word.push(new_id);
                    i += 2;
                } else {
                    new_word.push(word[i]);
                    i += 1;
                }
            }
            *new_word_freqs.entry(new_word).or_insert(0) += freq;
        }
        word_freqs = new_word_freqs;
    }

    // Add special tokens
    vocab.push(b"<pad>".to_vec());
    vocab.push(b"<unk>".to_vec());
    vocab.push(b"<bos>".to_vec());
    vocab.push(b"<eos>".to_vec());

    eprintln!("[bpe] Trained {} merges, vocab size = {}", merges.len(), vocab.len());

    BpeTokenizer::new(merges, vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_simple() {
        let corpus = "the the the cat cat sat the cat sat on the mat";
        let tok = train_bpe(corpus, 10);
        assert!(tok.vocab_size() > 260); // At least some merges happened

        // Roundtrip
        let encoded = tok.encode("the cat");
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, "the cat");
    }

    #[test]
    fn train_compresses() {
        let corpus = "aaaa bbbb aaaa bbbb aaaa bbbb";
        let tok_0 = BpeTokenizer::byte_level();
        let tok_trained = train_bpe(corpus, 5);

        let text = "aaaa bbbb";
        let len_0 = tok_0.encode(text).len();
        let len_t = tok_trained.encode(text).len();
        // Trained tokenizer should produce fewer tokens
        assert!(len_t < len_0, "trained {} >= byte-level {}", len_t, len_0);
    }
}
