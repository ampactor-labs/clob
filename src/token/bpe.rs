//! Byte-Pair Encoding tokenizer.
//!
//! Supports training from corpus, encoding text → token IDs,
//! and decoding token IDs → text. Serializable vocabulary.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A trained BPE tokenizer.
#[derive(Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    /// Merge rules: (pair_a, pair_b) → merged_token, in priority order.
    merges: Vec<(u32, u32, u32)>,
    /// Token → byte sequence mapping.
    vocab: Vec<Vec<u8>>,
    /// Byte sequence → token ID.
    token_map: HashMap<Vec<u8>, u32>,
    /// Special tokens.
    pub pad_id: u32,
    pub unk_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl BpeTokenizer {
    pub fn new(merges: Vec<(u32, u32, u32)>, vocab: Vec<Vec<u8>>) -> Self {
        let mut token_map = HashMap::new();
        for (id, bytes) in vocab.iter().enumerate() {
            token_map.insert(bytes.clone(), id as u32);
        }
        let n = vocab.len() as u32;
        Self {
            merges, vocab, token_map,
            pad_id: n.saturating_sub(4),
            unk_id: n.saturating_sub(3),
            bos_id: n.saturating_sub(2),
            eos_id: n.saturating_sub(1),
        }
    }

    /// Byte-level tokenizer (256 bytes + 4 special = 260 tokens).
    pub fn byte_level() -> Self {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        vocab.push(b"<pad>".to_vec());
        vocab.push(b"<unk>".to_vec());
        vocab.push(b"<bos>".to_vec());
        vocab.push(b"<eos>".to_vec());
        Self::new(Vec::new(), vocab)
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    /// Encode text → token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.merges.is_empty() {
            return text.bytes().map(|b| b as u32).collect();
        }
        let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        for &(a, b, merged) in &self.merges {
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == a && tokens[i + 1] == b {
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        tokens
    }

    /// Decode token IDs → text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            let idx = id as usize;
            if idx < self.vocab.len() {
                bytes.extend_from_slice(&self.vocab[idx]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn decode_token(&self, id: u32) -> &[u8] {
        let idx = id as usize;
        if idx < self.vocab.len() { &self.vocab[idx] } else { b"<?>" }
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let bytes = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, bytes)
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_roundtrip() {
        let tok = BpeTokenizer::byte_level();
        let text = "hello world! 🌍";
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(text, decoded);
    }

    #[test]
    fn simple_merge() {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        vocab.push(b"he".to_vec()); // 256
        vocab.push(b"hel".to_vec()); // 257
        vocab.push(b"<pad>".to_vec());
        vocab.push(b"<unk>".to_vec());
        vocab.push(b"<bos>".to_vec());
        vocab.push(b"<eos>".to_vec());
        let merges = vec![(104, 101, 256), (256, 108, 257)]; // h+e→he, he+l→hel
        let tok = BpeTokenizer::new(merges, vocab);
        let encoded = tok.encode("hello");
        assert_eq!(encoded, vec![257, 108, 111]); // "hel" "l" "o"
        assert_eq!(tok.decode(&encoded), "hello");
    }
}
