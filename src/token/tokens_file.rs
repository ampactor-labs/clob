//! `.tokens` artifact — a cached array of `u32` token ids plus the vocab it
//! was encoded against.
//!
//! Phase L produces these from a corpus once (via `clob encode`) so that
//! `ingest` and friends can skip BPE re-tokenization on every run. The format
//! is deliberately trivial: a 4-byte magic, a version, the encoder's vocab
//! size, a count, then `count` little-endian `u32`s. It is kept separate from
//! the `.clob` model format (`io::format`) on purpose — a corpus cache is not
//! a model and must not share its version line.
//!
//! The `vocab_size` stamp lets a consumer reject a cache whose ids can't index
//! the model it's about to feed: a `.tokens` file encoded with a 4096-vocab
//! BPE must not be streamed into a 256-vocab model, even if the caller forgot
//! to pass the matching `--tokenizer`.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic for `.tokens` files. Distinct from the `.clob` model magic.
pub const MAGIC: &[u8; 4] = b"CLTK";
/// On-disk format version for `.tokens`. v2 added the `vocab_size` stamp.
pub const VERSION: u32 = 2;

/// A decoded `.tokens` file: the ids and the vocabulary they were encoded
/// against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenFile {
    /// Vocab size of the tokenizer that produced `tokens`. Every id is `<`
    /// this value.
    pub vocab_size: u32,
    /// The encoded token stream.
    pub tokens: Vec<u32>,
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Write a token array to `path` as
/// `MAGIC | version | vocab_size | count | u32×count`.
pub fn save_tokens(path: &Path, tokens: &[u32], vocab_size: u32) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&vocab_size.to_le_bytes())?;
    w.write_all(&(tokens.len() as u64).to_le_bytes())?;
    for &t in tokens {
        w.write_all(&t.to_le_bytes())?;
    }
    w.flush()
}

/// Read a `.tokens` file written by [`save_tokens`]. Rejects a bad magic, an
/// unknown version (including the pre-stamp v1 — regenerate it), a truncated
/// body, or trailing bytes — a corrupt cache must fail loudly, never feed
/// garbage ids into the model.
pub fn load_tokens(path: &Path) -> io::Result<TokenFile> {
    let mut r = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid(format!("not a .tokens file (bad magic {magic:?})")));
    }

    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    let version = u32::from_le_bytes(u32buf);
    if version != VERSION {
        return Err(invalid(format!(
            "unsupported .tokens version {version} (this build writes {VERSION}); \
             regenerate via scripts/tokenize_corpus.sh"
        )));
    }

    r.read_exact(&mut u32buf)?;
    let vocab_size = u32::from_le_bytes(u32buf);

    let mut u64buf = [0u8; 8];
    r.read_exact(&mut u64buf)?;
    let count = u64::from_le_bytes(u64buf) as usize;

    let mut tokens = Vec::with_capacity(count);
    for _ in 0..count {
        r.read_exact(&mut u32buf)?; // UnexpectedEof here ⇒ truncated cache.
        tokens.push(u32::from_le_bytes(u32buf));
    }

    // Reject trailing bytes: the declared count must be exact.
    let mut extra = [0u8; 1];
    if r.read(&mut extra)? != 0 {
        return Err(invalid("trailing bytes after declared token count"));
    }
    Ok(TokenFile { vocab_size, tokens })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_carries_vocab() {
        let path = std::env::temp_dir().join("clob_tokens_rt.tokens");
        let toks = vec![0u32, 1, 259, 4000, 4095, 42];
        save_tokens(&path, &toks, 4096).unwrap();
        let tf = load_tokens(&path).unwrap();
        assert_eq!(tf.tokens, toks);
        assert_eq!(tf.vocab_size, 4096);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_round_trips() {
        let path = std::env::temp_dir().join("clob_tokens_empty.tokens");
        save_tokens(&path, &[], 256).unwrap();
        let tf = load_tokens(&path).unwrap();
        assert!(tf.tokens.is_empty());
        assert_eq!(tf.vocab_size, 256);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let path = std::env::temp_dir().join("clob_tokens_badmagic.tokens");
        std::fs::write(&path, b"XXXX\x02\x00\x00\x00").unwrap();
        assert!(load_tokens(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn old_version_is_rejected() {
        // A v1-style header (no vocab field) must fail loudly, not be
        // misparsed with the count read out of the old token position.
        let path = std::env::temp_dir().join("clob_tokens_v1.tokens");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_tokens(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncated_body_is_rejected() {
        // Declares 4 tokens but supplies 1.
        let path = std::env::temp_dir().join("clob_tokens_trunc.tokens");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&256u32.to_le_bytes());
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_tokens(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
