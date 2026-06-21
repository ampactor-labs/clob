//! `.tokens` artifact — a cached array of `u32` token ids.
//!
//! Phase L produces these from a corpus once (via `clob encode`) so that
//! `ingest` and friends can skip BPE re-tokenization on every run. The
//! format is deliberately trivial: a 4-byte magic, a version, a count, then
//! `count` little-endian `u32`s. It is kept separate from the `.clob` model
//! format (`io::format`) on purpose — a corpus cache is not a model and must
//! not share its version line.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Magic for `.tokens` files. Distinct from the `.clob` model magic.
pub const MAGIC: &[u8; 4] = b"CLTK";
/// On-disk format version for `.tokens`.
pub const VERSION: u32 = 1;

/// Write a token array to `path` as `MAGIC | version | count | u32×count`.
pub fn save_tokens(path: &Path, tokens: &[u32]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(tokens.len() as u64).to_le_bytes())?;
    for &t in tokens {
        w.write_all(&t.to_le_bytes())?;
    }
    w.flush()
}

/// Read a `.tokens` file written by [`save_tokens`]. Rejects a bad magic, an
/// unknown version, a truncated body, or trailing bytes — a corrupt cache
/// must fail loudly, never feed garbage ids into the model.
pub fn load_tokens(path: &Path) -> io::Result<Vec<u32>> {
    let mut r = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("not a .tokens file (bad magic {magic:?})"),
        ));
    }

    let mut u32buf = [0u8; 4];
    r.read_exact(&mut u32buf)?;
    let version = u32::from_le_bytes(u32buf);
    if version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported .tokens version {version} (this build writes {VERSION})"),
        ));
    }

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
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing bytes after declared token count",
        ));
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let path = std::env::temp_dir().join("clob_tokens_rt.tokens");
        let toks = vec![0u32, 1, 259, 4000, u32::MAX, 42];
        save_tokens(&path, &toks).unwrap();
        let back = load_tokens(&path).unwrap();
        assert_eq!(toks, back);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_round_trips() {
        let path = std::env::temp_dir().join("clob_tokens_empty.tokens");
        save_tokens(&path, &[]).unwrap();
        assert_eq!(load_tokens(&path).unwrap(), Vec::<u32>::new());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let path = std::env::temp_dir().join("clob_tokens_badmagic.tokens");
        std::fs::write(&path, b"XXXX\x01\x00\x00\x00").unwrap();
        assert!(load_tokens(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn truncated_body_is_rejected() {
        // Declares 4 tokens but only supplies 1.
        let path = std::env::temp_dir().join("clob_tokens_trunc.tokens");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(4u64).to_le_bytes());
        bytes.extend_from_slice(&(7u32).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        assert!(load_tokens(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
