//! sha256 helpers — thin wrappers over the `sha2` crate.
//!
//! Two-line functions, but having them in one place means the manifest
//! and the split-check test (Phase L) agree on exactly how bytes
//! become hashes.

use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Hex-encoded sha256 of a file's contents. Returns `None` on IO failure
/// so callers can treat "unreadable file" and "missing file" uniformly —
/// the manifest emits `null` for either.
pub fn hash_file(path: &Path) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).ok()?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Some(hex_encode(&hasher.finalize()))
}

/// Hex-encoded sha256 of a byte slice.
pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_encode(&hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bytes_is_known_sha256() {
        assert_eq!(
            hash_bytes(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        );
    }

    #[test]
    fn hash_bytes_matches_hash_file() {
        let path = std::env::temp_dir().join("clob_sha_test.bin");
        let payload = b"the ternary kernel writes its own manifest";
        std::fs::write(&path, payload).unwrap();
        let from_bytes = hash_bytes(payload);
        let from_file = hash_file(&path).unwrap();
        assert_eq!(from_bytes, from_file);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_returns_none() {
        let path = std::env::temp_dir().join("clob_definitely_not_here_sha.bin");
        let _ = std::fs::remove_file(&path);
        assert!(hash_file(&path).is_none());
    }
}
