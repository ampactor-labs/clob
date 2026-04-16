//! Ternary matrix: packed {-1, 0, 1} weights with per-row scale factors.
//!
//! Encoding: 2 bits per trit, 4 trits per byte.
//!   00 = 0
//!   01 = +1
//!   10 = -1
//!   11 = (unused, treated as 0)
//!
//! Storage is 32-row SIMD-tiled for AVX2 alignment.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Packed ternary weight matrix with per-row scale factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TernaryMatrix {
    /// Packed ternary data: 4 trits per byte, 32-row tiled.
    packed: Vec<u8>,
    /// Per-row scale factor β (absmean from quantization).
    scales: Vec<f32>,
    /// Number of output features (rows).
    rows: usize,
    /// Number of input features (columns).
    cols: usize,
}

/// Map from 2-bit encoding to trit value.
pub const TRIT_DECODE: [i8; 4] = [0, 1, -1, 0];

impl TernaryMatrix {
    /// Bytes needed to pack `n` trits.
    pub fn packed_len(n: usize) -> usize {
        (n + 3) / 4
    }

    /// Pack from unpacked i8 trit values and scale factors.
    ///
    /// `trits` is row-major: `trits[row * cols + col]`.
    /// Each value must be in {-1, 0, 1}.
    pub fn pack(trits: &[i8], scales: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(trits.len(), rows * cols);
        assert_eq!(scales.len(), rows);

        let r_chunks = (rows + 31) / 32;
        let c_chunks = (cols + 3) / 4;
        let mut packed = vec![0u8; r_chunks * c_chunks * 32];

        for r in 0..rows {
            for c in 0..cols {
                let trit = trits[r * cols + c];
                let encoded = match trit {
                    0 => 0u8,
                    1 => 1u8,
                    -1 => 2u8,
                    _ => panic!("invalid trit value: {}", trit),
                };

                let r_chunk = r / 32;
                let r_rem = r % 32;
                let c_chunk = c / 4;
                let c_rem = c % 4;

                let block_idx = r_chunk * c_chunks + c_chunk;
                let byte_idx = block_idx * 32 + r_rem;
                let bit_offset = c_rem * 2;

                packed[byte_idx] |= encoded << bit_offset;
            }
        }

        Self { packed, scales: scales.to_vec(), rows, cols }
    }

    /// Create from already-packed data (loaded from file).
    pub fn from_raw(packed: Vec<u8>, scales: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(scales.len(), rows);
        Self { packed, scales, rows, cols }
    }

    /// Unpack to dense i8 trit values.
    pub fn unpack(&self) -> Vec<i8> {
        let mut trits = vec![0i8; self.rows * self.cols];
        let c_chunks = (self.cols + 3) / 4;

        for r in 0..self.rows {
            for c in 0..self.cols {
                let r_chunk = r / 32;
                let r_rem = r % 32;
                let c_chunk = c / 4;
                let c_rem = c % 4;

                let block_idx = r_chunk * c_chunks + c_chunk;
                let byte_idx = block_idx * 32 + r_rem;
                let bit_offset = c_rem * 2;

                let encoded = (self.packed[byte_idx] >> bit_offset) & 0x03;
                trits[r * self.cols + c] = TRIT_DECODE[encoded as usize];
            }
        }
        trits
    }

    /// Generate a random ternary matrix for testing.
    pub fn random(rows: usize, cols: usize, rng: &mut impl Rng) -> Self {
        let mut trits = vec![0i8; rows * cols];
        let mut scales = vec![0.0f32; rows];

        for r in 0..rows {
            let mut count = 0usize;
            for c in 0..cols {
                let val = rng.gen_range(0u8..3);
                let trit: i8 = match val {
                    0 => 0,
                    1 => 1,
                    2 => -1,
                    _ => unreachable!(),
                };
                trits[r * cols + c] = trit;
                if trit != 0 { count += 1; }
            }
            scales[r] = if count > 0 { count as f32 / cols as f32 } else { 0.0 };
        }

        Self::pack(&trits, &scales, rows, cols)
    }

    pub fn packed_data(&self) -> &[u8] { &self.packed }
    pub fn scales(&self) -> &[f32] { &self.scales }
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    /// Total byte size of the packed representation.
    pub fn byte_size(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let trits: Vec<i8> = vec![1, 0, -1, 1, 0, 0, -1, -1, 1, 0, 1, -1];
        let scales = vec![0.5, 0.5, 0.5];
        let mat = TernaryMatrix::pack(&trits, &scales, 3, 4);
        let unpacked = mat.unpack();
        assert_eq!(trits, unpacked);
    }

    #[test]
    fn random_roundtrip() {
        let mut rng = rand::thread_rng();
        let mat = TernaryMatrix::random(16, 32, &mut rng);
        let unpacked = mat.unpack();
        for &v in &unpacked {
            assert!(v == -1 || v == 0 || v == 1);
        }
        let repacked = TernaryMatrix::pack(&unpacked, mat.scales(), mat.rows(), mat.cols());
        assert_eq!(repacked.unpack(), unpacked);
    }

    #[test]
    fn large_matrix_roundtrip() {
        let mut rng = rand::thread_rng();
        let mat = TernaryMatrix::random(512, 512, &mut rng);
        let unpacked = mat.unpack();
        let repacked = TernaryMatrix::pack(&unpacked, mat.scales(), 512, 512);
        assert_eq!(repacked.unpack(), unpacked);
    }
}
