//! AVX2 SIMD micro-kernels for ternary accumulation.
//!
//! Uses vpshufb-based ternary lookup with i16→i32 overflow protection.
//! Only compiled on x86_64 targets.

use crate::simd::TernaryKernel;
use crate::tensor::ternary::TernaryMatrix;

/// AVX2 kernel backend.
pub struct Avx2Kernel;

const TRIT_DECODE: [i8; 4] = [0, 1, -1, 0];

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use super::*;
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    unsafe fn build_lut(a: i8, b: i8) -> (__m256i, __m256i) {
        let mut lo = [0i8; 16];
        let mut hi = [0i8; 16];
        for i in 0..16 {
            let trit_a = TRIT_DECODE[i & 0x03];
            let trit_b = TRIT_DECODE[(i >> 2) & 0x03];
            let sum = (a as i32 * trit_a as i32) + (b as i32 * trit_b as i32);
            lo[i] = (sum & 0xFF) as i8;
            hi[i] = ((sum >> 8) & 0xFF) as i8;
        }
        let lut_lo128 = _mm_loadu_si128(lo.as_ptr() as *const __m128i);
        let lut_hi128 = _mm_loadu_si128(hi.as_ptr() as *const __m128i);
        (_mm256_broadcastsi128_si256(lut_lo128), _mm256_broadcastsi128_si256(lut_hi128))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn process_chunk_avx2(
        start_r: usize,
        out_chunk: &mut [f32],
        packed: &[u8],
        quantized: &[i8],
        scales: &[f32],
        cols: usize,
        act_scale: f32,
    ) {
        let r_chunks = (out_chunk.len() + 31) / 32;
        let c_chunks = (cols + 3) / 4;

        for r_c in 0..r_chunks {
            let global_r_chunk = (start_r / 32) + r_c;
            let rows_to_write = std::cmp::min(32, out_chunk.len() - r_c * 32);

            let mut acc_0 = _mm256_setzero_si256();
            let mut acc_1 = _mm256_setzero_si256();
            let mut acc_2 = _mm256_setzero_si256();
            let mut acc_3 = _mm256_setzero_si256();

            let mut acc16_0 = _mm256_setzero_si256();
            let mut acc16_1 = _mm256_setzero_si256();
            let mut overflow_counter = 0;

            for c_c in 0..c_chunks {
                let c = c_c * 4;
                let a0 = *quantized.get(c).unwrap_or(&0);
                let a1 = *quantized.get(c + 1).unwrap_or(&0);
                let a2 = *quantized.get(c + 2).unwrap_or(&0);
                let a3 = *quantized.get(c + 3).unwrap_or(&0);

                let (lut0_lo, lut0_hi) = build_lut(a0, a1);
                let (lut1_lo, lut1_hi) = build_lut(a2, a3);

                let block_idx = global_r_chunk * c_chunks + c_c;
                let w_ptr = packed.as_ptr().add(block_idx * 32);
                let w_vec = _mm256_loadu_si256(w_ptr as *const __m256i);

                let m_0f = _mm256_set1_epi8(0x0F);
                let idx0 = _mm256_and_si256(w_vec, m_0f);
                let idx1 = _mm256_and_si256(_mm256_srli_epi16(w_vec, 4), m_0f);

                let s0_lo = _mm256_shuffle_epi8(lut0_lo, idx0);
                let s0_hi = _mm256_shuffle_epi8(lut0_hi, idx0);
                let s1_lo = _mm256_shuffle_epi8(lut1_lo, idx1);
                let s1_hi = _mm256_shuffle_epi8(lut1_hi, idx1);

                let sum0_16_lo = _mm256_unpacklo_epi8(s0_lo, s0_hi);
                let sum0_16_hi = _mm256_unpackhi_epi8(s0_lo, s0_hi);
                let sum1_16_lo = _mm256_unpacklo_epi8(s1_lo, s1_hi);
                let sum1_16_hi = _mm256_unpackhi_epi8(s1_lo, s1_hi);

                let tot_16_lo = _mm256_add_epi16(sum0_16_lo, sum1_16_lo);
                let tot_16_hi = _mm256_add_epi16(sum0_16_hi, sum1_16_hi);

                acc16_0 = _mm256_add_epi16(acc16_0, tot_16_lo);
                acc16_1 = _mm256_add_epi16(acc16_1, tot_16_hi);

                overflow_counter += 1;
                if overflow_counter == 60 {
                    acc_0 = _mm256_add_epi32(acc_0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc16_0)));
                    acc_1 = _mm256_add_epi32(acc_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc16_0, 1)));
                    acc_2 = _mm256_add_epi32(acc_2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc16_1)));
                    acc_3 = _mm256_add_epi32(acc_3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc16_1, 1)));
                    acc16_0 = _mm256_setzero_si256();
                    acc16_1 = _mm256_setzero_si256();
                    overflow_counter = 0;
                }
            }

            // Flush remaining i16 accumulators
            acc_0 = _mm256_add_epi32(acc_0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc16_0)));
            acc_1 = _mm256_add_epi32(acc_1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc16_0, 1)));
            acc_2 = _mm256_add_epi32(acc_2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc16_1)));
            acc_3 = _mm256_add_epi32(acc_3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc16_1, 1)));

            let mut out32 = [0i32; 32];
            _mm256_storeu_si256(out32[0..8].as_mut_ptr() as *mut __m256i, acc_0);
            _mm256_storeu_si256(out32[8..16].as_mut_ptr() as *mut __m256i, acc_2);
            _mm256_storeu_si256(out32[16..24].as_mut_ptr() as *mut __m256i, acc_1);
            _mm256_storeu_si256(out32[24..32].as_mut_ptr() as *mut __m256i, acc_3);

            for i in 0..rows_to_write {
                let r = r_c * 32 + i;
                out_chunk[r] = out32[i] as f32 * act_scale * scales[start_r + r];
            }
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn ternary_accumulate_avx2(
        activations: &[f32],
        weights: &TernaryMatrix,
        output: &mut [f32],
    ) {
        let rows = weights.rows();
        let cols = weights.cols();

        let abs_max = activations.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let act_scale = if abs_max > 0.0 { abs_max / 127.0 } else { 1.0 };
        let inv_act_scale = if abs_max > 0.0 { 127.0 / abs_max } else { 0.0 };

        let mut quantized = vec![0i8; cols];
        for (i, &v) in activations.iter().enumerate() {
            quantized[i] = (v * inv_act_scale).round().clamp(-127.0, 127.0) as i8;
        }

        let packed = weights.packed_data();
        let scales = weights.scales();

        // Process in 32-row chunks
        let rows_per_chunk = ((rows + rayon::current_num_threads() - 1)
            / rayon::current_num_threads() + 31) & !31;

        use rayon::prelude::*;
        output[..rows].par_chunks_mut(rows_per_chunk).enumerate().for_each(
            |(chunk_idx, out_chunk)| {
                let start_r = chunk_idx * rows_per_chunk;
                unsafe {
                    process_chunk_avx2(
                        start_r, out_chunk, packed, &quantized, scales,
                        cols, act_scale,
                    );
                }
            },
        );
    }
}

impl TernaryKernel for Avx2Kernel {
    fn ternary_accumulate(activations: &[f32], weights: &TernaryMatrix, output: &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { avx2_impl::ternary_accumulate_avx2(activations, weights, output); }
                return;
            }
        }
        crate::simd::scalar::ScalarKernel::ternary_accumulate(activations, weights, output);
    }

    fn sigmoid_inplace(x: &mut [f32]) {
        crate::simd::scalar::ScalarKernel::sigmoid_inplace(x);
    }

    fn silu_inplace(x: &mut [f32]) {
        crate::simd::scalar::ScalarKernel::silu_inplace(x);
    }

    fn hadamard(a: &[f32], b: &[f32], out: &mut [f32]) {
        crate::simd::scalar::ScalarKernel::hadamard(a, b, out);
    }

    fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
        crate::simd::scalar::ScalarKernel::rmsnorm(x, weight, eps);
    }

    fn softplus_inplace(x: &mut [f32]) {
        crate::simd::scalar::ScalarKernel::softplus_inplace(x);
    }
}
