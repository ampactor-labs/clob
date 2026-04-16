//! x86-64 machine code emitter for ternary module forward passes.
//!
//! Emits a function: extern "C" fn(input: *const f32, output: *mut f32, len: usize)
//! that computes output = W ⊛ input using only addss/subss (no mulps for trits).

use crate::compile::ir::{ModuleIR, Op, Trit};

/// Emitted machine code.
pub struct MachineCode {
    pub code: Vec<u8>,
    pub text_size: usize,
}

/// Emit x86-64 machine code from IR.
pub fn emit_x86(ir: &ModuleIR) -> MachineCode {
    let mut code = Vec::with_capacity(ir.ops.len() * 8 + 64);

    // Function prologue:
    // push rbp; mov rbp, rsp
    // rdi = input ptr, rsi = output ptr, rdx = len
    code.extend_from_slice(&[0x55]);                             // push rbp
    code.extend_from_slice(&[0x48, 0x89, 0xe5]);                 // mov rbp, rsp

    // Zero the output buffer first: memset(output, 0, n_outputs * 4)
    // We'll use a simple rep stosb approach
    // push rdi; mov rdi, rsi; xor eax, eax; mov ecx, n_outputs*4; rep stosb; pop rdi
    let out_bytes = (ir.n_outputs * 4) as u32;
    code.extend_from_slice(&[0x57]);                             // push rdi (save input ptr)
    code.extend_from_slice(&[0x48, 0x89, 0xf7]);                 // mov rdi, rsi (output ptr)
    code.extend_from_slice(&[0x31, 0xc0]);                       // xor eax, eax
    code.extend_from_slice(&[0xb9]);                             // mov ecx, imm32
    code.extend_from_slice(&out_bytes.to_le_bytes());
    code.extend_from_slice(&[0xf3, 0xaa]);                       // rep stosb
    code.extend_from_slice(&[0x5f]);                             // pop rdi (restore input ptr)

    // Process ops
    // rdi = input base, rsi = output base
    // We use xmm0 as accumulator, xmm1 as temp
    for op in &ir.ops {
        match op {
            Op::Accumulate { row: _, col, trit } => {
                let col_offset = (*col * 4) as i32;
                // movss xmm1, [rdi + col*4]  (load input[col])
                code.extend_from_slice(&[0xf3, 0x0f, 0x10, 0x8f]);
                code.extend_from_slice(&col_offset.to_le_bytes());

                match trit {
                    Trit::Plus => {
                        // addss xmm0, xmm1
                        code.extend_from_slice(&[0xf3, 0x0f, 0x58, 0xc1]);
                    }
                    Trit::Minus => {
                        // subss xmm0, xmm1
                        code.extend_from_slice(&[0xf3, 0x0f, 0x5c, 0xc1]);
                    }
                    Trit::Zero => {} // unreachable in lowered IR
                }
            }
            Op::Scale { row: _, factor } => {
                // Load scale factor into xmm1 via integer register
                let bits = factor.to_bits();
                // mov eax, imm32; movd xmm1, eax; mulss xmm0, xmm1
                code.extend_from_slice(&[0xb8]);
                code.extend_from_slice(&bits.to_le_bytes());
                code.extend_from_slice(&[0x66, 0x0f, 0x6e, 0xc8]); // movd xmm1, eax
                code.extend_from_slice(&[0xf3, 0x0f, 0x59, 0xc1]); // mulss xmm0, xmm1
            }
            Op::Store { row } => {
                let row_offset = (*row * 4) as i32;
                // movss [rsi + row*4], xmm0
                code.extend_from_slice(&[0xf3, 0x0f, 0x11, 0x86]);
                code.extend_from_slice(&row_offset.to_le_bytes());
                // xorps xmm0, xmm0 (reset accumulator for next row)
                code.extend_from_slice(&[0x0f, 0x57, 0xc0]);
            }
        }
    }

    // Function epilogue: pop rbp; ret
    code.extend_from_slice(&[0x5d]);                             // pop rbp
    code.extend_from_slice(&[0xc3]);                             // ret

    let text_size = code.len();
    MachineCode { code, text_size }
}
