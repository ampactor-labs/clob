//! The Kernel — a self-improving ternary recurrent intelligence.
//!
//! Three subsystems:
//!   1. Ternary Recurrent Core — thinks (additions only, O(1) memory)
//!   2. Crystallization Engine — learns (experience → compression → structure)
//!   3. Episodic Memory — remembers (SSD-backed, novelty-filtered)
//!
//! Phase 2 additions:
//!   4. Online Learning — streaming ternary SGD, STE gradients
//!   5. Self-Hosting Compiler — IR → x86-64 → .so → dlopen
//!   6. Mycelium Network — peer discovery, module exchange, spores
//!   7. BPE Tokenizer — byte-pair encoding, perception layer

pub mod tensor;
pub mod simd;
pub mod nn;
pub mod model;
pub mod io;
pub mod memory;
pub mod crystal;
pub mod token;
pub mod perceive;
pub mod learn;
pub mod compile;
pub mod net;
pub mod metrics;
pub mod util;
pub mod config;
pub mod eval;
