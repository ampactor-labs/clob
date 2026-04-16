//! The Kernel — a self-improving ternary recurrent intelligence.
//!
//! Three subsystems:
//!   1. Ternary Recurrent Core — thinks (additions only, O(1) memory)
//!   2. Crystallization Engine — learns (experience → compression → structure)
//!   3. Episodic Memory — remembers (SSD-backed, novelty-filtered)

pub mod tensor;
pub mod simd;
pub mod nn;
pub mod model;
pub mod io;
pub mod memory;
pub mod crystal;
