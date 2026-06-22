//! Neural network layers — all ternary, all zero-allocation hot path.

pub mod ternary_linear;
pub mod embed;
pub mod rmsnorm;
pub mod ssm;
pub mod mlgru;
pub mod glu;
pub mod energy;
pub mod confidence;
pub mod readout;
