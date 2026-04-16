//! Self-hosting compiler — crystallized modules → native x86-64 code.
//!
//! IR → x86-64 machine code → ELF .so → dlopen hot-swap.

pub mod ir;
pub mod lower;
pub mod x86;
pub mod elf;
pub mod jit;
