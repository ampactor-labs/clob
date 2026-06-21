//! Shared utilities: seed derivation, and (in later phases) hashing,
//! logging, watchdog, split-check. Kept deliberately thin — nothing
//! in this module depends on any subsystem beyond std + rand.

pub mod seed;
pub mod sha;
pub mod split_check;
