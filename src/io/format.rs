//! Binary format spec for .clob model files.
//!
//! Layout:
//!   [4 bytes]  magic "CLOB"
//!   [4 bytes]  version (u32 LE = 1)
//!   [4 bytes]  config_size (u32 LE)
//!   [N bytes]  bincode-serialized KernelConfig
//!   [4 bytes]  entries_count (u32 LE)
//!   [M bytes]  bincode-serialized Vec<WeightEntry>
//!   [rest]     contiguous tensor data

use crate::model::config::KernelConfig;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub const MAGIC: &[u8; 4] = b"CLOB";
pub const VERSION: u32 = 1;

/// Describes one weight tensor in the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEntry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub dtype: WeightDtype,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightDtype {
    F32,
    TernaryPacked,
}

/// File header.
#[derive(Debug)]
pub struct FileHeader {
    pub config: KernelConfig,
    pub weight_entries: Vec<WeightEntry>,
}

/// Write a .clob file header.
pub fn write_header<W: Write>(writer: &mut W, config: &KernelConfig, entries: &[WeightEntry]) -> std::io::Result<usize> {
    let mut total = 0;

    // Magic + version
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    total += 8;

    // Config
    let config_bytes = bincode::serialize(config).expect("serialize config");
    writer.write_all(&(config_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&config_bytes)?;
    total += 4 + config_bytes.len();

    // Weight entries
    let entries_bytes = bincode::serialize(entries).expect("serialize entries");
    writer.write_all(&(entries_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&entries_bytes)?;
    total += 4 + entries_bytes.len();

    Ok(total)
}

/// Read a .clob file header. Returns (header, bytes_consumed).
pub fn read_header<R: Read>(reader: &mut R) -> std::io::Result<(FileHeader, usize)> {
    let mut total = 0;

    // Magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    total += 4;

    // Version
    let mut ver = [0u8; 4];
    reader.read_exact(&mut ver)?;
    let version = u32::from_le_bytes(ver);
    if version != VERSION {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("unsupported version {}", version)));
    }
    total += 4;

    // Config
    let mut size_buf = [0u8; 4];
    reader.read_exact(&mut size_buf)?;
    let config_size = u32::from_le_bytes(size_buf) as usize;
    let mut config_bytes = vec![0u8; config_size];
    reader.read_exact(&mut config_bytes)?;
    let config: KernelConfig = bincode::deserialize(&config_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    total += 4 + config_size;

    // Weight entries
    reader.read_exact(&mut size_buf)?;
    let entries_size = u32::from_le_bytes(size_buf) as usize;
    let mut entries_bytes = vec![0u8; entries_size];
    reader.read_exact(&mut entries_bytes)?;
    let weight_entries: Vec<WeightEntry> = bincode::deserialize(&entries_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    total += 4 + entries_size;

    Ok((FileHeader { config, weight_entries }, total))
}
