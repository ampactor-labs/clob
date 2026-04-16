//! Zero-copy memory-mapped model loading.

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// A memory-mapped model file.
pub struct MappedModel {
    pub mmap: Mmap,
    pub header_size: usize,
}

impl MappedModel {
    /// Memory-map a .clob file.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse header to find where tensor data starts
        let mut cursor = std::io::Cursor::new(&mmap[..]);
        let (_, header_size) = crate::io::format::read_header(&mut cursor)?;

        Ok(Self { mmap, header_size })
    }

    /// Raw tensor data region (after the header).
    pub fn tensor_data(&self) -> &[u8] {
        &self.mmap[self.header_size..]
    }
}
