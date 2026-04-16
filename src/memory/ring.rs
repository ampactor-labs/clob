//! SSD-backed ring buffer for episodic memory.
//!
//! Fixed capacity, append-only writes, oldest-first eviction.
//! Thread-safe via mutex.

use crate::memory::episode::Episode;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// SSD-backed episodic memory ring buffer.
pub struct EpisodicMemory {
    /// Directory storing episode files.
    dir: PathBuf,
    /// In-memory index of episodes.
    index: Mutex<RingIndex>,
    /// Maximum number of episodes.
    capacity: usize,
}

struct RingIndex {
    /// Episodes in the ring buffer (newest at the end).
    episodes: Vec<EpisodeRef>,
    /// Write cursor (wraps around capacity).
    write_cursor: usize,
    /// Total episodes ever written.
    total_written: u64,
}

#[derive(Clone)]
struct EpisodeRef {
    /// File path for this episode.
    path: PathBuf,
    /// Whether consumed by crystallization.
    consumed: bool,
    /// Approximate size in bytes.
    size: usize,
}

impl EpisodicMemory {
    /// Create or open an episodic memory in the given directory.
    pub fn open(dir: &Path, capacity: usize) -> std::io::Result<Self> {
        fs::create_dir_all(dir)?;

        // Scan for existing episodes
        let mut episodes = Vec::new();
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "ep") {
                    if let Ok(data) = fs::read(&path) {
                        if let Some(ep) = Episode::from_bytes(&data) {
                            episodes.push(EpisodeRef {
                                path,
                                consumed: ep.consumed,
                                size: data.len(),
                            });
                        }
                    }
                }
            }
        }

        // Sort by filename (which encodes timestamp order)
        episodes.sort_by(|a, b| a.path.cmp(&b.path));

        let write_cursor = episodes.len() % capacity;
        let total_written = episodes.len() as u64;

        Ok(Self {
            dir: dir.to_path_buf(),
            index: Mutex::new(RingIndex { episodes, write_cursor, total_written }),
            capacity,
        })
    }

    /// Store a new episode. Evicts oldest if at capacity.
    pub fn store(&self, episode: &Episode) -> std::io::Result<()> {
        let bytes = episode.to_bytes();
        let mut index = self.index.lock().unwrap();

        let filename = format!("{:016x}.ep", index.total_written);
        let path = self.dir.join(&filename);

        // Evict if at capacity
        if index.episodes.len() >= self.capacity {
            let evict_idx = index.write_cursor % self.capacity;
            if evict_idx < index.episodes.len() {
                let old = &index.episodes[evict_idx];
                let _ = fs::remove_file(&old.path);
                index.episodes[evict_idx] = EpisodeRef {
                    path: path.clone(),
                    consumed: false,
                    size: bytes.len(),
                };
            }
        } else {
            index.episodes.push(EpisodeRef {
                path: path.clone(),
                consumed: false,
                size: bytes.len(),
            });
        }

        fs::write(&path, &bytes)?;
        index.write_cursor = (index.write_cursor + 1) % self.capacity;
        index.total_written += 1;

        Ok(())
    }

    /// Read unconsumed episodes (for crystallization).
    pub fn read_unconsumed(&self, limit: usize) -> Vec<Episode> {
        let index = self.index.lock().unwrap();
        let mut result = Vec::new();

        for ep_ref in index.episodes.iter() {
            if ep_ref.consumed { continue; }
            if let Ok(data) = fs::read(&ep_ref.path) {
                if let Some(ep) = Episode::from_bytes(&data) {
                    if !ep.consumed {
                        result.push(ep);
                        if result.len() >= limit { break; }
                    }
                }
            }
        }
        result
    }

    /// Mark episodes as consumed (by timestamp).
    pub fn mark_consumed(&self, timestamps: &[u64]) {
        let mut index = self.index.lock().unwrap();
        for ep_ref in index.episodes.iter_mut() {
            if let Ok(data) = fs::read(&ep_ref.path) {
                if let Some(mut ep) = Episode::from_bytes(&data) {
                    if timestamps.contains(&ep.timestamp) {
                        ep.consumed = true;
                        ep_ref.consumed = true;
                        let _ = fs::write(&ep_ref.path, ep.to_bytes());
                    }
                }
            }
        }
    }

    /// Stats.
    pub fn stats(&self) -> MemoryStats {
        let index = self.index.lock().unwrap();
        let total = index.episodes.len();
        let consumed = index.episodes.iter().filter(|e| e.consumed).count();
        let total_bytes: usize = index.episodes.iter().map(|e| e.size).sum();
        MemoryStats {
            total_episodes: total,
            unconsumed: total - consumed,
            consumed,
            total_bytes,
            total_ever_written: index.total_written,
            capacity: self.capacity,
        }
    }
}

/// Episodic memory statistics.
pub struct MemoryStats {
    pub total_episodes: usize,
    pub unconsumed: usize,
    pub consumed: usize,
    pub total_bytes: usize,
    pub total_ever_written: u64,
    pub capacity: usize,
}
