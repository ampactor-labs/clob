//! Perception layer — input sources that feed token streams into the kernel.
//!
//! Sources: stdin (interactive), file, network (future).

pub mod active;

use crate::token::bpe::BpeTokenizer;
use std::io::{self, BufRead, Read};
use std::path::Path;

/// A perception event: tokens + metadata.
pub struct Percept {
    /// Token IDs.
    pub tokens: Vec<u32>,
    /// Source description.
    pub source: String,
    /// Whether this is the end of the input stream.
    pub is_eof: bool,
}

/// Read a line from stdin, tokenize it.
pub fn perceive_stdin(tokenizer: &BpeTokenizer) -> io::Result<Percept> {
    let mut line = String::new();
    let bytes_read = io::stdin().lock().read_line(&mut line)?;
    if bytes_read == 0 {
        return Ok(Percept { tokens: vec![], source: "stdin".into(), is_eof: true });
    }
    let line = line.trim();
    if line.is_empty() {
        return Ok(Percept { tokens: vec![], source: "stdin".into(), is_eof: false });
    }
    let tokens = tokenizer.encode(line);
    Ok(Percept { tokens, source: "stdin".into(), is_eof: false })
}

/// Read an entire file and tokenize it.
pub fn perceive_file(path: &Path, tokenizer: &BpeTokenizer) -> io::Result<Percept> {
    let mut content = String::new();
    std::fs::File::open(path)?.read_to_string(&mut content)?;
    let tokens = tokenizer.encode(&content);
    Ok(Percept {
        tokens,
        source: format!("file:{}", path.display()),
        is_eof: true,
    })
}

/// Stream a file line by line, yielding percepts.
pub struct FileStream {
    lines: Vec<String>,
    pos: usize,
    source: String,
}

impl FileStream {
    pub fn open(path: &Path) -> io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let lines: Vec<String> = content.lines().map(String::from).collect();
        Ok(Self { lines, pos: 0, source: format!("file:{}", path.display()) })
    }

    pub fn next_percept(&mut self, tokenizer: &BpeTokenizer) -> Option<Percept> {
        if self.pos >= self.lines.len() { return None; }
        let line = &self.lines[self.pos];
        self.pos += 1;
        let tokens = tokenizer.encode(line);
        Some(Percept {
            tokens,
            source: self.source.clone(),
            is_eof: self.pos >= self.lines.len(),
        })
    }

    pub fn remaining(&self) -> usize { self.lines.len() - self.pos }
}
