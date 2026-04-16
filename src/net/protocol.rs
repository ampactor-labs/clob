//! Wire protocol for kernel-to-kernel communication.
//!
//! Length-prefixed bincode messages over TCP.

use serde::{Deserialize, Serialize};

/// Protocol version.
pub const PROTOCOL_VERSION: u32 = 1;

/// Wire message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    /// Initial handshake.
    Hello {
        version: u32,
        kernel_id: u64,
        config_hash: u64,
        n_modules: usize,
    },
    /// Offer crystallized modules.
    Offer {
        modules: Vec<ModuleOffer>,
    },
    /// Request a specific module.
    Request {
        module_id: u64,
    },
    /// Transfer module data.
    Transfer {
        module_id: u64,
        d_model: usize,
        packed_weights: Vec<u8>,
        scales: Vec<f32>,
        domain_signature: Vec<f32>,
        metadata: Vec<u8>,
    },
    /// Acknowledge receipt.
    Ack {
        module_id: u64,
        useful: bool,
    },
    /// Keep-alive.
    Heartbeat {
        n_modules: usize,
        total_activations: u64,
        uptime_secs: u64,
    },
    /// Graceful disconnect.
    Goodbye,
}

/// Module metadata for offers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleOffer {
    pub module_id: u64,
    pub domain_signature_hash: u64,
    pub n_source_episodes: usize,
    pub sparsity: f32,
    pub activation_count: u64,
}

/// Encode a message to bytes (length-prefixed bincode).
pub fn encode_message(msg: &Message) -> Vec<u8> {
    let payload = bincode::serialize(msg).expect("serialize message");
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(4 + payload.len());
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&payload);
    buf
}

/// Decode a message from a reader.
pub fn decode_message(reader: &mut impl std::io::Read) -> std::io::Result<Message> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;

    if len > 100_000_000 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("message too large: {} bytes", len)));
    }

    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload)?;

    bincode::deserialize(&payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
