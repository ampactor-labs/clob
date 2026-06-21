//! Peer management — connection pool with trust scoring.

use crate::net::protocol::{self, Message};
use std::collections::HashMap;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};

use std::time::{Duration, Instant};

/// A connected peer.
pub struct Peer {
    pub addr: SocketAddr,
    pub kernel_id: u64,
    pub stream: TcpStream,
    /// EMA trust score (0.0 = untrusted, 1.0 = fully trusted).
    pub trust: f32,
    /// Modules received from this peer.
    pub modules_received: usize,
    /// Modules that turned out to be useful.
    pub modules_useful: usize,
    /// Last heartbeat time.
    pub last_seen: Instant,
}

impl Peer {
    pub fn connect(
        addr: SocketAddr,
        my_kernel_id: u64,
        my_config_hash: u64,
        my_n_modules: usize,
    ) -> std::io::Result<Self> {
        let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(5))?;
        stream.set_nodelay(true)?;

        let mut peer = Self {
            addr,
            kernel_id: 0,
            stream,
            trust: 0.5, // neutral
            modules_received: 0,
            modules_useful: 0,
            last_seen: Instant::now(),
        };

        // Send hello, advertising our architecture fingerprint so the peer can
        // tell whether our modules are shape-compatible with theirs.
        let hello = Message::Hello {
            version: protocol::PROTOCOL_VERSION,
            kernel_id: my_kernel_id,
            config_hash: my_config_hash,
            n_modules: my_n_modules,
        };
        peer.send(&hello)?;

        // Receive hello back. A differing config_hash means modules from this
        // peer were grown by a differently-shaped kernel — surface it loudly
        // rather than silently accepting incompatible weights later.
        if let Ok(Message::Hello { kernel_id, config_hash, .. }) = peer.recv() {
            peer.kernel_id = kernel_id;
            if config_hash != my_config_hash {
                eprintln!(
                    "[net] WARN: peer #{kernel_id} config_hash {config_hash:#018x} differs \
                     from ours {my_config_hash:#018x}; modules may be incompatible",
                );
            }
        }

        Ok(peer)
    }

    pub fn send(&mut self, msg: &Message) -> std::io::Result<()> {
        let bytes = protocol::encode_message(msg);
        self.stream.write_all(&bytes)?;
        self.stream.flush()
    }

    pub fn recv(&mut self) -> std::io::Result<Message> {
        protocol::decode_message(&mut self.stream)
    }

    /// Update trust score based on whether a received module was useful.
    pub fn update_trust(&mut self, useful: bool) {
        let value = if useful { 1.0 } else { 0.0 };
        self.trust = 0.9 * self.trust + 0.1 * value;
        self.modules_received += 1;
        if useful { self.modules_useful += 1; }
    }

    pub fn is_alive(&self) -> bool {
        self.last_seen.elapsed() < Duration::from_secs(60)
    }
}

/// Peer registry.
pub struct PeerRegistry {
    pub peers: HashMap<u64, Peer>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self { peers: HashMap::new() }
    }

    pub fn add(&mut self, peer: Peer) {
        self.peers.insert(peer.kernel_id, peer);
    }

    pub fn remove(&mut self, kernel_id: u64) {
        self.peers.remove(&kernel_id);
    }

    pub fn active_count(&self) -> usize {
        self.peers.values().filter(|p| p.is_alive()).count()
    }

    /// Get peers sorted by trust (highest first).
    pub fn by_trust(&self) -> Vec<u64> {
        let mut ids: Vec<(u64, f32)> = self.peers.iter()
            .filter(|(_, p)| p.is_alive())
            .map(|(&id, p)| (id, p.trust))
            .collect();
        ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ids.into_iter().map(|(id, _)| id).collect()
    }
}

impl Default for PeerRegistry {
    fn default() -> Self { Self::new() }
}
