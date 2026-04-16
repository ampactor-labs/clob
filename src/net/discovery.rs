//! Peer discovery — mDNS for local, manual for direct.

use std::net::SocketAddr;

/// Discovery result.
pub struct DiscoveredPeer {
    pub addr: SocketAddr,
    pub source: DiscoverySource,
}

#[derive(Debug, Clone)]
pub enum DiscoverySource {
    Manual,
    MDNS,
}

/// Parse peer addresses from CLI arguments.
pub fn parse_peers(addrs: &[String]) -> Vec<DiscoveredPeer> {
    addrs.iter().filter_map(|s| {
        s.parse::<SocketAddr>().ok().map(|addr| DiscoveredPeer {
            addr,
            source: DiscoverySource::Manual,
        })
    }).collect()
}

/// mDNS discovery (placeholder — will use raw UDP multicast).
pub fn discover_mdns(service_name: &str, timeout_ms: u64) -> Vec<DiscoveredPeer> {
    let multicast_addr: SocketAddr = "224.0.0.251:5353".parse().unwrap();

    // Construct mDNS query for _clob._tcp.local
    let query = build_mdns_query(service_name);

    // Send query on all interfaces
    if let Ok(socket) = std::net::UdpSocket::bind("0.0.0.0:0") {
        let _ = socket.set_read_timeout(Some(std::time::Duration::from_millis(timeout_ms)));
        let _ = socket.send_to(&query, multicast_addr);

        // Collect responses
        let mut peers = Vec::new();
        let mut buf = [0u8; 4096];
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);

        while std::time::Instant::now() < deadline {
            if let Ok((len, src)) = socket.recv_from(&mut buf) {
                // Parse mDNS response (simplified — look for SRV records)
                if let Some(port) = parse_mdns_srv(&buf[..len]) {
                    let peer_addr = SocketAddr::new(src.ip(), port);
                    peers.push(DiscoveredPeer {
                        addr: peer_addr,
                        source: DiscoverySource::MDNS,
                    });
                }
            }
        }
        peers
    } else {
        Vec::new()
    }
}

/// Register this kernel on mDNS.
pub fn register_mdns(service_name: &str, port: u16) -> std::io::Result<()> {
    let response = build_mdns_response(service_name, port);
    let socket = std::net::UdpSocket::bind("0.0.0.0:5353")?;
    socket.set_multicast_loop_v4(true)?;
    socket.join_multicast_v4(&"224.0.0.251".parse().unwrap(), &"0.0.0.0".parse().unwrap())?;
    // Announce presence
    let multicast_addr: SocketAddr = "224.0.0.251:5353".parse().unwrap();
    socket.send_to(&response, multicast_addr)?;
    Ok(())
}

fn build_mdns_query(service: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    // DNS header: ID=0, QR=0 (query), QDCOUNT=1
    buf.extend_from_slice(&[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
    // QNAME: encode service name
    for part in service.split('.') {
        buf.push(part.len() as u8);
        buf.extend_from_slice(part.as_bytes());
    }
    buf.push(0); // root
    buf.extend_from_slice(&[0, 12, 0, 1]); // QTYPE=PTR, QCLASS=IN
    buf
}

fn build_mdns_response(service: &str, port: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(128);
    // DNS header: ID=0, QR=1 (response), ANCOUNT=1
    buf.extend_from_slice(&[0, 0, 0x84, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
    // Answer: SRV record
    for part in service.split('.') {
        buf.push(part.len() as u8);
        buf.extend_from_slice(part.as_bytes());
    }
    buf.push(0);
    buf.extend_from_slice(&[0, 33, 0, 1]); // TYPE=SRV, CLASS=IN
    buf.extend_from_slice(&[0, 0, 0, 120]); // TTL=120
    let rdlen = 6u16; // priority(2) + weight(2) + port(2)
    buf.extend_from_slice(&rdlen.to_be_bytes());
    buf.extend_from_slice(&[0, 0]); // priority
    buf.extend_from_slice(&[0, 0]); // weight
    buf.extend_from_slice(&port.to_be_bytes());
    buf
}

fn parse_mdns_srv(data: &[u8]) -> Option<u16> {
    // Simplified: look for SRV record type (33) and extract port
    if data.len() < 12 { return None; }
    // Skip header, scan for type=33
    let mut i = 12;
    while i + 10 < data.len() {
        // Skip name
        while i < data.len() && data[i] != 0 {
            if data[i] & 0xc0 == 0xc0 { i += 2; break; }
            i += data[i] as usize + 1;
        }
        if i < data.len() && data[i] == 0 { i += 1; }

        if i + 10 > data.len() { break; }
        let rtype = u16::from_be_bytes([data[i], data[i + 1]]);
        i += 8; // type(2) + class(2) + ttl(4)
        if i + 2 > data.len() { break; }
        let rdlen = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        i += 2;

        if rtype == 33 && rdlen >= 6 && i + 6 <= data.len() {
            let port = u16::from_be_bytes([data[i + 4], data[i + 5]]);
            return Some(port);
        }
        i += rdlen;
    }
    None
}
