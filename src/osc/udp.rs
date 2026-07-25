//! UDP OSC sender targeting VRChat (default `127.0.0.1:9000`), or any host on
//! the LAN (issue #3).

use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};

use anyhow::Context;

use super::encode::encode_param;
use super::{OscParam, OscSink};

/// Sends OSC parameter updates to VRChat over UDP.
///
/// One datagram is sent per parameter (see [`crate::osc`] design notes). The
/// socket is bound to an ephemeral local port (`0.0.0.0:0`); the resolved
/// target address is fixed at construction.
#[derive(Debug)]
pub struct UdpOscSender {
    socket: UdpSocket,
    target: SocketAddr,
}

impl UdpOscSender {
    /// Bind a local UDP socket and resolve `host:port` as the OSC target.
    ///
    /// VRChat's default is `127.0.0.1:9000`. Returns `Err` if `host` cannot be
    /// resolved or the local socket cannot be bound.
    pub fn new(host: &str, port: u16) -> anyhow::Result<Self> {
        let target = (host, port)
            .to_socket_addrs()
            .with_context(|| format!("invalid OSC target host:port {host}:{port}"))?
            .next()
            .with_context(|| format!("no address resolved for {host}:{port}"))?;
        let socket = UdpSocket::bind("0.0.0.0:0").context("failed to bind local UDP socket")?;
        Ok(Self { socket, target })
    }

    /// The resolved target address OSC datagrams are sent to.
    pub fn target(&self) -> SocketAddr {
        self.target
    }

    /// The local address the sending socket is bound to.
    pub fn local_addr(&self) -> anyhow::Result<SocketAddr> {
        self.socket.local_addr().context("no local address")
    }
}

impl OscSink for UdpOscSender {
    fn send(&mut self, params: &[OscParam]) -> anyhow::Result<()> {
        for param in params {
            let bytes = encode_param(param)?;
            self.socket
                .send_to(&bytes, self.target)
                .with_context(|| format!("failed to send OSC to {}", self.target))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_resolves_loopback_target() {
        let sender = UdpOscSender::new("127.0.0.1", 9000).unwrap();
        assert_eq!(sender.target(), "127.0.0.1:9000".parse().unwrap());
        // Bound to an ephemeral local port, so a real address exists.
        assert!(sender.local_addr().is_ok());
    }

    #[test]
    fn new_rejects_invalid_host() {
        let err = UdpOscSender::new("!!! not a host !!!", 9000);
        assert!(err.is_err(), "invalid host must return Err");
    }
}
