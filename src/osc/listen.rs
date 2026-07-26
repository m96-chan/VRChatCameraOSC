//! Non-blocking OSC receiver for VRChat's client→tracker messages —
//! specifically `/avatar/change` (issue #18 phase 3). VRChat broadcasts its
//! OSC output (including the avatar id on every avatar switch) to UDP port
//! 9001 by default (VRChat OSC docs).
//!
//! # Design (CLAUDE.md rule 8, recorded)
//!
//! The socket is **polled once per pipeline step** rather than serviced on a
//! dedicated thread: one nonblocking `recv_from` syscall per frame (~30/s) is
//! negligible next to inference, needs no cross-thread state or locking, and
//! an avatar switch is a human-scale event where one frame of extra latency
//! is irrelevant. VRChat also floods 9001 with parameter feedback; every
//! pending datagram is drained per poll so the receive buffer cannot back up
//! between steps.

use std::io::ErrorKind;
use std::net::UdpSocket;

/// Nonblocking UDP OSC receiver that watches for `/avatar/change`.
#[derive(Debug)]
pub struct AvatarChangeListener {
    socket: UdpSocket,
    buf: Vec<u8>,
}

impl AvatarChangeListener {
    /// Bind on `0.0.0.0:port` and switch to nonblocking mode. Passing port 0
    /// binds an OS-assigned ephemeral port (used by tests — see
    /// [`local_port`](Self::local_port)); the config layer treats a
    /// configured `listen_port = 0` as *disabled* and never constructs a
    /// listener for it.
    pub fn bind(port: u16) -> anyhow::Result<Self> {
        let socket = UdpSocket::bind(("0.0.0.0", port))
            .map_err(|e| anyhow::anyhow!("binding OSC listen port {port}: {e}"))?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            // Max practical UDP payload; VRChat messages are far smaller.
            buf: vec![0; 65536],
        })
    }

    /// The actually-bound local port (resolves port 0 to the assigned one).
    pub fn local_port(&self) -> u16 {
        self.socket.local_addr().map(|a| a.port()).unwrap_or(0)
    }

    /// Drain every pending datagram and return the avatar id from the most
    /// recent `/avatar/change` message, if any arrived since the last poll.
    /// Never blocks; malformed packets and other addresses are ignored.
    pub fn poll_avatar_change(&mut self) -> Option<String> {
        let mut latest = None;
        loop {
            match self.socket.recv_from(&mut self.buf) {
                Ok((len, _)) => {
                    if let Ok((_, packet)) = rosc::decoder::decode_udp(&self.buf[..len]) {
                        extract_avatar_change(&packet, &mut latest);
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                // Transient errors (e.g. ICMP-induced ECONNRESET on Windows)
                // just end this poll; the next step polls again.
                Err(_) => break,
            }
        }
        latest
    }
}

/// Pull the avatar id out of a packet (recursing into bundles).
fn extract_avatar_change(packet: &rosc::OscPacket, latest: &mut Option<String>) {
    match packet {
        rosc::OscPacket::Message(msg) => {
            if msg.addr == "/avatar/change" {
                if let Some(rosc::OscType::String(id)) = msg.args.first() {
                    *latest = Some(id.clone());
                }
            }
        }
        rosc::OscPacket::Bundle(bundle) => {
            for p in &bundle.content {
                extract_avatar_change(p, latest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn send(port: u16, packet: &rosc::OscPacket) {
        let client = UdpSocket::bind("127.0.0.1:0").unwrap();
        let bytes = rosc::encoder::encode(packet).unwrap();
        client.send_to(&bytes, ("127.0.0.1", port)).unwrap();
    }

    fn change_msg(id: &str) -> rosc::OscPacket {
        rosc::OscPacket::Message(rosc::OscMessage {
            addr: "/avatar/change".to_string(),
            args: vec![rosc::OscType::String(id.to_string())],
        })
    }

    fn recv_with_retry(l: &mut AvatarChangeListener) -> Option<String> {
        // UDP loopback is fast but not synchronous; poll briefly.
        for _ in 0..100 {
            if let Some(id) = l.poll_avatar_change() {
                return Some(id);
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        None
    }

    #[test]
    fn empty_socket_polls_none_without_blocking() {
        let mut l = AvatarChangeListener::bind(0).unwrap();
        assert!(l.local_port() != 0);
        assert_eq!(l.poll_avatar_change(), None);
    }

    #[test]
    fn receives_avatar_change_id() {
        let mut l = AvatarChangeListener::bind(0).unwrap();
        send(l.local_port(), &change_msg("avtr_abc"));
        assert_eq!(recv_with_retry(&mut l).as_deref(), Some("avtr_abc"));
        // Drained: nothing left.
        assert_eq!(l.poll_avatar_change(), None);
    }

    #[test]
    fn last_of_several_changes_wins_and_other_addresses_are_ignored() {
        let mut l = AvatarChangeListener::bind(0).unwrap();
        send(
            l.local_port(),
            &rosc::OscPacket::Message(rosc::OscMessage {
                addr: "/avatar/parameters/JawOpen".to_string(),
                args: vec![rosc::OscType::Float(0.5)],
            }),
        );
        send(l.local_port(), &change_msg("avtr_first"));
        send(l.local_port(), &change_msg("avtr_second"));
        // Wait until both datagrams are queued, then drain in one poll.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(recv_with_retry(&mut l).as_deref(), Some("avtr_second"));
    }

    #[test]
    fn bundled_avatar_change_is_found() {
        let mut l = AvatarChangeListener::bind(0).unwrap();
        let bundle = rosc::OscPacket::Bundle(rosc::OscBundle {
            timetag: rosc::OscTime {
                seconds: 0,
                fractional: 1,
            },
            content: vec![change_msg("avtr_bundled")],
        });
        send(l.local_port(), &bundle);
        assert_eq!(recv_with_retry(&mut l).as_deref(), Some("avtr_bundled"));
    }
}
