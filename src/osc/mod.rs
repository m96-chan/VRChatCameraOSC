//! OSC output to VRChat: message building + UDP transport / dry-run monitor.
//!
//! # Address & value design (issue #3, CLAUDE.md rule 8)
//!
//! VRChat avatar parameters live under `/avatar/parameters/<Name>` and accept
//! float/bool/int values (verified against the VRChat OSC docs). We therefore
//! commit to the following so it is not re-litigated:
//!
//! - **Address:** every [`OscParam`] maps to exactly one OSC address,
//!   `/avatar/parameters/{name}` (see [`OscParam::address`]). Callers pass the
//!   bare parameter name; the prefix is owned here.
//! - **Type mapping:** each param carries a single OSC argument —
//!   `Float -> OscType::Float(f32)`, `Bool -> OscType::Bool`,
//!   `Int -> OscType::Int(i32)`. VRChat's input parameters are exactly these
//!   three types, so no other `OscType` variants are emitted.
//! - **Transport:** one OSC message per parameter, sent as an individual UDP
//!   datagram (`rosc::encoder::encode` + `UdpSocket::send_to`). We deliberately
//!   do **not** bundle, matching VRChat's per-parameter input handling and
//!   keeping the dry-run monitor line-for-line with the wire.
//! - **Monitor format:** the dry-run [`MonitorSink`] renders each message as
//!   `"<address> <tag> <value>"` where `tag` is `f`/`b`/`i` — a stable,
//!   greppable, one-line-per-message form for the TUI and tests.

mod encode;
mod monitor;
mod rate;
mod udp;

pub use encode::{encode_param, to_message};
pub use monitor::MonitorSink;
pub use rate::SendRate;
pub use udp::UdpOscSender;

/// A single VRChat avatar parameter update.
#[derive(Debug, Clone, PartialEq)]
pub struct OscParam {
    /// Parameter name (without the `/avatar/parameters/` prefix).
    pub name: String,
    pub value: OscValue,
}

impl OscParam {
    pub fn float(name: impl Into<String>, v: f32) -> Self {
        Self {
            name: name.into(),
            value: OscValue::Float(v),
        }
    }
    pub fn bool(name: impl Into<String>, v: bool) -> Self {
        Self {
            name: name.into(),
            value: OscValue::Bool(v),
        }
    }
    pub fn int(name: impl Into<String>, v: i32) -> Self {
        Self {
            name: name.into(),
            value: OscValue::Int(v),
        }
    }

    /// Full OSC address for this parameter.
    pub fn address(&self) -> String {
        format!("/avatar/parameters/{}", self.name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OscValue {
    Float(f32),
    Bool(bool),
    Int(i32),
}

/// A sink for OSC parameter updates. Implemented by the real UDP sender and by
/// the dry-run monitor (issue #3).
pub trait OscSink {
    fn send(&mut self, params: &[OscParam]) -> anyhow::Result<()>;
}
