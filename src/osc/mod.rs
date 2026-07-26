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

/// A single OSC message to VRChat — usually an avatar parameter update, or
/// (issue #19) a raw-address message like the native eye-tracking endpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct OscParam {
    /// Parameter name (without the `/avatar/parameters/` prefix), **or** a
    /// full OSC address if it starts with `/` (see [`OscParam::address`]).
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

    /// A single-float message at a raw OSC address (`addr` must start with
    /// `/`), e.g. `/tracking/eye/EyesClosedAmount` (issue #19).
    pub fn raw_float(addr: impl Into<String>, v: f32) -> Self {
        let addr = addr.into();
        debug_assert!(addr.starts_with('/'), "raw address must start with /");
        Self {
            name: addr,
            value: OscValue::Float(v),
        }
    }

    /// A four-float message at a raw OSC address (`addr` must start with
    /// `/`), e.g. `/tracking/eye/LeftRightPitchYaw` (issue #19).
    pub fn raw_floats4(addr: impl Into<String>, v: [f32; 4]) -> Self {
        let addr = addr.into();
        debug_assert!(addr.starts_with('/'), "raw address must start with /");
        Self {
            name: addr,
            value: OscValue::Floats4(v),
        }
    }

    /// Full OSC address for this message: `name` verbatim when it is already
    /// a raw address (starts with `/`), else `/avatar/parameters/{name}`.
    pub fn address(&self) -> String {
        if self.name.starts_with('/') {
            self.name.clone()
        } else {
            format!("/avatar/parameters/{}", self.name)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OscValue {
    Float(f32),
    Bool(bool),
    Int(i32),
    /// Four floats in one message (`/tracking/eye/LeftRightPitchYaw` — the
    /// only multi-argument message VRChat expects from us; issue #19).
    Floats4([f32; 4]),
}

/// A sink for OSC parameter updates. Implemented by the real UDP sender and by
/// the dry-run monitor (issue #3).
pub trait OscSink {
    fn send(&mut self, params: &[OscParam]) -> anyhow::Result<()>;
}
