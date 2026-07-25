//! OSC output to VRChat: message building + UDP transport / dry-run monitor.
//!
//! VRChat avatar parameters live under `/avatar/parameters/<Name>` and accept
//! float/bool/int values. See the VRChat OSC docs.

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
