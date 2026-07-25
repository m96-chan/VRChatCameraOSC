//! VRChatCameraOSC entry point.
//!
//! Wires the realtime loop capture → track → map → OSC and drives the TUI.
//! The full wiring lands with issue #8; for now this is a minimal smoke entry
//! so `cargo run` builds and exits cleanly.

use anyhow::Result;

fn main() -> Result<()> {
    println!("VRChatCameraOSC — scaffold. Realtime loop + TUI land with issue #8.");
    Ok(())
}
