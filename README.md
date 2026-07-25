# VRChatCameraOSC

Realtime face tracking for VRChat, driven from your webcam and delivered over OSC — written in Rust for a fast, low-latency loop, with a TUI front end.

## Overview

`VRChatCameraOSC` captures your face from a camera, extracts a face mesh, and streams the resulting expression / head-pose parameters to VRChat over OSC so your own avatar moves in realtime. No GUI window — everything runs in a terminal UI.

```
┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌─────────┐
│  Camera  │──▶│  Face Mesh   │──▶│ OSC Parameters│──▶│ VRChat  │
│ capture  │   │  tracking    │   │  (UDP/OSC)    │   │ avatar  │
└──────────┘   └──────────────┘   └───────────────┘   └─────────┘
                         ▲
                   ┌───────────┐
                   │    TUI    │  status / config / live values
                   └───────────┘
```

## Features

- **Camera capture & tracking** — grab frames from a webcam and track the face in realtime.
- **Face mesh** — extract facial landmarks (face mesh) to derive expression and pose.
- **Hand tracking** — track hand/finger landmarks and drive avatar gestures. *(planned)*
- **OSC output** — connect to VRChat over OSC and move your own avatar.
- **Realtime face tracking → avatar** — map tracked landmarks to avatar parameters live.
- **TUI** — monitor and control everything from the terminal.

## Tech

- **Language:** Rust — chosen for a fast, predictable realtime loop.
- **UI:** TUI (terminal user interface).
- **Transport:** OSC over UDP to VRChat.

## Status

🚧 Early development. Interfaces and parameters are subject to change.

## Getting Started

> Requires a recent stable Rust toolchain ([rustup](https://rustup.rs/)).

```bash
# build
cargo build --release

# run
cargo run --release
```

VRChat must have **OSC enabled** (Action Menu → Options → OSC → Enabled).

## Configuration

OSC host/port, camera device, and tracking settings will be configurable from the TUI and/or a config file. (TBD — see Roadmap.)

## Roadmap

- [ ] Camera capture pipeline
- [ ] Face mesh landmark extraction
- [ ] Hand / finger landmark extraction
- [ ] Landmark → VRChat OSC parameter mapping
- [ ] OSC sender (UDP)
- [ ] TUI: live values, status, and configuration
- [ ] Config file support

## License

TBD.
