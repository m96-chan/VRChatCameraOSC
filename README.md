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

🚧 Early development. Interfaces and parameters are subject to change. The full
pipeline (capture → detect → face mesh → mapping → OSC) is wired end-to-end,
including face auto-crop (S3FD) before FAN. Model weights auto-download on
first run. Hand/finger tracking is the main remaining gap (see Roadmap).

## Architecture

Each stage sits behind a trait so platform- and model-specific pieces stay
swappable:

| Stage | Module | Notes |
|-------|--------|-------|
| Capture | `capture::CameraSource` | `capture::native::NativeCamera` — AVFoundation on macOS, V4L2 on Linux (`Windows` planned), via `nokhwa`; synthetic `FakeCamera` for tests/headless |
| Detection | `tracking::detect::FaceDetector` | `sfd::SfdDetector` — the **S3FD** face detector ported to candle; auto-crops before FAN |
| Tracking | `tracking::FaceTracker` | `fan::FanTracker` — the face-alignment **2DFAN4** net ported to pure-Rust **candle** |
| Mapping | `mapping::Mapper` | iBUG-68 landmarks → normalised avatar params (mouth/blink/brows/head pose), clamped + smoothed |
| OSC | `osc::OscSink` | `UdpOscSender` to VRChat, or `MonitorSink` dry-run |
| Loop | `pipeline::Pipeline` | `capture → track → map → OSC`, driven by the TUI or headless monitor |
| Models | `models::ensure_present` | auto-downloads `models/*.safetensors` from a GitHub Release on first run |

### Model & PyTorch parity

Face landmarks come from a candle port of [`1adrianb/face-alignment`](https://github.com/1adrianb/face-alignment)
(2DFAN4). The port is validated for **numeric parity** against the PyTorch
reference: pretrained weights are exported to safetensors and the Rust output is
compared to PyTorch on identical input. Observed agreement is to f32 precision
(full-network max abs diff ≈ 4e-7). See [`reference/`](reference/) and the
`fan_parity` / `fan_convblock_parity` / `fan_units` tests.

## Getting Started

> Requires a recent stable Rust toolchain ([rustup](https://rustup.rs/)).

```bash
# build
cargo build --release

# run (TUI)
cargo run --release

# headless OSC monitor / dry-run demo (no VRChat needed):
#   prints the /avatar/parameters/* it would send, using a synthetic camera
cargo run --release -- --monitor --fake --frames 20
```

The tracker loads FAN weights from `models/2dfan4.safetensors` and the S3FD
detector from `models/s3fd.safetensors`. **Both auto-download on first run**
(from this repo's [`models-v1` release](https://github.com/m96-chan/VRChatCameraOSC/releases/tag/models-v1))
if missing — no Python/PyTorch needed. Offline, or if the download fails, the
loop still runs with tracking disabled and prints a clear message.

Contributors validating the candle port against PyTorch (numeric-parity
tests) still use the reference harness to regenerate these from the original
pretrained weights — see [`reference/README.md`](reference/README.md):

```bash
cd reference && uv venv --python 3.11
uv pip install --python .venv "torch>=2.2" "numpy<2" safetensors face-alignment scikit-image
.venv/bin/python gen_fixtures.py   # downloads + converts the pretrained weights
```

CLI flags: `--monitor` (headless), `--fake` (synthetic camera), `--frames N`
(stop after N frames), `--weights PATH`, `--detector PATH` (a custom
`--weights`/`--detector` path is never auto-downloaded).

VRChat must have **OSC enabled** (Action Menu → Options → OSC → Enabled) for the
end-to-end avatar path.

## Configuration

OSC host/port, camera device, and tracking settings will be configurable from the TUI and/or a config file. (TBD — see Roadmap.)

## Roadmap

- [x] Camera capture pipeline
- [x] Face mesh landmark extraction (FAN / candle, PyTorch-parity verified)
- [x] Face detector (S3FD / candle) — auto-crop the face before FAN
- [ ] Hand / finger landmark extraction
- [x] Landmark → VRChat OSC parameter mapping
- [x] OSC sender (UDP) + dry-run monitor
- [x] TUI: live values, status, and configuration
- [x] Config file support

## License

TBD.
