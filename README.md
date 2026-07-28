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
- **VRCFaceTracking-compatible output** — the tracker speaks the standard
  Unified Expressions `v2/` parameter set that VRCFT sends (issues #18/#21;
  this is the **only** output format — the former app-specific `custom10`
  set is retired). VRCFT-ready avatars work with **zero avatar-side setup**;
  plain avatars get a [setup wizard](#avatar-setup-required-for-the-avatar-to-actually-move).
  **Avatar-aware**: reads the worn avatar's OSC config (and follows
  `/avatar/change`) to send exactly the parameters it declares — float,
  bool, and **binary** (`<Name>1/2/4/8` + `Negative`) forms — at their
  exact addresses. **OSCQuery** (mDNS) makes both directions work with a
  VRChat on another machine, and lets VRChat auto-discover the tracker with
  no port setup.
- **Native eye tracking** (default ON) — gaze + blink over VRChat's own
  `/tracking/eye/*` OSC endpoints, which drive the avatar's **existing eye
  bones directly** — works on any avatar, no parameters, no wizard (issue #19).
- **TUI** — monitor and control everything from the terminal.

## Tech

- **Language:** Rust — chosen for a fast, predictable realtime loop.
- **UI:** TUI (terminal user interface).
- **Transport:** OSC over UDP to VRChat.

## Status

🚧 Early development. Interfaces and parameters are subject to change. The full
pipeline (capture → detect → face mesh → mapping → OSC) is wired end-to-end
on the **MediaPipe** stack (YuNet → FaceMesh V2 → Blendshape V2, ported from
[AvataCam](https://github.com/m96-chan/AvataCam)). Model weights
auto-download on first run. Hand/finger tracking is the main remaining gap
(see Roadmap).

> The original FAN (S3FD → 2DFAN4) 68-landmark backend and the app-specific
> `custom10` parameter set were **retired in issue #21** — everything now
> rides the standard VRCFT Unified Expressions `v2/*` format.

## Architecture

Each stage sits behind a trait so platform- and model-specific pieces stay
swappable:

| Stage | Module | Notes |
|-------|--------|-------|
| Capture | `capture::CameraSource` | `capture::native::NativeCamera` — AVFoundation on macOS, V4L2 on Linux, MediaFoundation on Windows (build-verified in CI; runtime verification on real Windows hardware pending), via `nokhwa`; synthetic `FakeCamera` for tests/headless |
| Tracking | `tracking::arkit::ArkitFaceTracker` | `mediapipe::MediapipeTracker` — **YuNet** detector → rotated eyes-aligned ROI → **FaceMesh V2** (478 3-D landmarks, on **burn-wgpu** GPU with candle-onnx CPU fallback) → **Blendshape V2** (52 ARKit coefficients) + per-axis head rotation. Behind a trait so a different expression model can slot in ("models are pluggable") |
| Mapping | `mapping::unified::UnifiedMapper` | 52 ARKit coefficients → Unified Expressions shapes (VRCFT LiveLink correlation) → the VRCFT `v2/` parameter set + `*TrackingActive` bools, with per-channel neutral-baseline calibration (per-eye blink self-calibration) and One-Euro smoothing (`mapping::arkit` shared machinery) |
| Avatar gating | `mapping::avatar` + `osc::AvatarChangeListener` | reads VRChat's per-avatar OSC config JSON and follows `/avatar/change` (nonblocking listen on `[osc] listen_port`, default 9001) → send only declared params at exact addresses, in declared float/bool/binary forms (issue #18 phases 2–3) |
| OSCQuery | `osc::oscquery` | mDNS (`mdns-sd`) advertisement of our OSC input (`_osc._udp` + `_oscjson._tcp` with a minimal HTTP `?HOST_INFO`/root-node responder) so VRChat auto-discovers us, plus discovery of `VRChat-Client-*` and HTTP fetch of the avatar's declared parameters — local-file → OSCQuery → blind-prefix priority (issue #18 phase 3b) |
| Eye (native) | `mapping::eye::NativeEyeMapper` | 12 eye channels → `/tracking/eye/LeftRightPitchYaw` (degrees) + `EyesClosedAmount`, appended to the parameter output; `[eye]` / `--native-eye` (issue #19) |
| OSC | `osc::OscSink` | `UdpOscSender` to VRChat, or `MonitorSink` dry-run |
| Loop | `pipeline::Pipeline` | `capture → track → map → OSC`, driven by the TUI or headless monitor |
| Models | `models::ensure_present` | auto-downloads `models/*.onnx` from a GitHub Release on first run |

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

The tracker loads `models/face_detection.onnx` (YuNet),
`models/face_landmarks.onnx` (FaceMesh V2), and `models/face_blendshapes.onnx`
(Blendshape V2). **All auto-download on first run** (from this
repo's [`models-v1` release](https://github.com/m96-chan/VRChatCameraOSC/releases/tag/models-v1))
if missing — no Python/PyTorch needed. Offline, or if the download fails, the
loop still runs with tracking disabled and prints a clear message.

CLI flags: `--monitor` (headless), `--fake` (synthetic camera), `--frames N`
(stop after N frames), `--native-eye on|off` (overrides `[eye]
native`; see Avatar setup below), `--oscquery on|off` (overrides `[osc]
oscquery`; see Avatar setup below), `--detect-interval N`
(see Performance below), `--calibrate-frames N` (see Calibration below; `0`
skips calibration).

VRChat must have **OSC enabled** (Action Menu → Options → OSC → Enabled) for the
end-to-end avatar path.

### Neutral-pose calibration

The tracker captures a short window at startup (`--calibrate-frames N`,
default 10 frames) — **hold a relaxed, forward-facing expression** while it
says `calibrating neutral pose ...`. In the TUI, press **`c`** at any time to
redo it (camera angle or lighting changed, or you weren't ready the first
time).

The raw Blendshape V2 coefficients have a small
non-zero resting baseline per channel (e.g. `jawOpen` ≈ 0.2 with the mouth
closed, and each eye's `eyeBlink` baseline differs). Calibration subtracts
the per-channel resting baseline so a resting face reads `0` everywhere, and
blink is self-calibrated **per eye** — open reads `0`, a blink a fixed gain
above your own baseline saturates to `1`. The head's "facing front" pose is
baselined the same way. If startup calibration is skipped, the same
baselines are accumulated automatically over the first frames of tracking.

### Performance

The tracker runs at **camera rate — measured 30 FPS**
(640×480@30 webcam, `--release`, default features). Three things make that
possible (issue #17):

- **FaceMesh on burn-wgpu** (`mesh-gpu` feature, default ON): GPU via
  Vulkan/Metal/DX12 with only a graphics driver — no CUDA toolkit. The
  candle-onnx CPU interpreter measured ~105 ms/frame for this
  depthwise-conv-heavy graph (and is kernel-launch-bound on CUDA), nowhere
  near realtime; the burn executor uploads the weights to the GPU once at
  load. No usable GPU → automatic fallback to the CPU stage (a few FPS).
- **Async safety-net redetect**: YuNet (~216 ms on CPU) only reseeds the ROI —
  it runs synchronously on the first frame and after losing the face, and
  otherwise about once a second **on its own thread**, so the frame loop
  never blocks on it. Between detections the ROI comes from the previous
  frame's landmarks.
- **Cached ONNX initializers** (fork addition): graph weights are extracted
  once at load instead of being re-parsed from the proto every evaluation.

The blendshape stage (~2 ms) stays on CPU.

## Avatar setup (required for the avatar to actually move)

VRChat's OSC layer only **writes values into parameters your avatar already
defines** — it doesn't create facial expressions on its own, and none of
VRChat's built-in avatar parameters (`GestureLeft/Right`, `Viseme`, `AFK`,
etc.) cover facial expression or head pose. A stock/default avatar, or any
avatar that hasn't been specifically wired up, **will not react** to this
app's output.

**Exception — eye tracking**: VRChat's native `/tracking/eye/*` OSC endpoints
(sent by default, `[eye] native` / `--native-eye`) drive the avatar's
**existing eye bones directly**, so gaze and blink work on *any* avatar with
eye bones configured — no parameters, no wizard, nothing to set up (issue
#19; [VRChat docs](https://docs.vrchat.com/docs/osc-eye-tracking)). Gaze
range gains are `[eye] yaw_range_deg` / `pitch_range_deg` (defaults 30°/25°).
If the face is lost, the data times out after 10 s and VRChat's own eye
behaviour resumes automatically.

Since issue #21 there is a **single output format**: the standard VRCFT
(Unified Expressions) `v2/*` parameter set. Expressiveness tiers at a glance
(松竹梅 — issue #19):

| Tier | Avatar-side work | What moves |
|---|---|---|
| 梅 | run the `unity/` wizard (plain avatars) | the wizard's v2/* subset: blink, brows, jaw, smile, stretch, head |
| 竹 | none for the eyes (native eye, default ON) | + gaze & blink on any avatar with eye bones |
| 松 | none (FT-ready avatar) | full Unified Expressions float set (~145 params) |

### Option 1 — VRCFT-ready avatars: no setup (issue #18)

If your avatar already supports **VRCFaceTracking / Unified Expressions**
(most commercial face-tracking-ready avatars do), just run the tracker: it
emits the same `/avatar/parameters/<prefix>v2/<Name>` parameters VRCFT
sends, plus the
`EyeTrackingActive` / `ExpressionTrackingActive` / `LipTrackingActive` bools
that FT avatars gate their animator layers on. The avatar needs **no wizard
and no Unity work**. Formulas are ported from VRCFT's own parameter
definitions and its official LiveLink (ARKit) module; see
[`src/mapping/unified.rs`](src/mapping/unified.rs) for sources and details.

**Avatar-aware gating + binary parameters** (issue #18 phases 2–3): like
VRCFT itself, the tracker reads VRChat's per-avatar OSC config JSON
(`.../AppData/LocalLow/VRChat/VRChat/OSC/usr_*/Avatars/<avatarid>.json`) and
then sends **only the parameters the avatar declares, at their exact declared
addresses** (any user prefix works, not just `FT/`) — each value in whichever
forms are declared: **float**, plain **bool** (VRCFT's `value < 0.5`
semantics), and **binary** bit params (`<Name>1/2/4/8...` + optional
`<Name>Negative`, `BinaryBaseParameter` encoding ported faithfully), so
binary-only synced-parameter avatars now move too. Avatar switches are
followed live via VRChat's `/avatar/change` OSC output; at startup the most
recently modified avatar config is used as a best-effort guess. See
[`src/mapping/avatar.rs`](src/mapping/avatar.rs).

**OSCQuery — remote-host VRChat, zero setup** (issue #18 phase 3b, on by
default): the tracker also speaks
[OSCQuery](https://github.com/vrchat-community/vrc-oscquery-lib), in both
directions, like VRCFT does. It **advertises itself** over mDNS
(`_osc._udp` + `_oscjson._tcp`, instance `VRChatCameraOSC-<hex>`, with a
minimal HTTP endpoint serving `?HOST_INFO` and an address tree declaring
`/avatar/change`), so VRChat — on the same machine **or another host on the
LAN** — auto-discovers the tracker and starts sending it `/avatar/change`
with no VRChat-side configuration. And it **discovers VRChat** the same way
(`_oscjson._tcp`, `VRChat-Client-*`) and fetches the worn avatar's declared
parameters over HTTP, so avatar-aware gating above works even when the
local config-file directory doesn't exist (VRChat on another machine).
Source priority: local config file → OSCQuery fetch → blind-prefix
fallback. Every failure (no mDNS, HTTP error, malformed JSON) degrades to
the next source; nothing crashes the loop. See
[`src/osc/oscquery.rs`](src/osc/oscquery.rs).

Related config keys (`config.toml`):

- `[osc] listen_port` — UDP port for VRChat's OSC output (`/avatar/change`).
  Default `9001` (VRChat's default); `0` disables the listener.
- `[osc] oscquery` — OSCQuery advertisement + discovery (above). Default
  `true`; set `false` (or `--oscquery off`) when you configure ports
  manually and don't want mDNS traffic. `listen_port = 0` also disables the
  advertisement (there'd be nothing to advertise).
- `[mapping] avatar_config_dir` — explicit avatar-OSC-config directory.
  Unset → auto-discovery: `%USERPROFILE%\AppData\LocalLow\VRChat\VRChat\OSC`
  on Windows, the Steam/Proton prefix
  (`~/.steam/steam/steamapps/compatdata/438100/pfx/...`) on Linux. Useful
  when VRChat runs on another machine — copy/mount its `OSC` folder and
  point this at it.
- `[mapping] vrcft_prefixes` — **fallback only**: when no avatar config is
  found (or the worn avatar has none) **and** OSCQuery finds nothing, the
  full float set is sent blind under each prefix (default `""` and `"FT/"`),
  exactly the phase-1 behavior. VRChat ignores undeclared addresses.

Current limits: if a *remote*
VRChat's mDNS advertisement only carries a loopback address (older VRChat
builds always advertise `127.0.0.1`), it cannot be resolved from another
machine — copy/mount its `OSC` folder and use `avatar_config_dir` instead.

### Option 2 — plain avatars: the `unity/` wizard builds a "lite" FT avatar

For an avatar with ordinary blend shapes and no face-tracking setup,
[`unity/`](unity/) is an editor wizard (VRChat SDK3, Humanoid avatars) that
wires the webcam-trackable subset of the same standard `v2/*` parameters to
the blend shapes / head bone your avatar already has — generating the VRC
Expression Parameters and the FX/Gesture Animator Controller layers for you:

| OSC address (`/avatar/parameters/…`) | Range | Meaning |
|---|---|---|
| `v2/EyeLidLeft`, `v2/EyeLidRight` | `0..1` | eyelid, VRCFT semantics: `0` = closed, `0.75` = relaxed open (declared default), `1` = wide |
| `v2/BrowUpLeft`, `v2/BrowUpRight` | `0..1` | eyebrow raise |
| `v2/JawOpen` | `0..1` | mouth opening amount |
| `v2/MouthSmileLeft`, `v2/MouthSmileRight` | `0..1` | mouth-corner lift (smile) |
| `v2/MouthStretchLeft`, `v2/MouthStretchRight` | `0..1` | mouth-corner horizontal stretch (wide/grin) |
| `v2/Head/Yaw`, `v2/Head/Pitch`, `v2/Head/Roll` | `-1..1` | head turn / nod / tilt (Humanoid head bone, additive layer) |

Because these are the standard VRCFT names, a wizard-made avatar also works
with VRCFaceTracking itself, and the tracker's avatar-aware gating sends
exactly this declared subset. Avatars set up with the retired pre-#21
custom10 wizard are migrated in place by re-running Apply (then re-upload).
See [`unity/README.md`](unity/README.md) for the full walkthrough, and
VRChat's own docs for the mechanics:
[OSC Avatar Parameters](https://docs.vrchat.com/docs/osc-avatar-parameters),
[Avatar Animator Parameters](https://creators.vrchat.com/avatars/animator-parameters/).

## Configuration

OSC host/port, camera device, and tracking settings will be configurable from the TUI and/or a config file. (TBD — see Roadmap.)

## Roadmap

- [x] Camera capture pipeline
- [x] MediaPipe tracking stack (YuNet → FaceMesh V2 → Blendshape V2, ARKit-52) — AvataCam-parity expression quality
- [ ] Hand / finger landmark extraction
- [x] ARKit blendshapes → OSC mapping (per-eye blink self-calibration, One-Euro smoothing)
- [x] VRCFaceTracking-compatible output — Unified Expressions `v2/` float params (issue #18 phase 1)
- [x] VRCFT binary parameter encoding + avatar-config param gating with `/avatar/change` tracking (issue #18 phases 2–3)
- [x] OSCQuery: mDNS advertisement + remote avatar-parameter discovery (issue #18 phase 3b)
- [x] Native eye tracking — `/tracking/eye/*` gaze + blink, any avatar, zero setup (issue #19)
- [x] Standardize on VRCFT `v2/*` everywhere — custom10 + FAN backend retired; wizard builds lite FT avatars (issue #21)
- [ ] Tongue tracking — needs a model with a tongue signal; R&D (issue #20)
- [x] OSC sender (UDP) + dry-run monitor
- [x] TUI: live values, status, and configuration
- [x] Config file support
- [x] Unity SDK: avatar setup wizard (Humanoid) — see [`unity/`](unity/)

## License

TBD.
