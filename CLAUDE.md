# CLAUDE.md

Guidance for AI assistants (and humans) working in the **VRChatCameraOSC**
repository.

> **Read [`README.md`](README.md) and the relevant issue first — every task
> starts there (rules 4 & 6).**

## Project summary

VRChatCameraOSC captures your face (and hands) from a single ordinary webcam,
estimates expression / head-pose / finger landmarks, and streams the resulting
parameters to **VRChat over OSC** so your own avatar moves in realtime. The
avatar itself is rendered by VRChat — this app only does capture → tracking →
OSC. No GUI window; everything runs in a TUI.

- **Language / stack:** Rust — chosen for a fast, predictable realtime loop.
- **UI:** TUI (terminal user interface) — the primary control surface, showing
  status, live tracked values, and configuration.
- **Transport:** OSC over UDP to VRChat. The tracker can run on the same machine
  as VRChat or on another host on the network (send to VRChat's OSC host:port).
- **Tracking:**
  - **Face mesh** — facial landmarks → expression / head pose.
  - **Hand tracking** — hand/finger landmarks → avatar gestures *(planned)*.
- **Models are pluggable:** keep each model behind a clear backend boundary so a
  different face/hand model can slot in without churn. Model runtime (e.g. an
  ONNX path via `ort`, or `candle`) is an implementation choice — decide it in an
  issue, not ad hoc.
- **Avatar mapping:** landmarks → VRChat avatar parameters (expression params,
  and/or the standard `GestureLeft` / `GestureRight` for hands). The exact
  parameter mapping is designed before it is wired up.
- **Target OS:** **cross-platform — macOS, Linux, and Windows** (decided
  2026-07-25; see issue #10). Development currently happens on Linux. Keep
  platform-specific code (camera capture: AVFoundation on macOS, V4L2 on
  Linux, MediaFoundation on Windows *(planned)*) behind the `capture::native`
  boundary so each backend slots in without churn. VRChat may run on a
  separate machine — OSC reaches it over the network.
- **[`unity/`](unity/) is a separate sub-project**: a Unity/C# VRChat SDK3
  Humanoid-avatar setup wizard (issue #16), distributed as a `.unitypackage`
  `Assets/`-folder drop-in (`unity/VRChatCameraOSC/`), not a UPM package —
  not covered by the Rust-focused rules below as written. It has its own
  `unity/README.md`, its own Unity Test Framework tests
  (`unity/VRChatCameraOSC/Tests/Editor/`), and its own build/verify loop
  (Unity Editor batch mode — `-runTests -testPlatform EditMode`), not
  `cargo`. Its 10-parameter spec
  (`unity/VRChatCameraOSC/Editor/OscParameterSpec.cs`) must be kept in sync
  by hand with `src/mapping/mod.rs` — there is no automated check across the
  Rust/C# boundary.

## Development rules (must follow)

These are hard rules. Do not skip them.

### 1. Develop with TDD
Practice test-driven development. Write a failing test first, make it pass with
the simplest change, then refactor (red → green → refactor). New behavior should
be accompanied by tests; do not add functionality without a test that covers it.

### 2. Always demo before pushing
**Never push without first demonstrating the change actually works.** Running the
test suite is necessary but not sufficient — exercise the real behavior (run the
app / the relevant component and observe it) before every `git push`. If a
change cannot be demoed for some reason, say so explicitly instead of pushing
silently. See [Demo before push](#demo-before-push) for the procedure.

### 3. Treat the documentation on GitHub as the source of truth
For any API, library, or tool manual, consult the **documentation published on
GitHub** (and the VRChat OSC docs) as the most up-to-date reference. Do not rely
on memory or training data for API details — verify against current upstream
docs. This matters especially for fast-moving dependencies.

### 4. Read `README.md` first
**Before starting any task, read [`README.md`](README.md).** It is the canonical
overview of the project's purpose, architecture, and current direction. Ground
your work in it before touching code or docs.

### 5. Update `README.md` when you finish
**When an implementation is done, update [`README.md`](README.md)** so it stays
accurate — features, roadmap checkboxes, usage, and any changed architecture or
commands. A change is not "done" until the README reflects it. Do this before
pushing (it is part of the demo/push checklist below).

### 6. Read the relevant issue before implementing
**Before writing any code, read the GitHub issue(s) covering the work**, plus any
linked/related issues and `docs/`. Understand the scope, task list, acceptance
criteria, and dependencies first. If no issue exists for the work, create one (or
ask) before implementing.

### 7. Update the issue to mark the work complete
**The issue is the definition of done.** When the work is finished, update the
relevant issue — check off completed tasks, note what was implemented/decided,
and close it (or mark it complete). Implementation is **not** "done" until the
issue is updated. Do this before/at push time.

### 8. Ask instead of guessing when there's ambiguity or a design gap
**If multiple interpretations/approaches are plausible, or the issue/design is
incomplete or inconsistent, stop and ask — do not guess and implement.** Don't
silently pick one path or paper over a design deficiency. Surface the ambiguity,
lay out the options with a recommendation, and get a decision before proceeding.
When the answer matters, record it in the issue/docs so it isn't re-litigated.

## Demo before push

A "demo" means exercising the real behavior of your change and observing the
result — not just a green test suite. Because VRChat renders the avatar, a demo
usually means **sending OSC and observing the effect**, and/or inspecting the
live values the tracker produces.

Typical demos as components land:

- **OSC dry-run / monitor** — run the tracker and print (or show in the TUI) the
  OSC messages it would send, so mapping can be verified without VRChat.
- **Live capture** — run against the webcam and confirm landmarks / parameters
  update in realtime in the TUI (watch FPS and latency).
- **End-to-end** — enable OSC in VRChat (Action Menu → Options → OSC → Enabled),
  run the tracker, and confirm the avatar moves as expected.

When you push, briefly record **what you ran and what you observed** in the
commit message or PR description so the demo is auditable.

## Working with the codebase

### Build & test

```bash
cargo build
cargo test
```

### Run

```bash
cargo run --release
```

VRChat must have **OSC enabled** (Action Menu → Options → OSC → Enabled) for the
end-to-end path.

### Before pushing — checklist
- [ ] `README.md` read at the start of the task (rule 4)
- [ ] Relevant issue(s) read before implementing (rule 6)
- [ ] Ambiguities / design gaps raised and resolved, not guessed (rule 8)
- [ ] Tests written first / updated (TDD)
- [ ] `cargo test` passes
- [ ] Change demoed against real behavior, and what you ran/observed is recorded (rule 2)
- [ ] Any API usage verified against current GitHub / VRChat OSC docs (rule 3)
- [ ] `README.md` updated to reflect the change (rule 5)
- [ ] Issue updated / tasks checked off / closed (rule 7)

## Conventions

- Keep new code consistent with the surrounding style; run `cargo fmt` and
  `cargo clippy`.
- **Cross-platform (macOS, Linux, Windows)** — currently developed/verified on
  Linux. Keep platform-specific code (e.g. capture: AVFoundation on macOS,
  V4L2 on Linux) behind clear boundaries so other targets slot in without
  churn.
- The TUI is the primary control surface — no separate GUI window. The avatar is
  rendered by VRChat, not by this app.

## License

TBD. (Keep this in sync with [`README.md`](README.md).)
