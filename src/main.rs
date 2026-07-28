//! VRChatCameraOSC entry point — wires the realtime loop capture → track → map
//! → OSC (issue #8) and drives it in one of two modes:
//!
//! - `--monitor` (headless): step the pipeline and print the OSC parameters it
//!   would send. This is the CLAUDE.md "OSC dry-run / monitor" demo and needs no
//!   TUI. `--frames N` bounds the run (useful for demos / CI).
//! - default (TUI): the terminal control surface, updating live values, status,
//!   and config while the pipeline runs.
//!
//! Camera: the native backend (AVFoundation on macOS, V4L2 on Linux) is used
//! when available; `--fake` forces the synthetic source (also the automatic
//! fallback when no webcam is present). Tracker: the MediaPipe face stack
//! (auto-downloaded ONNX models); without it the loop runs but emits no
//! parameters (a clear warning is shown).

use std::io::{self, Stdout};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use vrchat_camera_osc::capture::{CameraSource, FakeCamera, FpsCounter};
use vrchat_camera_osc::config::Config;
use vrchat_camera_osc::mapping::arkit::ArkitMappingConfig;
use vrchat_camera_osc::mapping::avatar::{self, AvatarWatcher};
use vrchat_camera_osc::mapping::eye::{EyeRange, NativeEyeMapper};
use vrchat_camera_osc::mapping::unified::UnifiedMapper;
use vrchat_camera_osc::models;
use vrchat_camera_osc::osc::oscquery::{MdnsVrchatDiscovery, OscQueryAdvertiser, OscQueryClient};
use vrchat_camera_osc::osc::{AvatarChangeListener, MonitorSink, OscSink, UdpOscSender};
use vrchat_camera_osc::pipeline::Pipeline;
use vrchat_camera_osc::tracking::arkit::ArkitFaceTracker;
use vrchat_camera_osc::tracking::mediapipe::MediapipeTracker;
use vrchat_camera_osc::tui::{render, UiState};

/// Default neutral-pose calibration window (issue #15): captured once at
/// startup (and again on demand via `c` in the TUI), assuming the subject
/// holds a relaxed, forward-facing expression throughout.
const DEFAULT_CALIBRATE_FRAMES: u32 = 10;

struct Args {
    monitor: bool,
    fake: bool,
    frames: Option<u64>,
    /// `None` = the backend default (~once a second safety-net redetect).
    detect_interval: Option<u32>,
    calibrate_frames: u32,
    /// CLI override for `eye.native` (`--native-eye on|off`).
    native_eye: Option<bool>,
    /// CLI override for `osc.oscquery` (`--oscquery on|off`).
    oscquery: Option<bool>,
}

fn parse_args() -> Args {
    let mut a = Args {
        monitor: false,
        fake: false,
        frames: None,
        detect_interval: None,
        calibrate_frames: DEFAULT_CALIBRATE_FRAMES,
        native_eye: None,
        oscquery: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--monitor" => a.monitor = true,
            "--fake" => a.fake = true,
            "--frames" => a.frames = it.next().and_then(|v| v.parse().ok()),
            "--detect-interval" => {
                if let Some(n) = it.next().and_then(|v| v.parse().ok()) {
                    a.detect_interval = Some(n);
                }
            }
            "--calibrate-frames" => {
                if let Some(n) = it.next().and_then(|v| v.parse().ok()) {
                    a.calibrate_frames = n;
                }
            }
            "--mapping" | "--backend" => {
                // Retired flags (issue #21): there is a single format/backend
                // now. Consume the value so the rest of the args still parse.
                let val = it.next();
                eprintln!(
                    "{arg} is retired (issue #21): the tracker always emits the \
                     VRCFT Unified Expressions v2/* set from the MediaPipe \
                     backend — ignoring {arg} {}",
                    val.as_deref().unwrap_or("")
                );
            }
            "--native-eye" => match it.next().as_deref() {
                Some("on") => a.native_eye = Some(true),
                Some("off") => a.native_eye = Some(false),
                other => {
                    eprintln!("--native-eye expects 'on' or 'off' (got {other:?}) — using config")
                }
            },
            "--oscquery" => match it.next().as_deref() {
                Some("on") => a.oscquery = Some(true),
                Some("off") => a.oscquery = Some(false),
                other => {
                    eprintln!("--oscquery expects 'on' or 'off' (got {other:?}) — using config")
                }
            },
            "-h" | "--help" => {
                println!(
                    "vrchat-camera-osc [--monitor] [--fake] [--frames N] \
                     [--native-eye on|off] [--oscquery on|off] \
                     [--detect-interval N] [--calibrate-frames N]"
                );
                std::process::exit(0);
            }
            _ => eprintln!("ignoring unknown argument: {arg}"),
        }
    }
    a
}

fn build_camera(cfg: &Config, fake: bool) -> Box<dyn CameraSource> {
    if !fake {
        if let Some(cam) =
            try_real_camera(cfg.camera.device_index, cfg.camera.width, cfg.camera.height)
        {
            return cam;
        }
        eprintln!("no webcam available — falling back to the synthetic FakeCamera");
    }
    Box::new(FakeCamera::new(cfg.camera.width, cfg.camera.height))
}

#[cfg(all(
    any(target_os = "macos", target_os = "linux", target_os = "windows"),
    feature = "camera"
))]
fn try_real_camera(index: u32, w: u32, h: u32) -> Option<Box<dyn CameraSource>> {
    use vrchat_camera_osc::capture::native::NativeCamera;
    match NativeCamera::open(index, w, h) {
        Ok(cam) => Some(Box::new(cam)),
        Err(e) => {
            eprintln!("could not open webcam {index}: {e}");
            None
        }
    }
}

#[cfg(not(all(
    any(target_os = "macos", target_os = "linux", target_os = "windows"),
    feature = "camera"
)))]
fn try_real_camera(_index: u32, _w: u32, _h: u32) -> Option<Box<dyn CameraSource>> {
    None
}

fn build_sink(cfg: &Config) -> Result<Box<dyn OscSink>> {
    if cfg.osc.dry_run {
        Ok(Box::new(MonitorSink::printing()))
    } else {
        Ok(Box::new(UdpOscSender::new(&cfg.osc.host, cfg.osc.port)?))
    }
}

/// Download `path` from `url` on first run. Never fails the caller: a
/// download error just logs and leaves `path` missing, which the existing
/// "tracking disabled" fallback handles.
fn ensure_model_present(path: &Path, url: &str) {
    if path.exists() {
        return;
    }
    eprintln!("downloading {} (first run only) ...", path.display());
    match vrchat_camera_osc::models::ensure_present(path, url) {
        Ok(()) => eprintln!("downloaded {}", path.display()),
        Err(e) => eprintln!(
            "could not download {}: {e:#} — continuing without it",
            path.display()
        ),
    }
}

/// Build the MediaPipe (ARKit-blendshape) tracker — issue #17. Auto-downloads
/// the three ONNX models on first run.
fn build_arkit_tracker(detect_interval: Option<u32>) -> Option<Box<dyn ArkitFaceTracker>> {
    let detection = models::default_face_detection_path();
    let landmarks = models::default_face_landmarks_path();
    let blendshapes = models::default_face_blendshapes_path();
    ensure_model_present(&detection, models::FACE_DETECTION_MODEL_URL);
    ensure_model_present(&landmarks, models::FACE_LANDMARKS_MODEL_URL);
    ensure_model_present(&blendshapes, models::FACE_BLENDSHAPES_MODEL_URL);

    match MediapipeTracker::from_paths(&detection, &landmarks, &blendshapes) {
        Ok(t) => {
            let t = match detect_interval {
                Some(n) => t.with_detect_interval(n),
                None => t, // keep the backend default (~once a second)
            };
            Some(Box::new(t))
        }
        Err(e) => {
            eprintln!(
                "failed to load the MediaPipe face stack: {e:#}\n\
                 The loop still runs; it just won't emit parameters."
            );
            None
        }
    }
}

/// Build the avatar-aware machinery (issue #18
/// phases 3 + 3b): the config-dir scan (explicit `[mapping]
/// avatar_config_dir` override, else per-platform VRChat defaults), the
/// `/avatar/change` listener on `[osc] listen_port` (default 9001; 0
/// disables it), and — when `oscquery` is on — mDNS discovery of a
/// (possibly remote) VRChat's OSCQuery endpoint plus our own advertisement
/// so VRChat auto-discovers us. Also reports what was (or wasn't) found, so
/// a silent fallback is visible to the user. The returned advertiser must be
/// kept alive for the run (dropping it withdraws the mDNS advertisement).
fn build_avatar_watcher(
    cfg: &Config,
    oscquery: bool,
) -> (AvatarWatcher, Option<OscQueryAdvertiser>) {
    let dirs = avatar::config_dirs(cfg.mapping.avatar_config_dir.as_deref());
    match avatar::find_most_recent_config(&dirs) {
        Some(c) => eprintln!(
            "vrcft: gating to avatar '{}' ({}) from its OSC config",
            c.name, c.id
        ),
        None if oscquery => eprintln!(
            "vrcft: no local avatar OSC config under {dirs:?} — trying OSCQuery \
             discovery (blind-prefix fallback if that finds nothing)"
        ),
        None => eprintln!(
            "vrcft: no avatar OSC config found under {:?} — sending the full \
             float set under the configured prefixes",
            dirs
        ),
    }
    let listener = match cfg.osc.listen_port {
        0 => None,
        port => match AvatarChangeListener::bind(port) {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!(
                    "vrcft: could not listen for /avatar/change on port {port}: {e:#} — \
                     avatar switches won't be detected this run"
                );
                None
            }
        },
    };
    let mut watcher = AvatarWatcher::new(dirs, listener);

    let mut advertiser = None;
    if oscquery {
        // Advertise ourselves (needs a live /avatar/change listen port —
        // that is the OSC input we are advertising).
        if cfg.osc.listen_port != 0 {
            match OscQueryAdvertiser::start(cfg.osc.listen_port) {
                Ok(adv) => {
                    eprintln!(
                        "oscquery: advertising '{}' — OSC on udp/{}, OSCQuery HTTP on tcp/{}{}",
                        adv.instance_name(),
                        cfg.osc.listen_port,
                        adv.http_port(),
                        if adv.mdns_active() {
                            ""
                        } else {
                            " (mDNS inactive — HTTP endpoint only)"
                        }
                    );
                    advertiser = Some(adv);
                }
                Err(e) => eprintln!(
                    "oscquery: could not start the advertiser: {e:#} — \
                     VRChat won't auto-discover this tracker"
                ),
            }
        } else {
            eprintln!("oscquery: [osc] listen_port = 0 — nothing to advertise");
        }
        // Discover VRChat's OSCQuery endpoint for avatar-parameter fetches.
        match MdnsVrchatDiscovery::start() {
            Ok(discovery) => {
                watcher = watcher.with_oscquery(OscQueryClient::new(Box::new(discovery)));
            }
            Err(e) => eprintln!(
                "oscquery: mDNS browse unavailable: {e:#} — \
                 remote avatar-parameter discovery disabled this run"
            ),
        }
    }
    (watcher, advertiser)
}

fn main() -> Result<()> {
    let args = parse_args();

    let cfg = Config::load_or_create(Config::default_path()).unwrap_or_else(|e| {
        eprintln!("using defaults (config load failed: {e})");
        Config::default()
    });
    cfg.validate()?;

    let camera = build_camera(&cfg, args.fake);
    let sink = build_sink(&cfg)?;
    let tracker = build_arkit_tracker(args.detect_interval);
    let mapper = UnifiedMapper::new(
        ArkitMappingConfig::default(),
        cfg.mapping.vrcft_prefixes.clone(),
    );
    let oscquery = args.oscquery.unwrap_or(cfg.osc.oscquery);
    let (watcher, advertiser) = build_avatar_watcher(&cfg, oscquery);
    // Kept alive for the whole run: dropping it withdraws our OSCQuery/mDNS
    // advertisement (issue #18 phase 3b).
    let _oscquery_advertiser: Option<OscQueryAdvertiser> = advertiser;
    let mut pipeline = Pipeline::new(camera, tracker, mapper, sink).with_avatar_watcher(watcher);
    if args.native_eye.unwrap_or(cfg.eye.native) {
        pipeline = pipeline.with_native_eye(NativeEyeMapper::new(
            ArkitMappingConfig::default(),
            EyeRange {
                yaw_deg: cfg.eye.yaw_range_deg,
                pitch_deg: cfg.eye.pitch_range_deg,
            },
        ));
    }

    calibrate_neutral(&mut pipeline, args.calibrate_frames);

    if args.monitor {
        run_monitor(pipeline, &cfg, args.frames, args.fake)
    } else {
        run_tui(pipeline, &cfg, args.calibrate_frames)
    }
}

/// Capture a short window assuming a relaxed, forward-facing expression and
/// re-derive the mapper's neutral-pose baselines from it (issue #15) — fixes
/// the synthetic-geometry defaults not matching every real face/camera, which
/// otherwise leaves some expression parameters clamped at one end of their
/// range. `frames == 0` (e.g. `--calibrate-frames 0`) skips this entirely.
/// Never fails the caller: an error just skips calibration for this run.
fn calibrate_neutral(pipeline: &mut Pipeline, frames: u32) {
    if frames == 0 {
        return;
    }
    println!("calibrating neutral pose — hold a relaxed, forward-facing expression...");
    match pipeline.calibrate_neutral(frames) {
        Ok(0) => eprintln!(
            "calibration found no face in {frames} frames — using default neutral thresholds"
        ),
        Ok(n) => println!("calibrated from {n}/{frames} frames"),
        Err(e) => eprintln!("calibration failed: {e:#} — using default neutral thresholds"),
    }
}

/// Headless: step the pipeline and print the OSC parameters each frame.
fn run_monitor(
    mut pipeline: Pipeline,
    cfg: &Config,
    frames: Option<u64>,
    fake_paced: bool,
) -> Result<()> {
    println!(
        "monitor: {} -> {}:{}{}",
        if cfg.osc.dry_run {
            "DRY-RUN"
        } else {
            "sending"
        },
        cfg.osc.host,
        cfg.osc.port,
        frames.map(|n| format!(" ({n} frames)")).unwrap_or_default()
    );
    let mut fps = FpsCounter::new(30);
    let mut last = Instant::now();
    let mut n = 0u64;
    loop {
        let outcome = pipeline.step()?;
        let now = Instant::now();
        fps.record(now.duration_since(last));
        last = now;

        if outcome.params.is_empty() {
            println!("[frame {n}] {:?} no face / no tracker", outcome.frame_size);
        } else {
            for p in &outcome.params {
                match p.value {
                    vrchat_camera_osc::osc::OscValue::Float(v) => {
                        println!("{} f {v:.4}", p.address())
                    }
                    vrchat_camera_osc::osc::OscValue::Bool(v) => {
                        println!("{} b {v}", p.address())
                    }
                    vrchat_camera_osc::osc::OscValue::Int(v) => {
                        println!("{} i {v}", p.address())
                    }
                    vrchat_camera_osc::osc::OscValue::Floats4([a, b, c, d]) => {
                        println!("{} f4 {a:.2} {b:.2} {c:.2} {d:.2}", p.address())
                    }
                }
            }
            println!("-- frame {n} @ {:.1} fps --", fps.fps());
        }

        n += 1;
        if let Some(max) = frames {
            if n >= max {
                break;
            }
        }
        // A real camera paces the loop itself (next_frame blocks at the
        // camera's rate); sleeping on top of that just subtracts from the
        // achievable FPS (16ms extra capped the loop at ~20 FPS on a 30 FPS
        // camera). Only the synthetic FakeCamera needs artificial pacing.
        if fake_paced {
            std::thread::sleep(Duration::from_millis(16));
        }
    }
    Ok(())
}

/// TUI: run the pipeline while drawing status / live values and handling keys.
fn run_tui(mut pipeline: Pipeline, cfg: &Config, calibrate_frames: u32) -> Result<()> {
    let mut state = UiState::new();
    state.set_osc_target(cfg.osc.host.clone(), cfg.osc.port);
    state.dry_run = cfg.osc.dry_run;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = tui_loop(&mut terminal, &mut state, &mut pipeline, calibrate_frames);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn tui_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    state: &mut UiState,
    pipeline: &mut Pipeline,
    calibrate_frames: u32,
) -> Result<()> {
    let mut fps = FpsCounter::new(30);
    let mut last = Instant::now();

    while !state.should_quit {
        if state.recalibrate_requested {
            state.recalibrate_requested = false;
            // Blocking is acceptable here: recalibration is a rare, brief,
            // user-initiated pause, not part of the steady-state frame loop.
            let _ = pipeline.calibrate_neutral(calibrate_frames);
            last = Instant::now();
            continue;
        }

        let outcome = pipeline.step()?;
        let now = Instant::now();
        fps.record(now.duration_since(last));
        last = now;

        state.set_capture_fps(fps.fps());
        state.set_tracker_fps(fps.fps());
        state.set_osc_send_rate(fps.fps());
        state.set_live_values(
            outcome
                .params
                .iter()
                .filter_map(|p| match p.value {
                    vrchat_camera_osc::osc::OscValue::Float(v) => Some((p.name.clone(), v)),
                    _ => None,
                })
                .collect(),
        );

        terminal.draw(|frame| render(frame, state))?;

        if event::poll(Duration::from_millis(5))? {
            if let Event::Key(key) = event::read()? {
                state.on_key(key.code);
            }
        }
    }
    Ok(())
}
