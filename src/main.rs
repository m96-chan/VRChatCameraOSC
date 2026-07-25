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
//! fallback when no webcam is present). Tracker: loaded from
//! `models/2dfan4.safetensors` when present; without it the loop runs but
//! emits no parameters (a clear warning is shown).

use std::io::{self, Stdout};
use std::path::{Path, PathBuf};
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
use vrchat_camera_osc::mapping::Mapper;
use vrchat_camera_osc::osc::{MonitorSink, OscSink, UdpOscSender};
use vrchat_camera_osc::pipeline::Pipeline;
use vrchat_camera_osc::tracking::detect::sfd::SfdDetector;
use vrchat_camera_osc::tracking::fan::FanTracker;
use vrchat_camera_osc::tracking::FaceTracker;
use vrchat_camera_osc::tui::{render, UiState};

/// Default FAN weights path — also the only path auto-downloaded on first
/// run (issue #12); a custom `--weights` path is left for the user to manage.
fn default_weights_path() -> PathBuf {
    PathBuf::from("models/2dfan4.safetensors")
}

/// Default S3FD detector path — see [`default_weights_path`].
fn default_detector_path() -> PathBuf {
    PathBuf::from("models/s3fd.safetensors")
}

const FAN_MODEL_URL: &str =
    "https://github.com/m96-chan/VRChatCameraOSC/releases/download/models-v1/2dfan4.safetensors";
const SFD_MODEL_URL: &str =
    "https://github.com/m96-chan/VRChatCameraOSC/releases/download/models-v1/s3fd.safetensors";

/// Default detector re-run interval (issue #13): the S3FD detector is the
/// dominant per-frame cost, and a face doesn't move far in a handful of
/// frames, so it's only re-run this often; landmarks-derived crops carry
/// tracking through the frames in between.
const DEFAULT_DETECT_INTERVAL: u32 = 8;

struct Args {
    monitor: bool,
    fake: bool,
    frames: Option<u64>,
    weights: PathBuf,
    detector: PathBuf,
    detect_interval: u32,
}

fn parse_args() -> Args {
    let mut a = Args {
        monitor: false,
        fake: false,
        frames: None,
        weights: default_weights_path(),
        detector: default_detector_path(),
        detect_interval: DEFAULT_DETECT_INTERVAL,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--monitor" => a.monitor = true,
            "--fake" => a.fake = true,
            "--frames" => a.frames = it.next().and_then(|v| v.parse().ok()),
            "--weights" => {
                if let Some(p) = it.next() {
                    a.weights = PathBuf::from(p);
                }
            }
            "--detector" => {
                if let Some(p) = it.next() {
                    a.detector = PathBuf::from(p);
                }
            }
            "--detect-interval" => {
                if let Some(n) = it.next().and_then(|v| v.parse().ok()) {
                    a.detect_interval = n;
                }
            }
            "-h" | "--help" => {
                println!(
                    "vrchat-camera-osc [--monitor] [--fake] [--frames N] \
                     [--weights FAN.safetensors] [--detector S3FD.safetensors] \
                     [--detect-interval N]"
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

#[cfg(all(any(target_os = "macos", target_os = "linux"), feature = "camera"))]
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

#[cfg(not(all(any(target_os = "macos", target_os = "linux"), feature = "camera")))]
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

/// Download `path` from `url` on first run, but only when `path` is still the
/// built-in default — a custom `--weights`/`--detector` path is the user's to
/// manage. Never fails the caller: a download error just logs and leaves
/// `path` missing, which the existing "tracking disabled" fallback handles.
fn ensure_default_model_present(path: &Path, default: &Path, url: &str) {
    if path != default || path.exists() {
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

fn build_tracker(
    weights: &PathBuf,
    detector_path: &PathBuf,
    detect_interval: u32,
) -> Option<Box<dyn FaceTracker>> {
    ensure_default_model_present(weights, &default_weights_path(), FAN_MODEL_URL);
    ensure_default_model_present(detector_path, &default_detector_path(), SFD_MODEL_URL);

    if !weights.exists() {
        eprintln!(
            "no FAN weights at {} — tracking disabled (run reference/gen_fixtures.py). \
             The loop still runs; it just won't emit parameters.",
            weights.display()
        );
        return None;
    }
    let mut tracker = match FanTracker::from_safetensors(weights) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("failed to load FAN weights {}: {e}", weights.display());
            return None;
        }
    };

    // Attach the S3FD detector when its weights are present, so the face is
    // auto-cropped; otherwise fall back to whole-frame tracking.
    if detector_path.exists() {
        match SfdDetector::from_safetensors(detector_path) {
            Ok(det) => {
                tracker = tracker
                    .with_detector(Box::new(det))
                    .with_detect_interval(detect_interval);
            }
            Err(e) => eprintln!(
                "failed to load detector {}: {e} — using whole-frame tracking",
                detector_path.display()
            ),
        }
    } else {
        eprintln!(
            "no detector weights at {} — using whole-frame tracking (see issue #9)",
            detector_path.display()
        );
    }
    Some(Box::new(tracker))
}

fn main() -> Result<()> {
    let args = parse_args();

    let cfg = Config::load_or_create(Config::default_path()).unwrap_or_else(|e| {
        eprintln!("using defaults (config load failed: {e})");
        Config::default()
    });
    cfg.validate()?;

    let camera = build_camera(&cfg, args.fake);
    let tracker = build_tracker(&args.weights, &args.detector, args.detect_interval);
    let mapper = Mapper::with_smoothing(cfg.tracking.smoothing);
    let sink = build_sink(&cfg)?;
    let pipeline = Pipeline::new(camera, tracker, mapper, sink);

    if args.monitor {
        run_monitor(pipeline, &cfg, args.frames)
    } else {
        run_tui(pipeline, &cfg)
    }
}

/// Headless: step the pipeline and print the OSC parameters each frame.
fn run_monitor(mut pipeline: Pipeline, cfg: &Config, frames: Option<u64>) -> Result<()> {
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
                if let vrchat_camera_osc::osc::OscValue::Float(v) = p.value {
                    println!("{} f {v:.4}", p.address());
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
        std::thread::sleep(Duration::from_millis(16));
    }
    Ok(())
}

/// TUI: run the pipeline while drawing status / live values and handling keys.
fn run_tui(mut pipeline: Pipeline, cfg: &Config) -> Result<()> {
    let mut state = UiState::new();
    state.set_osc_target(cfg.osc.host.clone(), cfg.osc.port);
    state.set_smoothing(cfg.tracking.smoothing);
    state.dry_run = cfg.osc.dry_run;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = tui_loop(&mut terminal, &mut state, &mut pipeline);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn tui_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    state: &mut UiState,
    pipeline: &mut Pipeline,
) -> Result<()> {
    let mut fps = FpsCounter::new(30);
    let mut last = Instant::now();

    while !state.should_quit {
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
