//! Render tests using ratatui's `TestBackend` (issue #6).
//!
//! Each test renders a state into an in-memory buffer and asserts the expected
//! text is present, which also exercises `render` for coverage.

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use vrchat_camera_osc::tui::state::Tab;
use vrchat_camera_osc::tui::{render, UiState};

/// Flatten the rendered buffer into a single string for substring assertions.
fn draw(state: &UiState, w: u16, h: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|frame| render(frame, state)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    buffer
        .content()
        .iter()
        .map(|cell| cell.symbol())
        .collect::<String>()
}

#[test]
fn header_shows_target_and_live_mode() {
    let mut s = UiState::new();
    s.set_osc_target("127.0.0.1", 9000);
    let out = draw(&s, 80, 24);
    assert!(out.contains("127.0.0.1:9000"), "target missing: {out}");
    assert!(out.contains("LIVE"), "mode missing: {out}");
    assert!(!out.contains("DRY-RUN"));
}

#[test]
fn header_shows_dry_run_when_enabled() {
    let mut s = UiState::new();
    s.dry_run = true;
    let out = draw(&s, 80, 24);
    assert!(out.contains("DRY-RUN"), "DRY-RUN missing: {out}");
}

#[test]
fn header_shows_fps_and_rate() {
    let mut s = UiState::new();
    s.set_capture_fps(30.0);
    s.set_tracker_fps(29.0);
    s.set_osc_send_rate(60.0);
    let out = draw(&s, 80, 24);
    assert!(out.contains("30.0"), "capture fps missing: {out}");
    assert!(out.contains("29.0"), "tracker fps missing: {out}");
    assert!(out.contains("60.0"), "send rate missing: {out}");
}

#[test]
fn tab_bar_lists_all_tabs() {
    let out = draw(&UiState::new(), 80, 24);
    assert!(out.contains("Status"));
    assert!(out.contains("Values"));
    assert!(out.contains("Config"));
}

#[test]
fn values_tab_shows_live_value_label() {
    let mut s = UiState::new();
    s.tab = Tab::Values;
    s.set_live_values(vec![("JawOpen".to_string(), 0.42)]);
    let out = draw(&s, 80, 24);
    assert!(out.contains("JawOpen"), "label missing: {out}");
}

#[test]
fn values_tab_empty_placeholder() {
    let mut s = UiState::new();
    s.tab = Tab::Values;
    let out = draw(&s, 80, 24);
    assert!(out.contains("no live values"), "placeholder missing: {out}");
}

#[test]
fn config_tab_shows_fields() {
    let mut s = UiState::new();
    s.tab = Tab::Config;
    let out = draw(&s, 80, 24);
    assert!(out.contains("Host"));
    assert!(out.contains("Port"));
    // Retired with the FAN backend (issue #21): no smoothing field/gauge.
    assert!(
        !out.contains("Smoothing"),
        "smoothing UI should be gone: {out}"
    );
}

#[test]
fn help_overlay_visible_only_when_enabled() {
    let mut s = UiState::new();
    let without = draw(&s, 80, 24);
    assert!(!without.contains("keybindings"));

    s.show_help = true;
    let with = draw(&s, 80, 24);
    assert!(with.contains("Help"), "help title missing: {with}");
    assert!(with.contains("keybindings"), "help body missing: {with}");
    assert!(with.contains("quit"), "help quit hint missing: {with}");
}

#[test]
fn status_tab_body_renders() {
    let mut s = UiState::new();
    s.tab = Tab::Status;
    let out = draw(&s, 80, 24);
    assert!(out.contains("OSC target"));
    assert!(out.contains("OSC rate"));
}

#[test]
fn renders_in_small_terminal_without_panic() {
    // Guard against layout math panicking on tiny sizes.
    let s = UiState::new();
    let _ = draw(&s, 20, 10);
}
