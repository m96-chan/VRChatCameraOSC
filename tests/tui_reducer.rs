//! Unit tests for the render-independent TUI state + reducer (issue #6).

use crossterm::event::KeyCode;
use vrchat_camera_osc::tui::state::{ConfigField, Tab};
use vrchat_camera_osc::tui::UiState;

#[test]
fn defaults_are_sensible() {
    let s = UiState::new();
    assert_eq!(s.tab, Tab::Status);
    assert!(!s.show_help);
    assert!(!s.should_quit);
    assert!(!s.dry_run);
    assert_eq!(s.host, "127.0.0.1");
    assert_eq!(s.port, 9000);
    assert_eq!(s.osc_target(), "127.0.0.1:9000");
    assert_eq!(s.selected_field, ConfigField::Host);
    assert!(s.live_values.is_empty());
}

#[test]
fn q_quits() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Char('q'));
    assert!(s.should_quit);
}

#[test]
fn esc_quits_when_help_closed() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Esc);
    assert!(s.should_quit);
}

#[test]
fn esc_closes_help_first_then_quits() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Char('?'));
    assert!(s.show_help);
    s.on_key(KeyCode::Esc);
    assert!(!s.show_help, "Esc should close help first");
    assert!(
        !s.should_quit,
        "Esc should not also quit while help was open"
    );
    s.on_key(KeyCode::Esc);
    assert!(s.should_quit, "second Esc quits");
}

#[test]
fn question_toggles_help() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Char('?'));
    assert!(s.show_help);
    s.on_key(KeyCode::Char('?'));
    assert!(!s.show_help);
}

#[test]
fn d_toggles_dry_run() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Char('d'));
    assert!(s.dry_run);
    s.on_key(KeyCode::Char('d'));
    assert!(!s.dry_run);
}

#[test]
fn tab_cycles_forward_and_wraps() {
    let mut s = UiState::new();
    assert_eq!(s.tab, Tab::Status);
    s.on_key(KeyCode::Tab);
    assert_eq!(s.tab, Tab::Values);
    s.on_key(KeyCode::Tab);
    assert_eq!(s.tab, Tab::Config);
    s.on_key(KeyCode::Tab);
    assert_eq!(s.tab, Tab::Status, "wraps around");
}

#[test]
fn backtab_and_arrows_switch_tabs() {
    let mut s = UiState::new();
    s.on_key(KeyCode::BackTab);
    assert_eq!(s.tab, Tab::Config, "prev from Status wraps to Config");
    s.on_key(KeyCode::Right);
    assert_eq!(s.tab, Tab::Status);
    s.on_key(KeyCode::Left);
    assert_eq!(s.tab, Tab::Config);
}

#[test]
fn up_down_move_config_cursor() {
    let mut s = UiState::new();
    assert_eq!(s.selected_field, ConfigField::Host);
    s.on_key(KeyCode::Down);
    assert_eq!(s.selected_field, ConfigField::Port);
    s.on_key(KeyCode::Down);
    assert_eq!(s.selected_field, ConfigField::Smoothing);
    s.on_key(KeyCode::Down);
    assert_eq!(s.selected_field, ConfigField::Host, "wraps");
    s.on_key(KeyCode::Up);
    assert_eq!(s.selected_field, ConfigField::Smoothing, "wraps back");
}

#[test]
fn editing_host_appends_and_backspaces() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Tab); // Values
    s.on_key(KeyCode::Tab); // Config
    assert_eq!(s.tab, Tab::Config);
    assert_eq!(s.selected_field, ConfigField::Host);
    // clear the default host
    for _ in 0.."127.0.0.1".len() {
        s.on_key(KeyCode::Backspace);
    }
    assert_eq!(s.host, "");
    for c in "10.0.0.5".chars() {
        s.on_key(KeyCode::Char(c));
    }
    assert_eq!(s.host, "10.0.0.5");
    s.on_key(KeyCode::Backspace);
    assert_eq!(s.host, "10.0.0.");
}

#[test]
fn editing_port_accepts_digits_only() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Right); // Values
    s.on_key(KeyCode::Right); // Config
    s.on_key(KeyCode::Down); // Port
    assert_eq!(s.selected_field, ConfigField::Port);
    // rewind default port to zero
    for _ in 0..6 {
        s.on_key(KeyCode::Backspace);
    }
    assert_eq!(s.port, 0);
    s.on_key(KeyCode::Char('9'));
    s.on_key(KeyCode::Char('0'));
    s.on_key(KeyCode::Char('0'));
    s.on_key(KeyCode::Char('0'));
    assert_eq!(s.port, 9000);
    // non-digit ignored
    s.on_key(KeyCode::Char('x'));
    assert_eq!(s.port, 9000);
}

#[test]
fn port_saturates_at_u16_max() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Right);
    s.on_key(KeyCode::Right);
    s.on_key(KeyCode::Down); // Port
    for _ in 0..12 {
        s.on_key(KeyCode::Char('9'));
    }
    assert_eq!(s.port, u16::MAX);
}

#[test]
fn smoothing_edited_with_plus_minus_and_clamped() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Right);
    s.on_key(KeyCode::Right); // Config
    s.on_key(KeyCode::Down);
    s.on_key(KeyCode::Down); // Smoothing
    assert_eq!(s.selected_field, ConfigField::Smoothing);
    s.set_smoothing(0.5);
    // a non +/- char while Smoothing is selected is ignored
    s.on_key(KeyCode::Char('a'));
    assert!((s.smoothing - 0.5).abs() < 1e-6);
    s.on_key(KeyCode::Char('+'));
    assert!((s.smoothing - 0.55).abs() < 1e-6);
    s.on_key(KeyCode::Char('-'));
    s.on_key(KeyCode::Char('-'));
    assert!((s.smoothing - 0.45).abs() < 1e-6);
    // clamp high
    for _ in 0..50 {
        s.on_key(KeyCode::Char('+'));
    }
    assert!((s.smoothing - 1.0).abs() < 1e-6);
    // clamp low
    for _ in 0..50 {
        s.on_key(KeyCode::Char('-'));
    }
    assert!((s.smoothing - 0.0).abs() < 1e-6);
}

#[test]
fn typing_ignored_off_config_tab() {
    let mut s = UiState::new();
    assert_eq!(s.tab, Tab::Status);
    s.on_key(KeyCode::Char('z'));
    assert_eq!(s.host, "127.0.0.1", "host unchanged when not on Config tab");
}

#[test]
fn backspace_on_smoothing_is_noop() {
    let mut s = UiState::new();
    s.on_key(KeyCode::Right);
    s.on_key(KeyCode::Right);
    s.on_key(KeyCode::Down);
    s.on_key(KeyCode::Down); // Smoothing
    let before = s.smoothing;
    s.on_key(KeyCode::Backspace);
    assert_eq!(s.smoothing, before);
}

#[test]
fn setters_update_state() {
    let mut s = UiState::new();
    s.set_capture_fps(30.0);
    s.set_tracker_fps(28.5);
    s.set_osc_send_rate(60.0);
    s.set_osc_target("192.168.1.10", 9001);
    s.set_live_values(vec![("JawOpen".to_string(), 0.42)]);
    s.set_smoothing(2.0); // clamps

    assert_eq!(s.capture_fps, 30.0);
    assert_eq!(s.tracker_fps, 28.5);
    assert_eq!(s.osc_send_rate, 60.0);
    assert_eq!(s.osc_target(), "192.168.1.10:9001");
    assert_eq!(s.live_values, vec![("JawOpen".to_string(), 0.42)]);
    assert_eq!(s.smoothing, 1.0);
}

#[test]
fn unhandled_key_is_noop() {
    let mut s = UiState::new();
    let before = s.clone();
    s.on_key(KeyCode::Home);
    assert_eq!(s, before);
}
