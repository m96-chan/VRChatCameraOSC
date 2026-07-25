//! Visual demo of the TUI render output (issue #6).
//!
//! Not an assertion-heavy test — it drives the real `render` into an in-memory
//! `TestBackend` and prints the resulting frames so the TUI can be eyeballed
//! without a terminal:
//!
//! ```sh
//! cargo test --test tui_demo -- --nocapture
//! ```

use crossterm::event::KeyCode;
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use vrchat_camera_osc::tui::{render, UiState};

fn print_frame(title: &str, state: &UiState, w: u16, h: u16) {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
    terminal.draw(|frame| render(frame, state)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    println!("\n===== {title} =====");
    for y in 0..h {
        let mut row = String::new();
        for x in 0..w {
            row.push_str(buffer[(x, y)].symbol());
        }
        println!("{}", row.trim_end());
    }
}

#[test]
fn demo_renders_every_view() {
    let mut state = UiState::new();
    state.set_capture_fps(30.0);
    state.set_tracker_fps(29.4);
    state.set_osc_send_rate(60.0);
    state.set_live_values(vec![
        ("JawOpen".to_string(), 0.42),
        ("BrowUp".to_string(), -0.13),
        ("HeadYaw".to_string(), 0.07),
    ]);

    print_frame("Status tab (LIVE)", &state, 80, 20);

    state.on_key(KeyCode::Char('d')); // toggle dry-run
    print_frame("Status tab (DRY-RUN)", &state, 80, 20);

    state.on_key(KeyCode::Char('d')); // back to live
    state.on_key(KeyCode::Tab); // Values
    print_frame("Values tab", &state, 80, 20);

    state.on_key(KeyCode::Tab); // Config
    state.on_key(KeyCode::Down); // Port selected
    print_frame("Config tab (Port selected)", &state, 80, 20);

    state.on_key(KeyCode::Char('?')); // help overlay
    print_frame("Help overlay", &state, 80, 20);
}
