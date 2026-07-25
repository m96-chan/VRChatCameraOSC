//! Terminal UI — the primary control surface (issue #6).
//!
//! The module is split so the interesting logic stays testable without a real
//! terminal:
//!
//! - [`state`] — render-independent [`UiState`] plus the pure [`UiState::on_key`]
//!   reducer. Unit-tested directly.
//! - [`render`] — pure drawing of that state with ratatui widgets. Exercised
//!   with ratatui's `TestBackend`.
//! - [`run`] — the thin loop that owns the real crossterm terminal. It performs
//!   I/O and is intentionally minimal; the app-loop wiring lands with issue #8.

pub mod render;
pub mod state;

pub use render::render;
pub use state::{ConfigField, Tab, UiState};

use std::io::{self, Stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

/// Run the TUI against the real terminal until the user quits.
///
/// This owns terminal setup/teardown (raw mode + alternate screen) and the
/// event loop. It is deliberately thin: all decisions live in the tested
/// [`UiState::on_key`] / [`render`] surface. The realtime capture→track→OSC
/// wiring that feeds telemetry into the state arrives with issue #8.
pub fn run() -> Result<()> {
    let mut state = UiState::new();

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let result = event_loop(&mut terminal, &mut state);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

/// The draw/poll loop, split out from terminal setup for clarity.
fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    state: &mut UiState,
) -> Result<()> {
    while !state.should_quit {
        terminal.draw(|frame| render(frame, state))?;
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                state.on_key(key.code);
            }
        }
    }
    Ok(())
}
