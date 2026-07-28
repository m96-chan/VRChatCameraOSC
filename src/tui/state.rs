//! Render-independent TUI state and the pure key reducer.
//!
//! Everything here is free of terminal I/O so it can be unit-tested directly.
//! The real terminal loop (see [`crate::tui::run`]) only calls these methods;
//! all decision logic lives in [`UiState::on_key`].

use crossterm::event::KeyCode;

/// Which panel is currently shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    /// Health/status: capture + tracker FPS, OSC target, send rate, dry-run.
    #[default]
    Status,
    /// Live tracked landmark / parameter values.
    Values,
    /// In-app configuration (OSC host/port, dry-run).
    Config,
}

impl Tab {
    /// All tabs in display order.
    pub const ALL: [Tab; 3] = [Tab::Status, Tab::Values, Tab::Config];

    /// Human-readable title used by the tab bar.
    pub fn title(self) -> &'static str {
        match self {
            Tab::Status => "Status",
            Tab::Values => "Values",
            Tab::Config => "Config",
        }
    }

    /// Zero-based index within [`Tab::ALL`].
    pub fn index(self) -> usize {
        match self {
            Tab::Status => 0,
            Tab::Values => 1,
            Tab::Config => 2,
        }
    }

    /// Next tab, wrapping around.
    pub fn next(self) -> Tab {
        Tab::ALL[(self.index() + 1) % Tab::ALL.len()]
    }

    /// Previous tab, wrapping around.
    pub fn prev(self) -> Tab {
        Tab::ALL[(self.index() + Tab::ALL.len() - 1) % Tab::ALL.len()]
    }
}

/// The editable field highlighted on the Config tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfigField {
    /// OSC destination host.
    #[default]
    Host,
    /// OSC destination UDP port.
    Port,
}

impl ConfigField {
    /// All fields in cursor order.
    pub const ALL: [ConfigField; 2] = [ConfigField::Host, ConfigField::Port];

    /// Label shown next to the field.
    pub fn label(self) -> &'static str {
        match self {
            ConfigField::Host => "Host",
            ConfigField::Port => "Port",
        }
    }

    fn index(self) -> usize {
        match self {
            ConfigField::Host => 0,
            ConfigField::Port => 1,
        }
    }

    fn next(self) -> ConfigField {
        ConfigField::ALL[(self.index() + 1) % ConfigField::ALL.len()]
    }

    fn prev(self) -> ConfigField {
        ConfigField::ALL[(self.index() + ConfigField::ALL.len() - 1) % ConfigField::ALL.len()]
    }
}

/// Render-independent UI state: status, live values, and config view.
///
/// The app loop feeds fresh telemetry via the `set_*` methods and forwards key
/// presses to [`UiState::on_key`]; [`crate::tui::render`] reads this state.
#[derive(Debug, Clone, PartialEq)]
pub struct UiState {
    /// Active panel.
    pub tab: Tab,
    /// Whether the help overlay is shown.
    pub show_help: bool,
    /// Set once the user asks to quit.
    pub should_quit: bool,

    // --- status / telemetry (pushed in by the app loop) ---
    /// Measured camera capture rate (frames/sec).
    pub capture_fps: f32,
    /// Measured tracker rate (frames/sec).
    pub tracker_fps: f32,
    /// Measured OSC send rate (messages/sec).
    pub osc_send_rate: f32,
    /// Snapshot of `(label, value)` for the live values panel.
    pub live_values: Vec<(String, f32)>,

    // --- config-edit fields ---
    /// OSC destination host being edited.
    pub host: String,
    /// OSC destination port being edited.
    pub port: u16,
    /// Dry-run: monitor only, do not actually send OSC.
    pub dry_run: bool,
    /// Config field under the edit cursor.
    pub selected_field: ConfigField,
    /// Set by `c`; the app loop performs the (blocking) recalibration, then
    /// clears this. Pure state only — no I/O happens in [`UiState`] itself.
    pub recalibrate_requested: bool,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            tab: Tab::default(),
            show_help: false,
            should_quit: false,
            capture_fps: 0.0,
            tracker_fps: 0.0,
            osc_send_rate: 0.0,
            live_values: Vec::new(),
            host: "127.0.0.1".to_string(),
            port: 9000,
            dry_run: false,
            selected_field: ConfigField::default(),
            recalibrate_requested: false,
        }
    }
}

impl UiState {
    /// A fresh state with sensible defaults.
    pub fn new() -> Self {
        Self::default()
    }

    // --- setters the app loop calls each frame ---

    /// Update the measured camera capture rate.
    pub fn set_capture_fps(&mut self, fps: f32) {
        self.capture_fps = fps;
    }

    /// Update the measured tracker rate.
    pub fn set_tracker_fps(&mut self, fps: f32) {
        self.tracker_fps = fps;
    }

    /// Update the measured OSC send rate.
    pub fn set_osc_send_rate(&mut self, rate: f32) {
        self.osc_send_rate = rate;
    }

    /// Replace the live values snapshot shown on the Values panel.
    pub fn set_live_values(&mut self, values: Vec<(String, f32)>) {
        self.live_values = values;
    }

    /// Set the OSC target host and port together.
    pub fn set_osc_target(&mut self, host: impl Into<String>, port: u16) {
        self.host = host.into();
        self.port = port;
    }

    /// The OSC destination formatted as `host:port`.
    pub fn osc_target(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    // --- pure reducer ---

    /// Apply a key press. Pure: mutates state only, never performs I/O.
    ///
    /// Keys:
    /// - `q` quit; `Esc` closes help if open else quits.
    /// - `?` toggles the help overlay.
    /// - `Tab`/`Right` next panel, `BackTab`/`Left` previous panel.
    /// - `Up`/`Down` move the Config field cursor.
    /// - `d` toggles dry-run.
    /// - `c` requests a neutral-pose recalibration (the app loop performs it).
    /// - On Config: type into the selected field (`Host` text, `Port`
    ///   digits), `Backspace` erases.
    pub fn on_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Esc => {
                if self.show_help {
                    self.show_help = false;
                } else {
                    self.should_quit = true;
                }
            }
            KeyCode::Char('?') => self.show_help = !self.show_help,
            KeyCode::Char('d') => self.dry_run = !self.dry_run,
            KeyCode::Char('c') => self.recalibrate_requested = true,
            KeyCode::Tab | KeyCode::Right => self.tab = self.tab.next(),
            KeyCode::BackTab | KeyCode::Left => self.tab = self.tab.prev(),
            KeyCode::Up => self.selected_field = self.selected_field.prev(),
            KeyCode::Down => self.selected_field = self.selected_field.next(),
            KeyCode::Backspace => self.backspace_field(),
            KeyCode::Char(c) => self.type_char(c),
            _ => {}
        }
    }

    /// Erase the last character of the selected editable field.
    fn backspace_field(&mut self) {
        match self.selected_field {
            ConfigField::Host => {
                self.host.pop();
            }
            ConfigField::Port => self.port /= 10,
        }
    }

    /// Route a typed character to the selected config field.
    ///
    /// Only meaningful on the Config tab; on other tabs typed characters are
    /// ignored so they cannot silently corrupt config.
    fn type_char(&mut self, c: char) {
        if self.tab != Tab::Config {
            return;
        }
        match self.selected_field {
            ConfigField::Host => {
                if !c.is_control() {
                    self.host.push(c);
                }
            }
            ConfigField::Port => {
                if let Some(digit) = c.to_digit(10) {
                    let next = (self.port as u32) * 10 + digit;
                    self.port = next.min(u16::MAX as u32) as u16;
                }
            }
        }
    }
}
