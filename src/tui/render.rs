//! Pure rendering of [`UiState`] with ratatui widgets.
//!
//! [`render`] performs no I/O, so it can be exercised with ratatui's
//! `TestBackend` for unit tests as well as driven by the real terminal loop.

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Tabs};
use ratatui::Frame;

use super::state::{ConfigField, Tab, UiState};

/// Draw the whole UI for the current [`UiState`].
pub fn render(frame: &mut Frame, state: &UiState) {
    let area = frame.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // status header
            Constraint::Length(3), // tab bar
            Constraint::Min(3),    // body
            Constraint::Length(1), // footer hint
        ])
        .split(area);

    render_header(frame, chunks[0], state);
    render_tabs(frame, chunks[1], state);
    render_body(frame, chunks[2], state);
    render_footer(frame, chunks[3]);

    if state.show_help {
        render_help(frame, area);
    }
}

/// Persistent status header, visible on every tab.
fn render_header(frame: &mut Frame, area: Rect, state: &UiState) {
    let mode = if state.dry_run { "DRY-RUN" } else { "LIVE" };
    let lines = vec![
        Line::from(format!(
            "OSC target {}   [{}]   send {:.1} msg/s",
            state.osc_target(),
            mode,
            state.osc_send_rate,
        )),
        Line::from(format!(
            "capture {:.1} FPS   tracker {:.1} FPS",
            state.capture_fps, state.tracker_fps,
        )),
    ];
    let block = Block::default()
        .borders(Borders::ALL)
        .title("VRChatCameraOSC");
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// Top tab bar.
fn render_tabs(frame: &mut Frame, area: Rect, state: &UiState) {
    let titles: Vec<Line> = Tab::ALL.iter().map(|t| Line::from(t.title())).collect();
    let tabs = Tabs::new(titles)
        .select(state.tab.index())
        .block(Block::default().borders(Borders::ALL))
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    frame.render_widget(tabs, area);
}

/// Dispatch the body to the active tab's renderer.
fn render_body(frame: &mut Frame, area: Rect, state: &UiState) {
    match state.tab {
        Tab::Status => render_status(frame, area, state),
        Tab::Values => render_values(frame, area, state),
        Tab::Config => render_config(frame, area, state),
    }
}

fn render_status(frame: &mut Frame, area: Rect, state: &UiState) {
    let mode = if state.dry_run { "DRY-RUN" } else { "LIVE" };
    let lines = vec![
        Line::from(format!("OSC target : {}", state.osc_target())),
        Line::from(format!("Mode       : {mode}")),
        Line::from(format!("Capture    : {:.1} FPS", state.capture_fps)),
        Line::from(format!("Tracker    : {:.1} FPS", state.tracker_fps)),
        Line::from(format!("OSC rate   : {:.1} msg/s", state.osc_send_rate)),
    ];
    let block = Block::default().borders(Borders::ALL).title("Status");
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_values(frame: &mut Frame, area: Rect, state: &UiState) {
    let items: Vec<ListItem> = if state.live_values.is_empty() {
        vec![ListItem::new("(no live values yet)")]
    } else {
        state
            .live_values
            .iter()
            .map(|(label, value)| ListItem::new(format!("{label:<20} {value:+.3}")))
            .collect()
    };
    let block = Block::default().borders(Borders::ALL).title("Live values");
    frame.render_widget(List::new(items).block(block), area);
}

fn render_config(frame: &mut Frame, area: Rect, state: &UiState) {
    let items: Vec<ListItem> = ConfigField::ALL
        .iter()
        .map(|field| {
            let value = match field {
                ConfigField::Host => state.host.clone(),
                ConfigField::Port => state.port.to_string(),
            };
            let marker = if *field == state.selected_field {
                "> "
            } else {
                "  "
            };
            let line = format!("{marker}{:<10}: {value}", field.label());
            let style = if *field == state.selected_field {
                Style::default().add_modifier(Modifier::REVERSED)
            } else {
                Style::default()
            };
            ListItem::new(Span::styled(line, style))
        })
        .collect();
    let block = Block::default().borders(Borders::ALL).title("Config");
    frame.render_widget(List::new(items).block(block), area);
}

fn render_footer(frame: &mut Frame, area: Rect) {
    let hint = Paragraph::new("Tab: switch  d: dry-run  c: recalibrate  ?: help  q: quit")
        .alignment(Alignment::Center);
    frame.render_widget(hint, area);
}

/// Centered help overlay listing the keybindings.
fn render_help(frame: &mut Frame, area: Rect) {
    let popup = centered_rect(60, 60, area);
    let lines = vec![
        Line::from("Help — keybindings"),
        Line::from(""),
        Line::from("Tab / Right     next panel"),
        Line::from("BackTab / Left  previous panel"),
        Line::from("Up / Down       move config cursor"),
        Line::from("d               toggle dry-run"),
        Line::from("c               recalibrate neutral pose"),
        Line::from("type            edit selected config field"),
        Line::from("Backspace       erase in field"),
        Line::from("?               toggle this help"),
        Line::from("q / Esc         quit"),
    ];
    let block = Block::default().borders(Borders::ALL).title("Help");
    frame.render_widget(Clear, popup);
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}

/// A `Rect` centered inside `area`, sized as a percentage of it.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}
