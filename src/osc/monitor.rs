//! Dry-run OSC sink: records the human-readable line for every message that
//! would be sent, so mapping can be verified without VRChat (issue #3).

use super::{OscParam, OscSink, OscValue};

/// Render a parameter as a single monitor line: `"<address> <tag> <value>"`.
///
/// `tag` is `f` (float), `b` (bool), `i` (int) or `f4` (four floats,
/// space-separated) — see [`crate::osc`] docs.
fn format_line(param: &OscParam) -> String {
    let addr = param.address();
    match param.value {
        OscValue::Float(v) => format!("{addr} f {v}"),
        OscValue::Bool(v) => format!("{addr} b {v}"),
        OscValue::Int(v) => format!("{addr} i {v}"),
        OscValue::Floats4([a, b, c, d]) => format!("{addr} f4 {a} {b} {c} {d}"),
    }
}

/// A dry-run [`OscSink`] that captures formatted lines instead of sending UDP.
///
/// Every [`send`](OscSink::send) appends one line per parameter to an internal
/// buffer readable via [`lines`](MonitorSink::lines); when printing is enabled
/// the same line is also written to stdout. The TUI and tests read the buffer.
#[derive(Debug, Default)]
pub struct MonitorSink {
    lines: Vec<String>,
    print: bool,
}

impl MonitorSink {
    /// A silent monitor that only records lines (the default for tests/TUI).
    pub fn new() -> Self {
        Self::default()
    }

    /// A monitor that also prints each line to stdout as it is recorded.
    pub fn printing() -> Self {
        Self {
            lines: Vec::new(),
            print: true,
        }
    }

    /// The lines recorded so far, oldest first.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Drop all recorded lines (e.g. between TUI redraws).
    pub fn clear(&mut self) {
        self.lines.clear();
    }
}

impl OscSink for MonitorSink {
    fn send(&mut self, params: &[OscParam]) -> anyhow::Result<()> {
        for param in params {
            let line = format_line(param);
            if self.print {
                println!("{line}");
            }
            self.lines.push(line);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_each_value_type() {
        assert_eq!(
            format_line(&OscParam::float("MouthOpen", 0.42)),
            "/avatar/parameters/MouthOpen f 0.42"
        );
        assert_eq!(
            format_line(&OscParam::bool("Jump", true)),
            "/avatar/parameters/Jump b true"
        );
        assert_eq!(
            format_line(&OscParam::int("VRCEmote", 3)),
            "/avatar/parameters/VRCEmote i 3"
        );
        assert_eq!(
            format_line(&OscParam::raw_floats4(
                "/tracking/eye/LeftRightPitchYaw",
                [1.0, -2.0, 3.0, -4.0]
            )),
            "/tracking/eye/LeftRightPitchYaw f4 1 -2 3 -4"
        );
    }

    #[test]
    fn send_captures_lines_in_order() {
        let mut sink = MonitorSink::new();
        sink.send(&[
            OscParam::float("Smile", 1.0),
            OscParam::bool("Blink", false),
        ])
        .unwrap();
        sink.send(&[OscParam::int("VRCEmote", 7)]).unwrap();

        assert_eq!(
            sink.lines(),
            [
                "/avatar/parameters/Smile f 1",
                "/avatar/parameters/Blink b false",
                "/avatar/parameters/VRCEmote i 7",
            ]
        );
    }

    #[test]
    fn clear_empties_the_buffer() {
        let mut sink = MonitorSink::new();
        sink.send(&[OscParam::float("A", 0.1)]).unwrap();
        assert_eq!(sink.lines().len(), 1);
        sink.clear();
        assert!(sink.lines().is_empty());
    }

    #[test]
    fn printing_monitor_also_records() {
        let mut sink = MonitorSink::printing();
        sink.send(&[OscParam::float("A", 0.5)]).unwrap();
        assert_eq!(sink.lines(), ["/avatar/parameters/A f 0.5"]);
    }
}
