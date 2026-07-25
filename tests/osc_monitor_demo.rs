//! Dry-run monitor demo (issue #3 acceptance criteria): a stub oscillator
//! drives the `MonitorSink` and produces well-formed `/avatar/parameters/...`
//! lines at a steady cadence. Run with output visible to eyeball the stream:
//!
//! ```text
//! cargo test --test osc_monitor_demo -- --nocapture
//! ```

use std::time::Duration;

use vrchat_camera_osc::osc::{MonitorSink, OscParam, OscSink, SendRate};

#[test]
fn oscillator_streams_well_formed_monitor_lines() {
    let mut sink = MonitorSink::new();
    let rate = SendRate::from_hz(60.0);
    assert_eq!(rate.interval(), Duration::from_secs_f64(1.0 / 60.0));

    // 10 frames of a stub oscillator -> MouthOpen in [0,1] plus a Blink toggle.
    for frame in 0..10 {
        let phase = frame as f32 / 10.0;
        let mouth = (phase * std::f32::consts::TAU).sin().abs();
        sink.send(&[
            OscParam::float("MouthOpen", mouth),
            OscParam::bool("Blink", frame % 2 == 0),
        ])
        .unwrap();
    }

    for line in sink.lines() {
        println!("{line}");
        assert!(line.starts_with("/avatar/parameters/"));
    }
    assert_eq!(sink.lines().len(), 20);
}
