//! Integration tests for the public capture surface: the `CameraSource`
//! contract driven through the synthetic `FakeCamera`, plus the pure conversion
//! and FPS helpers. No webcam required.

use std::time::Duration;

use vrchat_camera_osc::capture::{
    rgb8_to_frame, rgba8_to_frame, CameraSource, FakeCamera, FpsCounter, Frame, Pattern,
};

#[test]
fn fake_camera_drives_the_camera_source_contract() {
    let mut cam = FakeCamera::new(32, 24);
    assert_eq!(cam.resolution(), (32, 24));

    let (w, h) = cam.resolution();
    let frame = cam
        .next_frame()
        .expect("fake camera should always produce a frame");
    assert_eq!((frame.width, frame.height), (w, h));
    assert_eq!(frame.data.len(), Frame::expected_len(w, h));
}

#[test]
fn fps_counter_over_synthetic_capture_loop() {
    let mut cam = FakeCamera::with_pattern(16, 16, Pattern::Solid);
    let mut fps = FpsCounter::new(30);

    // Simulate a steady 60 fps loop by feeding fixed durations between grabs.
    for _ in 0..30 {
        let _ = cam.next_frame().unwrap();
        fps.record(Duration::from_nanos(16_666_667));
    }

    assert!((fps.fps() - 60.0).abs() < 0.1, "fps = {}", fps.fps());
}

#[test]
fn conversion_helpers_are_reachable_and_validate() {
    // Packed RGB passes straight through.
    let rgb = rgb8_to_frame(1, 1, &[9, 8, 7]).unwrap();
    assert_eq!(rgb.data, vec![9, 8, 7]);

    // RGBA drops alpha.
    let rgba = rgba8_to_frame(1, 1, &[9, 8, 7, 255]).unwrap();
    assert_eq!(rgba.data, vec![9, 8, 7]);

    // Wrong sizes are rejected, not panics.
    assert!(rgb8_to_frame(2, 2, &[0; 3]).is_err());
    assert!(rgba8_to_frame(2, 2, &[0; 3]).is_err());
}
