//! Manual demo of the real native backend (AVFoundation on macOS, V4L2 on
//! Linux, MediaFoundation on Windows). Ignored by default (it needs a webcam
//! + permission and cannot run in CI). Run explicitly:
//!
//! ```text
//! cargo test --test capture_native_demo -- --ignored --nocapture
//! ```
//!
//! It exercises device enumeration and, if a camera is present, grabs a single
//! frame and prints its resolution. Any failure (no camera / permission denied)
//! is printed as a clear message rather than panicking.

#![cfg(all(
    any(target_os = "macos", target_os = "linux", target_os = "windows"),
    feature = "camera"
))]

use vrchat_camera_osc::capture::native::NativeCamera;
use vrchat_camera_osc::capture::CameraSource;

#[test]
#[ignore = "requires a real webcam and camera permission"]
fn demo_enumerate_and_grab() {
    match NativeCamera::list_devices() {
        Ok(devices) => {
            println!("native camera devices ({}):", devices.len());
            for d in &devices {
                println!("  [{}] {}", d.index, d.name);
            }
            // Try each device until one opens and yields a frame; report the
            // clear error for any that fail (no panics).
            for d in &devices {
                match NativeCamera::open(d.index, 640, 480) {
                    Ok(mut cam) => {
                        let (w, h) = cam.resolution();
                        println!("opened camera {} ({}) at {w}x{h}", d.index, d.name);
                        match cam.next_frame() {
                            Ok(frame) => {
                                println!(
                                    "grabbed frame: {}x{} ({} bytes)",
                                    frame.width,
                                    frame.height,
                                    frame.data.len()
                                );
                                break;
                            }
                            Err(e) => println!("  frame grab failed (clear error): {e:#}"),
                        }
                    }
                    Err(e) => println!("  open {} failed (clear error): {e:#}", d.index),
                }
            }
        }
        Err(e) => println!("enumeration failed (clear error): {e:#}"),
    }
}
