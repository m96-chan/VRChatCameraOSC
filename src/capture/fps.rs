//! Frame-rate measurement with an injectable time source.
//!
//! [`FpsCounter`] keeps a bounded ring of recent frame durations and reports the
//! rolling average FPS. The time source is injected (feed [`Duration`]s via
//! [`FpsCounter::record`], or [`Instant`]s via [`FpsCounter::tick`]) so the math
//! is unit-testable without sleeping.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Rolling FPS estimator over the last `capacity` inter-frame durations.
#[derive(Debug, Clone)]
pub struct FpsCounter {
    capacity: usize,
    durations: VecDeque<Duration>,
    last: Option<Instant>,
}

impl FpsCounter {
    /// Create a counter that averages over up to `capacity` recent frames.
    ///
    /// `capacity` is clamped to at least 1.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            durations: VecDeque::with_capacity(capacity.max(1)),
            last: None,
        }
    }

    /// Record one inter-frame duration directly (fully deterministic).
    pub fn record(&mut self, dt: Duration) {
        if self.durations.len() == self.capacity {
            self.durations.pop_front();
        }
        self.durations.push_back(dt);
    }

    /// Record a frame observed at `now`, deriving the duration from the previous
    /// [`tick`](Self::tick). The first tick only seeds the clock (no sample).
    pub fn tick(&mut self, now: Instant) {
        if let Some(prev) = self.last {
            // `saturating_duration_since` guards against a non-monotonic clock.
            self.record(now.saturating_duration_since(prev));
        }
        self.last = Some(now);
    }

    /// Current rolling FPS. Returns `0.0` when there are no samples yet, or when
    /// the accumulated duration is zero (avoids division by zero).
    pub fn fps(&self) -> f32 {
        if self.durations.is_empty() {
            return 0.0;
        }
        let total: Duration = self.durations.iter().sum();
        let secs = total.as_secs_f32();
        if secs <= 0.0 {
            return 0.0;
        }
        self.durations.len() as f32 / secs
    }

    /// Number of duration samples currently held.
    pub fn sample_count(&self) -> usize {
        self.durations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-3 * b.abs().max(1.0)
    }

    #[test]
    fn empty_counter_is_zero_fps() {
        let c = FpsCounter::new(30);
        assert_eq!(c.sample_count(), 0);
        assert_eq!(c.fps(), 0.0);
    }

    #[test]
    fn single_sample_is_reciprocal_of_duration() {
        let mut c = FpsCounter::new(30);
        c.record(Duration::from_millis(20)); // 20ms => 50 fps
        assert_eq!(c.sample_count(), 1);
        assert!(close(c.fps(), 50.0), "got {}", c.fps());
    }

    #[test]
    fn steady_60fps_reads_60() {
        let mut c = FpsCounter::new(10);
        for _ in 0..10 {
            c.record(Duration::from_nanos(16_666_667)); // ~60 fps
        }
        assert!(close(c.fps(), 60.0), "got {}", c.fps());
    }

    #[test]
    fn averages_mixed_durations() {
        let mut c = FpsCounter::new(4);
        // Two frames at 10ms, two at 30ms => total 80ms for 4 frames => 50 fps.
        c.record(Duration::from_millis(10));
        c.record(Duration::from_millis(10));
        c.record(Duration::from_millis(30));
        c.record(Duration::from_millis(30));
        assert!(close(c.fps(), 50.0), "got {}", c.fps());
    }

    #[test]
    fn ring_buffer_evicts_oldest() {
        let mut c = FpsCounter::new(2);
        c.record(Duration::from_millis(100)); // would drag fps down
        c.record(Duration::from_millis(10));
        c.record(Duration::from_millis(10));
        // Only the last two 10ms samples remain => 100 fps.
        assert_eq!(c.sample_count(), 2);
        assert!(close(c.fps(), 100.0), "got {}", c.fps());
    }

    #[test]
    fn zero_duration_samples_report_zero_fps() {
        let mut c = FpsCounter::new(4);
        c.record(Duration::ZERO);
        c.record(Duration::ZERO);
        assert_eq!(c.sample_count(), 2);
        assert_eq!(c.fps(), 0.0);
    }

    #[test]
    fn tick_first_call_only_seeds_no_sample() {
        let mut c = FpsCounter::new(8);
        let t0 = Instant::now();
        c.tick(t0);
        assert_eq!(c.sample_count(), 0);
        assert_eq!(c.fps(), 0.0);
    }

    #[test]
    fn tick_derives_durations_from_instants() {
        let mut c = FpsCounter::new(8);
        let t0 = Instant::now();
        c.tick(t0);
        c.tick(t0 + Duration::from_millis(25)); // 40 fps
        c.tick(t0 + Duration::from_millis(50)); // another 40 fps
        assert_eq!(c.sample_count(), 2);
        assert!(close(c.fps(), 40.0), "got {}", c.fps());
    }

    #[test]
    fn tick_handles_non_monotonic_clock() {
        let mut c = FpsCounter::new(8);
        let t0 = Instant::now();
        c.tick(t0 + Duration::from_millis(50));
        c.tick(t0); // clock went backwards -> saturates to zero duration
        assert_eq!(c.sample_count(), 1);
        assert_eq!(c.fps(), 0.0);
    }

    #[test]
    fn capacity_is_clamped_to_one() {
        let mut c = FpsCounter::new(0);
        c.record(Duration::from_millis(20));
        c.record(Duration::from_millis(10)); // evicts the first
        assert_eq!(c.sample_count(), 1);
        assert!(close(c.fps(), 100.0), "got {}", c.fps());
    }
}
