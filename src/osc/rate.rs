//! Send-cadence helper for the realtime loop: caps how often OSC is emitted so
//! the tracker does not flood VRChat (issue #3).

use std::time::{Duration, Instant};

/// Enforces a minimum interval between sends.
///
/// Call [`should_send`](SendRate::should_send) each loop iteration; it returns
/// `true` at most once per configured interval, `false` in between. The first
/// call always returns `true`.
#[derive(Debug, Clone)]
pub struct SendRate {
    interval: Duration,
    last: Option<Instant>,
}

impl SendRate {
    /// Build from a target frequency in Hz (e.g. `60.0`). A non-positive or
    /// non-finite `hz` yields a zero interval (send every call).
    pub fn from_hz(hz: f64) -> Self {
        let interval = if hz.is_finite() && hz > 0.0 {
            Duration::from_secs_f64(1.0 / hz)
        } else {
            Duration::ZERO
        };
        Self::from_interval(interval)
    }

    /// Build from an explicit minimum interval between sends.
    pub fn from_interval(interval: Duration) -> Self {
        Self {
            interval,
            last: None,
        }
    }

    /// The configured minimum interval between sends.
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Whether a send is due at `now`, updating internal state when it returns
    /// `true`. Deterministic and clock-injectable for tests.
    pub fn should_send_at(&mut self, now: Instant) -> bool {
        match self.last {
            Some(last) if now.duration_since(last) < self.interval => false,
            _ => {
                self.last = Some(now);
                true
            }
        }
    }

    /// Convenience wrapper over [`should_send_at`](SendRate::should_send_at)
    /// using the current instant.
    pub fn should_send(&mut self) -> bool {
        self.should_send_at(Instant::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_hz_computes_interval() {
        assert_eq!(
            SendRate::from_hz(100.0).interval(),
            Duration::from_millis(10)
        );
    }

    #[test]
    fn non_positive_or_nan_hz_is_zero_interval_and_always_sends() {
        for hz in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            let mut rate = SendRate::from_hz(hz);
            assert_eq!(rate.interval(), Duration::ZERO);
            let now = Instant::now();
            assert!(rate.should_send_at(now));
            assert!(rate.should_send_at(now));
        }
    }

    #[test]
    fn throttles_until_interval_elapses() {
        let mut rate = SendRate::from_interval(Duration::from_millis(10));
        let t0 = Instant::now();

        assert!(rate.should_send_at(t0), "first call always sends");
        assert!(
            !rate.should_send_at(t0 + Duration::from_millis(5)),
            "too soon"
        );
        assert!(
            rate.should_send_at(t0 + Duration::from_millis(10)),
            "interval elapsed"
        );
        assert!(
            !rate.should_send_at(t0 + Duration::from_millis(12)),
            "too soon again"
        );
        assert!(rate.should_send_at(t0 + Duration::from_millis(20)));
    }

    #[test]
    fn should_send_uses_real_clock() {
        let mut rate = SendRate::from_hz(1.0);
        assert!(rate.should_send());
        assert!(!rate.should_send());
    }
}
