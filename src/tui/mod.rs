//! Terminal UI — the primary control surface. Implemented by issue #6.
//!
//! State/reducer logic is kept render-independent so it can be unit-tested
//! without a real terminal.

/// Render-independent UI state (status, live values, config view).
#[derive(Debug, Default)]
pub struct UiState {
    // fields added in issue #6
}

impl UiState {
    pub fn new() -> Self {
        Self::default()
    }
}
