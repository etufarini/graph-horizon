/*
 * GH Zero CLI Modules - Runtime - Tokens - Speed
 * Token throughput: tokens-per-second for the prefill and generation phases.
 * This file owns only the pure rate arithmetic and the carrier struct; the
 * clock (Instant capture) stays in the streaming loop, keeping timing out of
 * the token domain.
 */

use std::time::Duration;

// Tokens-per-second for prefill input and generated output. A 0.0 component
// means "not measured yet" and is hidden by the renderer.
#[derive(Default, Clone, Copy)]
pub(crate) struct Throughput {
    pub(crate) input: f32,
    pub(crate) output: f32,
}

// Tokens divided by elapsed seconds. A zero (or unmeasured) duration yields 0.0
// rather than infinity, so a cached/instant phase shows no speed instead of a
// nonsense spike; the renderer then hides that zero.
pub(crate) fn rate(tokens: usize, elapsed: Duration) -> f32 {
    let secs = elapsed.as_secs_f32();
    if secs <= 0.0 {
        return 0.0;
    }
    tokens as f32 / secs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_duration_is_zero_rate() {
        assert_eq!(rate(100, Duration::ZERO), 0.0);
    }

    #[test]
    fn divides_tokens_by_seconds() {
        assert_eq!(rate(120, Duration::from_secs(2)), 60.0);
    }
}
