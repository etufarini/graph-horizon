/*
 * Graph Horizon CLI Modules - Runtime - Tokens - Prune
 * Keeps a request within a precomputed token budget by dropping the oldest
 * conversation turns. Owns the pruning threshold (the context window minus the
 * configured output-token reserve) and the per-turn trimming;
 * the request is assembled by the caller and the per-message estimate lives in
 * estimate.rs.
 */

use super::super::ChatMessage;
use super::estimate::estimate_tokens;

// Headroom kept on top of the profile reserve to absorb the imprecision of the
// chars/4 estimate, which can run *under* the real token count. Without it a
// prompt estimated right at the threshold could, once undercounting is added to
// the full reserved generation, push real usage past the context window.
const ESTIMATE_HEADROOM_PERCENT: usize = 10;

// Tokens of history the request may use before pruning kicks in: the context
// window minus the configured output-token reserve, minus a
// small headroom for estimate imprecision. Returns None — pruning disabled —
// when there is no usable context limit, or when the reserve (plus headroom)
// leaves no room for any history. A `Some` result is therefore always > 0, so an
// active threshold guarantees real space exists for the reserved reply.
pub(crate) fn pruning_threshold(context_limit: Option<usize>, reserved: usize) -> Option<usize> {
    let context_limit = context_limit.filter(|&l| l > 0)?;
    let usable = context_limit.checked_sub(reserved)?;
    Some(usable * (100 - ESTIMATE_HEADROOM_PERCENT) / 100).filter(|&t| t > 0)
}

// Trims `messages` so the estimated request fits within `threshold`, dropping the
// oldest user/assistant turn pairs first. The leading system message (if any) and
// the final user message (the new prompt) are never removed. A `None` threshold
// disables pruning entirely (see `pruning_threshold`) and returns the request
// unchanged.
pub(crate) fn prune_to_limit(
    mut messages: Vec<ChatMessage>,
    threshold: Option<usize>,
) -> Vec<ChatMessage> {
    let Some(threshold) = threshold else {
        return messages;
    };

    // The history lives between the protected head (a leading system message)
    // and the protected tail (the final new prompt). We only ever drop from
    // inside that region, two messages at a time (one user/assistant pair).
    let head = usize::from(messages.first().is_some_and(ChatMessage::is_system));
    // `head + 2 < len` guarantees a full history pair sits before the final
    // prompt, so the new prompt (and the system message, if any) stay untouched.
    while budget(&messages) > threshold && head + 2 < messages.len() {
        messages.drain(head..head + 2);
    }

    messages
}

// Estimated token cost of the whole request. Characters are summed across all
// messages and converted once, so the chars/4 estimate is not re-truncated per
// message (which would undercount and prune too late).
fn budget(messages: &[ChatMessage]) -> usize {
    estimate_tokens(messages.iter().map(ChatMessage::chars).sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Builds a message of exactly `tokens` estimated tokens (chars/4).
    fn user(tokens: usize) -> ChatMessage {
        ChatMessage::user("x".repeat(tokens * 4))
    }
    fn assistant(tokens: usize) -> ChatMessage {
        ChatMessage::assistant("x".repeat(tokens * 4))
    }
    fn system(tokens: usize) -> ChatMessage {
        ChatMessage::system("x".repeat(tokens * 4))
    }

    #[test]
    fn threshold_disabled_without_context_limit() {
        // No context window (unset or zero) means the threshold cannot be derived.
        assert_eq!(pruning_threshold(None, 100), None);
        assert_eq!(pruning_threshold(Some(0), 100), None);
    }

    #[test]
    fn threshold_disabled_when_reserve_meets_or_exceeds_limit() {
        // Equal leaves no room for history; smaller leaves none either. Both
        // disable pruning rather than producing a zero or underflowing threshold.
        assert_eq!(pruning_threshold(Some(100), 100), None);
        assert_eq!(pruning_threshold(Some(80), 100), None);
    }

    #[test]
    fn threshold_is_usable_space_minus_estimate_headroom() {
        // 100 - 20 reserve = 80 usable; minus 10% headroom for estimate
        // imprecision = 72.
        assert_eq!(pruning_threshold(Some(100), 20), Some(72));
    }

    #[test]
    fn disabled_threshold_keeps_everything() {
        let msgs = vec![user(10), assistant(10), user(10)];
        let out = prune_to_limit(msgs, None);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn under_threshold_keeps_everything() {
        // threshold 80; total 70 stays untouched.
        let msgs = vec![
            user(10),
            assistant(10),
            user(10),
            assistant(10),
            user(10),
            assistant(10),
            user(10),
        ];
        let out = prune_to_limit(msgs, Some(80));
        assert_eq!(out.len(), 7);
    }

    #[test]
    fn drops_oldest_pairs_until_within_threshold() {
        // threshold 80; total 90 (4 pairs + new). One drop -> 70.
        let msgs = vec![
            user(10),
            assistant(10),
            user(10),
            assistant(10),
            user(10),
            assistant(10),
            user(10),
            assistant(10),
            user(10),
        ];
        let out = prune_to_limit(msgs, Some(80));
        assert_eq!(out.len(), 7);
    }

    #[test]
    fn never_drops_system_or_new_prompt() {
        // threshold 16; system + new prompt alone already exceed it, yet neither
        // the system message nor the new prompt may be dropped.
        let msgs = vec![system(10), user(10)];
        let out = prune_to_limit(msgs, Some(16));
        assert_eq!(out.len(), 2);
        assert!(out[0].is_system());
    }
}
