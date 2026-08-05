/*
 * Graph Orizon CLI Modules - Runtime - Tokens - Estimate
 * Heuristic token count from a character count.
 * Owns only the chars/4 approximation; callers sum the characters they want to
 * measure (via ChatMessage::chars) and divide here exactly once.
 */

// Estimates a token count as one quarter of the supplied Unicode character
// count. This is the common rough rule for English-like prompts and keeps the
// project dependency-free: no real tokenizer is loaded. Callers pass an already
// summed character total so the division happens once: dividing per message and
// summing the quotients would re-truncate each message and systematically
// undercount (a 3-char message would weigh 0 tokens instead of its fair share).
pub(crate) fn estimate_tokens(chars: usize) -> usize {
    chars / 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_chars_is_zero_tokens() {
        assert_eq!(estimate_tokens(0), 0);
    }

    #[test]
    fn divides_total_chars_by_four() {
        assert_eq!(estimate_tokens(9), 2);
    }
}
