/*
 * Graph Horizon CLI Modules - Plugins - Completion
 * Shared prefix-completion helpers used by every plugin that autocompletes from a
 * candidate list (the '@' attachment paths and the '/' slash commands). It owns
 * one idea: given what the user typed and the candidates that match it, return the
 * tail that extends the input to the candidates' longest common prefix.
 */

// The tail to append to `typed` so it reaches the longest common prefix of the
// candidates, or None when there is nothing left to add (no candidates, or the
// common prefix is no longer than what is already typed). `typed` must be a
// prefix of every candidate, which is how both callers build the list.
pub(crate) fn completion_tail(typed: &str, candidates: &[String]) -> Option<String> {
    let mut common = candidates.first()?.clone();
    for candidate in candidates.iter().skip(1) {
        common.truncate(shared_prefix_len(&common, candidate));
    }
    (common.len() > typed.len()).then(|| common[typed.len()..].to_string())
}

// Byte length of the shared leading prefix of two strings, on char boundaries.
pub(crate) fn shared_prefix_len(left: &str, right: &str) -> usize {
    left.chars()
        .zip(right.chars())
        .take_while(|(a, b)| a == b)
        .map(|(c, _)| c.len_utf8())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_tail_up_to_common_prefix() {
        let candidates = vec!["export".to_string(), "expand".to_string()];
        // Common prefix "exp"; already typed "ex" -> tail "p".
        assert_eq!(completion_tail("ex", &candidates).as_deref(), Some("p"));
        // A single candidate completes fully.
        assert_eq!(
            completion_tail("ex", &["export".to_string()]).as_deref(),
            Some("port")
        );
        // Nothing to add once the input already is the common prefix.
        assert_eq!(completion_tail("exp", &candidates), None);
        // No candidates, nothing to complete.
        assert_eq!(completion_tail("ex", &[]), None);
    }

    #[test]
    fn shared_prefix_respects_char_boundaries() {
        assert_eq!(shared_prefix_len("caffè", "caffè latte"), "caffè".len());
        assert_eq!(shared_prefix_len("àbc", "àxy"), "à".len());
        assert_eq!(shared_prefix_len("abc", "xyz"), 0);
    }
}
