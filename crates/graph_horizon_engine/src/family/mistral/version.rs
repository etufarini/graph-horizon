/*
 * graph_horizon_engine — Ministral 3 Instruct/Reasoning 2512 version contract
 * Owns release-pinned context policy, Reasoning system policy, and reference
 * dimensions. It performs no parsing, dispatch, I/O, or allocation.
 */

pub(super) const REASONING_SYSTEM_PROMPT: &str = concat!(
    "# HOW YOU SHOULD THINK AND ANSWER",
    "\n\nFirst draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.",
    "\n\nYour thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response."
);

#[cfg(test)]
pub(super) const RELEASE: &str = "Ministral 3 Instruct and Reasoning 2512";
#[cfg(test)]
pub(super) const MAX_CONTEXT: usize = 262_144;
pub(super) const DEFAULT_CONTEXT: usize = 32_768;
#[cfg(test)]
pub(super) const HEAD_COUNT: usize = 32;
#[cfg(test)]
pub(super) const KV_HEAD_COUNT: usize = 8;
#[cfg(test)]
pub(super) const KEY_LENGTH: usize = 128;
#[cfg(test)]
pub(super) const VALUE_LENGTH: usize = 128;
#[cfg(test)]
pub(super) const ROPE_DIMENSION: usize = 128;
#[cfg(test)]
pub(super) const Q_WIDTH: usize = 4_096;
#[cfg(test)]
pub(super) const K_WIDTH: usize = 1_024;
#[cfg(test)]
pub(super) const V_WIDTH: usize = 1_024;
#[cfg(test)]
pub(super) const ATTENTION_WIDTH: usize = 4_096;

#[cfg(test)]
pub(super) type ReferenceRow = (usize, usize, usize, bool);
#[cfg(test)]
pub(super) const REFERENCE_ROWS: [ReferenceRow; 3] = [
    (26, 3_072, 9_216, false),
    (34, 4_096, 14_336, true),
    (40, 5_120, 16_384, true),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versioned_contract_has_the_approved_release_rows() {
        assert_eq!(RELEASE, "Ministral 3 Instruct and Reasoning 2512");
        assert_eq!(MAX_CONTEXT, 262_144);
        const { assert!(DEFAULT_CONTEXT <= MAX_CONTEXT) };
        assert_eq!(
            REFERENCE_ROWS,
            [
                (26, 3_072, 9_216, false),
                (34, 4_096, 14_336, true),
                (40, 5_120, 16_384, true),
            ]
        );
        assert_eq!(Q_WIDTH, HEAD_COUNT * KEY_LENGTH);
        assert_eq!(K_WIDTH, KV_HEAD_COUNT * KEY_LENGTH);
        assert_eq!(V_WIDTH, KV_HEAD_COUNT * VALUE_LENGTH);
        assert_eq!(ATTENTION_WIDTH, HEAD_COUNT * VALUE_LENGTH);
        assert_eq!(ROPE_DIMENSION, KEY_LENGTH);
    }

    #[test]
    fn versioned_contract_pins_the_reasoning_system_prompt() {
        assert_eq!(
            REASONING_SYSTEM_PROMPT,
            "# HOW YOU SHOULD THINK AND ANSWER\n\nFirst draft your thinking process (inner monologue) until you arrive at a response. Format your response using Markdown, and use LaTeX for any mathematical equations. Write both your thoughts and the response in the same language as the input.\n\nYour thinking process must follow the template below:[THINK]Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate the response to the user.[/THINK]Here, provide a self-contained response."
        );
        assert_eq!(REASONING_SYSTEM_PROMPT.matches("[THINK]").count(), 1);
        assert_eq!(REASONING_SYSTEM_PROMPT.matches("[/THINK]").count(), 1);
        assert!(!REASONING_SYSTEM_PROMPT.ends_with('\n'));
    }
}
