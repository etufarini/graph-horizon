/*
 * Graph Horizon CLI Modules - Console - Session - Request
 * Single responsibility: assemble one complete outgoing message vector and
 * admit that exact request against the immutable context budget.
 */

use super::super::render::ChatTurn;
use crate::graph_horizon_cli::runtime::{CapacityError, ChatMessage, ContextBudget};

pub(super) struct PreparedRequest {
    pub(super) messages: Vec<ChatMessage>,
    pub(super) characters: usize,
}

// Assemble the exact engine request before checking its context capacity.
pub(super) fn assemble(
    system: Option<&str>,
    history: &[ChatTurn],
    expanded: &str,
    budget: ContextBudget,
) -> Result<PreparedRequest, CapacityError> {
    let messages: Vec<ChatMessage> = system
        .map(|s| ChatMessage::system(s.to_string()))
        .into_iter()
        .chain(history.iter().flat_map(|t| t.messages.iter().cloned()))
        .chain([ChatMessage::user(expanded.to_string())])
        .collect();
    let characters = messages
        .iter()
        .try_fold(0_usize, |sum, message| sum.checked_add(message.chars()))
        .ok_or_else(|| budget.overflow_error())?;
    budget.admit(characters)?;
    Ok(PreparedRequest {
        messages,
        characters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemble_includes_system_history_and_prompt() {
        let history = vec![ChatTurn::new(
            "raw".into(),
            "shown".into(),
            vec![
                ChatMessage::user("old user".into()),
                ChatMessage::assistant("old assistant".into()),
            ],
        )];
        let prepared = assemble(
            Some("system"),
            &history,
            "expanded prompt",
            ContextBudget::new(4096, 128).unwrap(),
        )
        .unwrap();

        assert_eq!(prepared.messages.len(), 4);
        assert_eq!(
            serde_json::to_value(&prepared.messages).unwrap(),
            serde_json::json!([
                {"role": "system", "content": "system"},
                {"role": "user", "content": "old user"},
                {"role": "assistant", "content": "old assistant"},
                {"role": "user", "content": "expanded prompt"}
            ])
        );
        assert_eq!(prepared.characters, 42);
    }
}
