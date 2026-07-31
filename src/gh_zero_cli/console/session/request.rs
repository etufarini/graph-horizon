/*
 * GH Zero CLI Modules - Console - Session - Request
 * Assembles one turn's outgoing request: message assembly, pruning, token count.
 * Owns only request preparation; the loop drives it once per non-command prompt.
 */

use super::super::render::ChatTurn;
use crate::gh_zero_cli::runtime::{ChatMessage, estimate_tokens, prune_to_limit};

// Builds the pruned request for one turn and the input-token estimate that
// describes it. `expanded` is the exact user message that enters both the
// provider request and, after the turn completes, the conversation history.
pub(super) fn assemble(
    system: Option<&str>,
    history: &[ChatTurn],
    expanded: &str,
    threshold: Option<usize>,
) -> (Vec<ChatMessage>, usize) {
    // Assemble the request: optional system prompt, the stored history, then the
    // new user message. The system prompt is re-prepended every turn and never
    // stored in history, so it is always first and never pruned.
    let request: Vec<ChatMessage> = system
        .map(|s| ChatMessage::system(s.to_string()))
        .into_iter()
        .chain(history.iter().flat_map(|t| t.messages.iter().cloned()))
        .chain([ChatMessage::user(expanded.to_string())])
        .collect();
    let request = prune_to_limit(request, threshold);
    // Prefill speed must describe the exact request sent to the provider: count
    // after pruning, before the request moves into the network future.
    let input_tokens = estimate_tokens(request.iter().map(ChatMessage::chars).sum());
    (request, input_tokens)
}
