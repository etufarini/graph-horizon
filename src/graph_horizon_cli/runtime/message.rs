/*
 * Graph Horizon CLI Modules - Runtime - Message
 * Single responsibility: represent text-only chat messages sent to local and
 * HTTP providers. It depends on serde and engine message types, and cannot
 * encode tools, functions, workspace data, or reasoning payloads.
 */

use graph_horizon_engine::{Message as InferMessage, Role};
use serde::Serialize;

// Role accepted by the chat completions API.
#[derive(Clone, Serialize)]
#[serde(rename_all = "lowercase")]
enum ChatRole {
    System,
    User,
    Assistant,
}

// One chat message sent to the model provider. The private role plus constructors
// are the invariant: only system/user/assistant text can be represented.
#[derive(Clone, Serialize)]
pub(crate) struct ChatMessage {
    role: ChatRole,
    content: String,
}

impl ChatMessage {
    // Plain text message with no tool fields, shared by the role constructors.
    fn plain(role: ChatRole, content: String) -> Self {
        Self { role, content }
    }

    // Creates a system message from a fixed instruction owned by the caller.
    pub(crate) fn system(content: String) -> Self {
        Self::plain(ChatRole::System, content)
    }

    // Creates a user message from prompt content owned by the caller.
    pub(crate) fn user(content: String) -> Self {
        Self::plain(ChatRole::User, content)
    }

    // Creates an assistant message from response content owned by the caller.
    pub(crate) fn assistant(content: String) -> Self {
        Self::plain(ChatRole::Assistant, content)
    }

    // Unicode scalar-value count of the private content sent on the wire.
    pub(crate) fn chars(&self) -> usize {
        self.content.chars().count()
    }

    // Converts into the engine's message type for the local inference path.
    // Owns the role mapping so ChatRole and the content field stay private to
    // this module; tool-call wire fields are not modelled in the engine history.
    pub(crate) fn into_engine(self) -> InferMessage {
        let role = match self.role {
            ChatRole::System => Role::System,
            ChatRole::User => Role::User,
            ChatRole::Assistant => Role::Assistant,
        };
        InferMessage {
            role,
            content: self.content,
        }
    }
}
