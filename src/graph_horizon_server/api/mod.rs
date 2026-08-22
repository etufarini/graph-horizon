/*
 * Graph Horizon headless server - API (OpenAI wire vocabulary, shared root)
 * Single responsibility: expose chat and properties wire vocabularies plus
 * typed chat validation errors. It depends on engine roles only for text-chat
 * mapping and does not model tools, workspace fields, or reasoning controls.
*/

use graph_horizon_engine::Role;

pub(super) mod chat;
pub(super) mod props;

// Typed chat validation errors. The handler pipeline maps every variant to a 400
// with a generic message. The messages carry no internal detail: no
// path, no stack trace, no system internal reaches the client.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ApiError {
    // Chat validation.
    EmptyMessages,
    UnknownRole,
    InvalidMaxTokens,
}

impl ApiError {
    // Generic, client-safe message. No path, no stack trace, no system internal.
    pub(super) fn message(&self) -> &'static str {
        match self {
            ApiError::EmptyMessages => "messages required",
            ApiError::UnknownRole => "unsupported message role",
            ApiError::InvalidMaxTokens => "invalid max_tokens",
        }
    }
}

pub(super) fn role_from_str(role: &str) -> Result<Role, ApiError> {
    match role {
        "system" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        _ => Err(ApiError::UnknownRole),
    }
}
