/*
 * gh_zero_engine — public text-message contract
 * Defines the only conversation values accepted by the engine: owned plain
 * text tagged as system, user, or assistant. It has no model, wire-format,
 * tool, workspace, or reasoning dependency.
 */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}
