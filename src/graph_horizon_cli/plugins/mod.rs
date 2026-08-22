/*
 * Graph Horizon CLI Modules - Plugins
 * Defines the plugin namespace used by the interactive CLI: workspace-bound
 * attachments, slash commands, shared completion logic, and JSON transcripts.
 * Each child owns its parsing and I/O boundary; this module only exposes them.
*/

pub(crate) mod attachments;
pub(crate) mod command;
pub(crate) mod completion;
pub(crate) mod transcript;
