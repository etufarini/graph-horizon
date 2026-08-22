/*
 * Graph Horizon CLI Modules
 * Single responsibility: expose the interactive chat-only CLI surface. Runtime
 * argument dispatch and Web startup live in their own domains; this module does
 * not export tools or reasoning surfaces.
*/

pub(crate) mod console;
pub(crate) mod plugins;
pub(crate) mod runtime;
pub(crate) mod startup;
pub(crate) mod style;
