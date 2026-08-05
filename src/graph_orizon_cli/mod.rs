/*
 * Graph Orizon CLI Modules
 * Single responsibility: expose the interactive chat-only CLI surface. Runtime
 * arguments, engine loading, and HTTP/web startup live in neutral app/server
 * domains; this module does not export tools or reasoning surfaces.
*/

pub(crate) mod console;
pub(crate) mod plugins;
pub(crate) mod runtime;
pub(crate) mod startup;
pub(crate) mod style;
