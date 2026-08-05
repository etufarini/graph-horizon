/*
 * Graph Horizon CLI Modules - Plugins - Command - Args
 * Extracts the trailing argument being completed from a command's remainder.
 * Pure parsing helpers for the completion stage; no registry or I/O knowledge.
 */

// Extracts the path argument once the separating space exists, while it is still a
// single trailing token. None when no space has been typed yet, or once a second
// whitespace-separated word begins (mirroring how '@' stops at whitespace).
pub(super) fn path_arg(rest: &str) -> Option<&str> {
    let arg = rest.strip_prefix(char::is_whitespace)?.trim_start();
    (!arg.contains(char::is_whitespace)).then_some(arg)
}
