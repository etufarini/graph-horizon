/*
 * GH Zero CLI Modules - Plugins - Attachments - Completion
 * Single responsibility: complete trailing attachment/transcript names by
 * listing an already-authorized directory descriptor. It never resolves or
 * returns filesystem paths and silently ignores non-UTF-8 entry names.
 */

use std::path::Path;

use super::{FileAuthority, at_token};
use crate::gh_zero_cli::plugins::completion::completion_tail;

// Returns the completion tail and the full match list for the trailing '@' token.
// Both are derived from one directory scan so the input phase can show the hint
// and the suggestion list without reading the filesystem twice per frame.
pub(crate) fn complete_at_token(
    files: &FileAuthority,
    prompt: &str,
) -> (Option<String>, Vec<String>) {
    match at_token(prompt) {
        Some(token) => complete_path(files, token, None),
        None => (None, Vec::new()),
    }
}

// Returns the completion tail and the matching workspace entries for a partial
// path in one scan. When `only_ext` is Some, regular files are kept only if their
// extension matches it (directories are always kept so the user can still descend
// into them); None lists every entry. Shared by '@' attachments (no filter) and
// the '/import' command (.md).
pub(crate) fn complete_path(
    files: &FileAuthority,
    partial: &str,
    only_ext: Option<&str>,
) -> (Option<String>, Vec<String>) {
    if partial.starts_with('/')
        || partial.contains("//")
        || partial.split('/').any(|part| part == "." || part == "..")
    {
        return (None, Vec::new());
    }
    match partial_matches(files, partial, only_ext) {
        Some((pattern, matches)) => (completion_tail(pattern, &matches), matches),
        None => (None, Vec::new()),
    }
}

// Splits a partial path into its directory and the trailing name pattern, then
// lists the matching entries. Returns None when nothing matches so callers can
// treat "no completion" uniformly.
fn partial_matches<'a>(
    files: &FileAuthority,
    partial: &'a str,
    only_ext: Option<&str>,
) -> Option<(&'a str, Vec<String>)> {
    let (dir_path, pattern) = match partial.rfind('/') {
        Some(i) => (&partial[..i + 1], &partial[i + 1..]),
        None => ("", partial),
    };

    let matches = list_matches(files, dir_path, pattern, only_ext);
    if matches.is_empty() {
        return None;
    }

    Some((pattern, matches))
}

// Lists files and directories in the specified path that start with the given
// pattern. Directories are always included (trailing '/'); files are filtered by
// `only_ext` when set, so '/import' can surface only Markdown files.
fn list_matches(
    files: &FileAuthority,
    dir_path: &str,
    pattern: &str,
    only_ext: Option<&str>,
) -> Vec<String> {
    let search_dir = dir_path.strip_suffix('/').unwrap_or(dir_path);
    let mut matches = Vec::new();

    if let Ok(entries) = files.entries(search_dir) {
        for (name, is_dir) in entries {
            if !name.starts_with(pattern) {
                continue;
            }
            if is_dir {
                matches.push(format!("{}/", name));
            } else if only_ext.is_none_or(|ext| has_extension(&name, ext)) {
                matches.push(name);
            }
        }
    }

    matches.sort();
    matches
}

// True when the file name's extension equals `ext` (case-sensitive, no dot).
fn has_extension(name: &str, ext: &str) -> bool {
    Path::new(name).extension().and_then(|e| e.to_str()) == Some(ext)
}
