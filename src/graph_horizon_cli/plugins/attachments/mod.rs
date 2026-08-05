/*
 * Graph Horizon CLI Modules - Plugins - Attachments
 * Single responsibility: expand @filename tokens from already-open files rooted
 * by the startup authority. Completion lives in completion.rs and descriptor
 * traversal in path.rs; this module never reopens a checked path.
*/

use std::io::Read;
use std::path::Path;

mod completion;
mod path;

pub(crate) use completion::{complete_at_token, complete_path};
pub(crate) use path::FileAuthority;

// Scans the prompt for @filename tokens and replaces them with the file's content in Markdown blocks.
pub(crate) fn attach_local_files(files: &FileAuthority, prompt: &str) -> String {
    let mut expanded = String::new();
    let mut last_end = 0;

    // Use a simple scanner to find @ tokens.
    for (i, _) in prompt.match_indices('@') {
        // Add text before the @
        expanded.push_str(&prompt[last_end..i]);

        let after_at = &prompt[i + 1..];
        let filename_len = attachment_token_len(after_at);
        let filename = &after_at[..filename_len];

        if !filename.is_empty()
            && let Ok(content) = read_local_file(files, filename)
        {
            // The fence must be longer than any backtick run in the content,
            // or a file containing ``` would close the block early and leak
            // the rest of the file outside the code fence.
            let fence = "`".repeat(longest_backtick_run(&content).max(2) + 1);
            expanded.push_str("\n\n### File: ");
            expanded.push_str(filename);
            expanded.push('\n');
            expanded.push_str(&fence);
            // Try to guess extension for markdown highlight.
            if let Some(ext) = Path::new(filename).extension().and_then(|s| s.to_str()) {
                expanded.push_str(ext);
            }
            expanded.push('\n');
            expanded.push_str(&content);
            expanded.push('\n');
            expanded.push_str(&fence);
            expanded.push('\n');
            last_end = i + 1 + filename_len;
            continue;
        }

        // If not a file or empty, just push the @ and continue.
        expanded.push('@');
        last_end = i + 1;
    }

    // Add remaining text
    expanded.push_str(&prompt[last_end..]);
    expanded
}

// Extracts the path token after '@' only when it is the trailing token of the
// prompt: anything after the token (e.g. another word) means the '@' is already
// closed and there is nothing left to complete.
fn at_token(prompt: &str) -> Option<&str> {
    let at_index = prompt.rfind('@')?;
    let after_at = &prompt[at_index + 1..];
    let token_len = attachment_token_len(after_at);
    (token_len == after_at.len()).then_some(after_at)
}

// Reads a file if it's valid UTF-8 and within allowed paths.
fn read_local_file(files: &FileAuthority, filename: &str) -> Result<String, std::io::Error> {
    let mut file = files.open_read(filename)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

// Length of the longest consecutive run of backticks in `text`, used to size
// an enclosing Markdown fence that the content cannot accidentally close.
fn longest_backtick_run(text: &str) -> usize {
    let mut longest = 0;
    let mut current = 0;
    for c in text.chars() {
        if c == '`' {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    longest
}

// Determines the length of the attachment token following an '@' character.
fn attachment_token_len(input: &str) -> usize {
    input
        .find(|c: char| !is_attachment_char(c))
        .unwrap_or(input.len())
}

// Determines if a character is valid in an attachment token (part of a filename).
fn is_attachment_char(c: char) -> bool {
    !c.is_whitespace() && (!c.is_ascii_punctuation() || matches!(c, '.' | '_' | '-' | '/'))
}

// Tests for the file attachment plugin, including attaching existing files.
// Handling non-existent files, and completing partial filenames.
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::fs::File;
    use std::io::Write;

    fn files() -> FileAuthority {
        FileAuthority::capture().unwrap()
    }

    #[test]
    fn test_attach_local_files_no_file() {
        let prompt = "Hello @non_existent.txt";
        let result = attach_local_files(&files(), prompt);
        assert_eq!(result, "Hello @non_existent.txt");
    }

    #[test]
    fn test_attach_local_files_with_file() {
        let filename = "test_file.txt";
        let mut file = File::create(filename).unwrap();
        writeln!(file, "File content").unwrap();

        let prompt = "Check this: @test_file.txt";
        let result = attach_local_files(&files(), prompt);

        assert!(result.contains("### File: test_file.txt"));
        assert!(result.contains("```txt\nFile content\n\n```"));

        fs::remove_file(filename).unwrap();
    }

    #[test]
    fn test_attach_local_files_with_subdir_file() {
        let dir = "test_subdir";
        fs::create_dir_all(dir).unwrap();
        let filename = "test_subdir/test_file.txt";
        let mut file = File::create(filename).unwrap();
        writeln!(file, "Subdir content").unwrap();

        let prompt = "Check this: @test_subdir/test_file.txt";
        let result = attach_local_files(&files(), prompt);

        assert!(result.contains("### File: test_subdir/test_file.txt"));
        assert!(result.contains("```txt\nSubdir content\n\n```"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_complete_at_token() {
        let filename1 = "comp_test_1.txt";
        let filename2 = "comp_test_2.txt";
        File::create(filename1).unwrap();
        File::create(filename2).unwrap();

        let prompt = "Look at @comp_te";
        let authority = files();
        let (completion, matches) = complete_at_token(&authority, prompt);
        assert_eq!(completion, Some("st_".to_string()));
        assert_eq!(matches.len(), 2);

        let prompt2 = "Look at @comp_test_1";
        let (completion2, _) = complete_at_token(&authority, prompt2);
        assert_eq!(completion2, Some(".txt".to_string()));

        fs::remove_file(filename1).unwrap();
        fs::remove_file(filename2).unwrap();
    }

    #[test]
    fn test_attach_file_with_backticks_uses_longer_fence() {
        let filename = "fence_test.md";
        let mut file = File::create(filename).unwrap();
        // Content containing a ``` fence must not close the enclosing block.
        write!(file, "```rust\nlet x = 1;\n```").unwrap();

        let result = attach_local_files(&files(), "See @fence_test.md");
        assert!(result.contains("````md\n```rust\nlet x = 1;\n```\n````"));

        fs::remove_file(filename).unwrap();
    }

    #[test]
    fn authority_rejects_disallowed_paths() {
        let authority = files();
        let absolute = std::env::current_dir()
            .unwrap()
            .join("Cargo.toml")
            .display()
            .to_string();

        assert!(authority.open_read(&absolute).is_err());
        assert!(authority.open_read("../Cargo.toml").is_err());
        assert!(authority.open_create("/tmp/evil.json").is_err());
        assert!(authority.open_create("../evil.json").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn authority_rejects_symlink_components() {
        use std::os::unix::fs::symlink;

        let dir = "symlink_probe_dir";
        let target = "symlink_probe_target.txt";
        let link = format!("{dir}/link.txt");
        fs::create_dir_all(dir).unwrap();
        File::create(target).unwrap();
        symlink(format!("../{target}"), &link).unwrap();

        assert!(files().open_read(&link).is_err());

        fs::remove_file(&link).unwrap();
        fs::remove_file(target).unwrap();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_autocomplete_rejects_absolute_path() {
        let prompt = "Look at @/";
        let (completion, matches) = complete_at_token(&files(), prompt);
        assert_eq!(completion, None);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_complete_path_filters_by_extension() {
        let dir = "ext_filter_dir";
        fs::create_dir_all(dir).unwrap();
        File::create(format!("{dir}/keep.md")).unwrap();
        File::create(format!("{dir}/skip.txt")).unwrap();
        fs::create_dir_all(format!("{dir}/sub")).unwrap();

        // With a .md filter only the Markdown file and the directory survive.
        let authority = files();
        let (_, mut matches) = complete_path(&authority, &format!("{dir}/"), Some("md"));
        matches.sort();
        assert_eq!(matches, vec!["keep.md".to_string(), "sub/".to_string()]);

        // Without a filter every entry is listed.
        assert_eq!(
            complete_path(&authority, &format!("{dir}/"), None).1.len(),
            3
        );

        fs::remove_dir_all(dir).unwrap();
    }
}
