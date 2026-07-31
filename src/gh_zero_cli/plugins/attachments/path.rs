/*
 * GH Zero CLI Linux anchored-file authority
 * Owns the immutable startup-directory descriptor and opens user-requested
 * descendants relative to already-open directories. It never changes the
 * current directory or returns a validated path for later reopening.
 */

use std::ffi::CString;
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
#[cfg(test)]
use std::path::Path;

const DENIED: io::ErrorKind = io::ErrorKind::PermissionDenied;

const O_DIRECTORY: i32 = 0o200000;
const O_NOFOLLOW: i32 = 0o400000;
const O_CLOEXEC: i32 = 0o2000000;
const O_RDONLY: i32 = 0;
const O_WRONLY: i32 = 1;
const O_CREAT: i32 = 0o100;
const O_TRUNC: i32 = 0o1000;
const O_NONBLOCK: i32 = 0o4000;

unsafe extern "C" {
    fn openat(dirfd: i32, path: *const std::ffi::c_char, flags: i32, ...) -> i32;
}

pub(crate) struct FileAuthority {
    root: File,
}

impl FileAuthority {
    pub(crate) fn capture() -> io::Result<Self> {
        // Open "." first: a concurrent rename of its directory name cannot
        // redirect this descriptor to a replacement at the canonical path.
        let root = File::open(".")?;
        let _canonical_startup_root = std::env::current_dir()?.canonicalize()?;
        if !root.metadata()?.is_dir() {
            return Err(denied());
        }
        Ok(Self { root })
    }

    #[cfg(test)]
    fn from_root(root: &Path) -> io::Result<Self> {
        let root = File::open(root)?;
        if !root.metadata()?.is_dir() {
            return Err(denied());
        }
        Ok(Self { root })
    }

    pub(crate) fn open_read(&self, requested: &str) -> io::Result<File> {
        let components = components(requested)?;
        let (leaf, parents) = components.split_last().ok_or_else(denied)?;
        let parent = self.walk(parents)?;
        let file = open_relative(
            parent.as_raw_fd(),
            leaf,
            O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC,
            0,
        )?;
        if !file.metadata()?.is_file() {
            return Err(denied());
        }
        Ok(file)
    }

    pub(crate) fn open_create(&self, requested: &str) -> io::Result<File> {
        let components = components(requested)?;
        let (leaf, parents) = components.split_last().ok_or_else(denied)?;
        let parent = self.walk(parents)?;
        let file = open_relative(
            parent.as_raw_fd(),
            leaf,
            O_WRONLY | O_CREAT | O_TRUNC | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC,
            0o600,
        )?;
        if !file.metadata()?.is_file() {
            return Err(denied());
        }
        Ok(file)
    }

    pub(crate) fn entries(&self, requested_dir: &str) -> io::Result<Vec<(String, bool)>> {
        let dir = if requested_dir.is_empty() {
            self.walk(&[])?
        } else {
            let parts = components(requested_dir)?;
            self.walk(&parts)?
        };
        // `/proc/self/fd` reopens the already-authorized directory object, not
        // a user path. Root renames therefore do not change authority.
        let descriptor_path = format!("/proc/self/fd/{}", dir.as_raw_fd());
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(descriptor_path)? {
            let entry = entry?;
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            entries.push((name, entry.file_type()?.is_dir()));
        }
        Ok(entries)
    }

    fn walk(&self, components: &[CString]) -> io::Result<File> {
        let dot = CString::new(".").expect("literal has no NUL");
        let mut dir = open_relative(
            self.root.as_raw_fd(),
            &dot,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC,
            0,
        )?;
        for component in components {
            // Invariant: each next lookup is relative to the descriptor returned
            // by the prior lookup. Renames can only preserve that anchor or fail.
            dir = open_relative(
                dir.as_raw_fd(),
                component,
                O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC,
                0,
            )?;
        }
        Ok(dir)
    }
}

fn components(requested: &str) -> io::Result<Vec<CString>> {
    let bytes = requested.as_bytes();
    if bytes.is_empty()
        || bytes[0] == b'/'
        || bytes.last() == Some(&b'/')
        || bytes.windows(2).any(|pair| pair == b"//")
    {
        return Err(denied());
    }
    requested
        .split('/')
        .map(|component| {
            if component == "." || component == ".." || component.is_empty() {
                return Err(denied());
            }
            CString::new(component.as_bytes()).map_err(|_| denied())
        })
        .collect()
}

fn open_relative(dir: RawFd, name: &CString, flags: i32, mode: u32) -> io::Result<File> {
    // SAFETY: `name` is NUL-terminated, `dir` belongs to a live File for the
    // duration of the call, and a successful fd is transferred to exactly one File.
    let fd = unsafe { openat(dir, name.as_ptr(), flags, mode) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: openat returned a new owned descriptor.
    Ok(unsafe { File::from_raw_fd(fd) })
}

fn denied() -> io::Error {
    io::Error::from(DENIED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn root(label: &str) -> (std::path::PathBuf, FileAuthority) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("gh-zero-authority-{label}-{nonce}"));
        std::fs::create_dir(&path).unwrap();
        let authority = FileAuthority::from_root(&path).unwrap();
        (path, authority)
    }

    #[test]
    fn rejects_invalid_components_and_non_regular_targets() {
        let (root, authority) = root("invalid");
        std::fs::create_dir(root.join("dir")).unwrap();
        for path in ["", ".", "..", "a//b", "/tmp/file", "dir/"] {
            assert!(authority.open_read(path).is_err(), "{path}");
            assert!(authority.open_create(path).is_err(), "{path}");
        }
        assert!(authority.open_read("dir").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn nested_handles_read_and_create_without_reopening_paths() {
        let (root, authority) = root("nested");
        std::fs::create_dir(root.join("nested")).unwrap();
        std::fs::write(root.join("nested/input.txt"), "inside").unwrap();
        let mut input = authority.open_read("nested/input.txt").unwrap();
        let mut text = String::new();
        input.read_to_string(&mut text).unwrap();
        assert_eq!(text, "inside");
        authority
            .open_create("nested/output.txt")
            .unwrap()
            .write_all(b"created")
            .unwrap();
        assert_eq!(
            std::fs::read(root.join("nested/output.txt")).unwrap(),
            b"created"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_inside_outside_and_existing_leaf_symlinks() {
        let (root, authority) = root("symlink");
        let outside = root.with_extension("outside");
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(root.join("inside.txt"), "inside").unwrap();
        std::fs::write(outside.join("sentinel.txt"), "outside").unwrap();
        symlink("inside.txt", root.join("inside-link")).unwrap();
        symlink(outside.join("sentinel.txt"), root.join("outside-link")).unwrap();
        symlink(&outside, root.join("parent-link")).unwrap();
        for path in ["inside-link", "outside-link", "parent-link/sentinel.txt"] {
            assert!(authority.open_read(path).is_err());
            assert!(authority.open_create(path).is_err());
        }
        assert_eq!(
            std::fs::read(outside.join("sentinel.txt")).unwrap(),
            b"outside"
        );
        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn opened_parent_survives_concurrent_name_swap() {
        let (root, authority) = root("swap");
        let outside = root.with_extension("outside");
        std::fs::create_dir(root.join("parent")).unwrap();
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(outside.join("sentinel"), "outside").unwrap();
        let parent = authority.walk(&components("parent").unwrap()).unwrap();
        std::fs::rename(root.join("parent"), root.join("old-parent")).unwrap();
        symlink(&outside, root.join("parent")).unwrap();
        let leaf = CString::new("created").unwrap();
        open_relative(
            parent.as_raw_fd(),
            &leaf,
            O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW | O_CLOEXEC,
            0o600,
        )
        .unwrap();
        assert!(root.join("old-parent/created").is_file());
        assert!(!outside.join("created").exists());
        assert_eq!(std::fs::read(outside.join("sentinel")).unwrap(), b"outside");
        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_dir_all(outside).unwrap();
    }

    #[test]
    fn root_rename_keeps_the_same_directory_object() {
        let (root, authority) = root("rename");
        std::fs::write(root.join("file"), "same").unwrap();
        let renamed = root.with_extension("renamed");
        std::fs::rename(&root, &renamed).unwrap();
        let mut text = String::new();
        authority
            .open_read("file")
            .unwrap()
            .read_to_string(&mut text)
            .unwrap();
        assert_eq!(text, "same");
        std::fs::remove_dir_all(renamed).unwrap();
    }

    #[test]
    fn removed_root_fails_without_falling_back_to_current_dir() {
        let (root, authority) = root("removed");
        std::fs::remove_dir(&root).unwrap();
        assert!(authority.open_create("must-not-escape").is_err());
        assert!(
            !std::env::current_dir()
                .unwrap()
                .join("must-not-escape")
                .exists()
        );
    }

    #[test]
    fn invalid_utf8_entries_are_ignored_by_completion_listing() {
        use std::os::unix::ffi::OsStringExt;

        let (root, authority) = root("non-utf8");
        let invalid = std::ffi::OsString::from_vec(vec![b'n', b'a', 0xff]);
        File::create(root.join(invalid)).unwrap();
        assert!(authority.entries("").unwrap().is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn permission_failure_is_closed_and_generic_at_the_caller() {
        let (root, authority) = root("permission");
        std::fs::create_dir(root.join("locked")).unwrap();
        std::fs::set_permissions(root.join("locked"), std::fs::Permissions::from_mode(0o0))
            .unwrap();
        if unsafe { geteuid() } != 0 {
            assert!(authority.open_read("locked/file").is_err());
        }
        std::fs::set_permissions(root.join("locked"), std::fs::Permissions::from_mode(0o700))
            .unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    unsafe extern "C" {
        fn geteuid() -> u32;
    }
}
