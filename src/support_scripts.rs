/*
 * Support script acceptance tests
 * Single responsibility: exercise the retained shell scripts as external
 * interfaces through disposable fixtures, proving installer/bootstrap safety,
 * quoted read-only model handling, class-sensitive semantic protocols, and
 * explicit external-verification output without real builds or network use.
 */

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[derive(Debug, PartialEq, Eq)]
struct CatalogRow<'a> {
    id: &'a str,
    chat: &'a str,
    q4_file: &'a str,
    bytes: u64,
    sha256: &'a str,
    q8_file: &'a str,
}

fn parse_catalog(text: &str) -> Result<Vec<CatalogRow<'_>>, u8> {
    use std::collections::HashSet;

    if text.contains('\r') {
        return Err(2);
    }
    let mut rows = Vec::new();
    let mut ids = HashSet::new();
    let mut q4_files = HashSet::new();
    let mut q8_files = HashSet::new();
    let mut hashes = HashSet::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 6
            || fields.iter().any(|field| {
                field.is_empty() || field.chars().any(|character| character.is_whitespace())
            })
        {
            return Err(2);
        }
        let [id, chat, q4_file, bytes, sha256, q8_file] = fields.as_slice() else {
            return Err(2);
        };
        if id.starts_with('-')
            || id.ends_with('-')
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            || !matches!(*chat, "instruct" | "reasoning")
            || !bytes.bytes().all(|byte| byte.is_ascii_digit())
            || bytes
                .parse::<u64>()
                .ok()
                .filter(|value| *value > 0)
                .is_none()
            || sha256.len() != 64
            || !sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || Path::new(q4_file).components().count() != 1
            || Path::new(q8_file).components().count() != 1
            || [q4_file, q8_file].iter().any(|file| {
                file.starts_with('-')
                    || !file.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                    })
            })
            || !ids.insert(*id)
            || !q4_files.insert(*q4_file)
            || !q8_files.insert(*q8_file)
            || !hashes.insert(*sha256)
        {
            return Err(2);
        }
        rows.push(CatalogRow {
            id,
            chat,
            q4_file,
            bytes: bytes.parse().unwrap(),
            sha256,
            q8_file,
        });
    }
    Ok(rows)
}

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn script(relative: &str, args: &[&str]) -> Output {
    Command::new("bash")
        .arg(repository().join(relative))
        .args(args)
        .output()
        .expect("run support script")
}

fn fixture_dir(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "graph-horizon support scripts {} {label}",
        std::process::id()
    ));
    if path.exists() {
        fs::remove_dir_all(&path).expect("remove stale fixture");
    }
    fs::create_dir_all(&path).expect("create fixture");
    path
}

fn write_executable(path: &Path, source: &[u8]) {
    fs::write(path, source).unwrap();
    let mut permissions = fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).unwrap();
}

fn installer_fixture(label: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let fixture = fixture_dir(label);
    let root = fixture.join("repository with spaces");
    let bin = fixture.join("bin");
    let log = fixture.join("build calls");
    fs::create_dir_all(root.join("support")).unwrap();
    fs::create_dir_all(root.join("web/frontend")).unwrap();
    fs::create_dir(&bin).unwrap();
    fs::copy(
        repository().join("support/install.sh"),
        root.join("support/install.sh"),
    )
    .unwrap();
    write_executable(
        &bin.join("uname"),
        br#"#!/usr/bin/env bash
case "$1" in
  -s) printf '%s\n' "$GRAPH_HORIZON_TEST_OS" ;;
  -m) printf '%s\n' "$GRAPH_HORIZON_TEST_ARCH" ;;
  *) exit 2 ;;
esac
"#,
    );
    write_executable(
        &bin.join("npm"),
        br#"#!/usr/bin/env bash
printf 'npm\t%s\n' "$*" >> "$GRAPH_HORIZON_TEST_LOG"
"#,
    );
    write_executable(
        &bin.join("cargo"),
        br#"#!/usr/bin/env bash
printf 'cargo\t%s\n' "$*" >> "$GRAPH_HORIZON_TEST_LOG"
profile=release
while (($#)); do
  if [[ "$1" == --profile ]]; then profile="$2"; shift; fi
  shift
done
mkdir -p "$GRAPH_HORIZON_FIXTURE_ROOT/target/$profile"
printf '#!/bin/sh\n' > "$GRAPH_HORIZON_FIXTURE_ROOT/target/$profile/graph-horizon"
"#,
    );
    write_executable(
        &bin.join("xcrun"),
        br#"#!/usr/bin/env bash
[[ -z "${GRAPH_HORIZON_XCRUN_FAIL:-}" ]]
"#,
    );
    (fixture, root, bin, log)
}

fn run_installer(
    fixture: &Path,
    root: &Path,
    bin: &Path,
    log: &Path,
    os: &str,
    arch: &str,
    args: &[&str],
) -> Output {
    Command::new("/bin/bash")
        .arg(root.join("support/install.sh"))
        .args(args)
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("HOME", fixture.join("home"))
        .env("GRAPH_HORIZON_TEST_OS", os)
        .env("GRAPH_HORIZON_TEST_ARCH", arch)
        .env("GRAPH_HORIZON_TEST_LOG", log)
        .env("GRAPH_HORIZON_FIXTURE_ROOT", root)
        .output()
        .unwrap()
}

fn bootstrap_fixture(label: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let fixture = fixture_dir(label);
    let bin = fixture.join("bin");
    let temp = fixture.join("temporary download");
    let argument_log = fixture.join("delegated arguments");
    fs::create_dir(&bin).unwrap();
    fs::copy(repository().join("install.sh"), fixture.join("install.sh")).unwrap();
    write_executable(
        &bin.join("curl"),
        br#"#!/usr/bin/env bash
[[ -z "${GRAPH_HORIZON_CURL_FAIL:-}" ]] || exit 22
[[ "$#" == 7 && "$1" == --fail && "$2" == --location && "$3" == --silent ]]
[[ "$4" == --show-error && "$5" == --output ]]
[[ "$7" == https://github.com/etufarini/gh-zero-engine-ministral3/archive/refs/heads/main.tar.gz ]]
cp "$GRAPH_HORIZON_TEST_ARCHIVE" "$6"
"#,
    );
    write_executable(
        &bin.join("mktemp"),
        br#"#!/usr/bin/env bash
mkdir "$GRAPH_HORIZON_TEST_TEMP"
printf '%s\n' "$GRAPH_HORIZON_TEST_TEMP"
"#,
    );
    write_executable(
        &bin.join("tar"),
        br#"#!/usr/bin/env bash
if [[ -n "${GRAPH_HORIZON_TAR_LIST:-}" && "$1" == -tzf ]]; then
  /bin/cat "$GRAPH_HORIZON_TAR_LIST"
  exit 0
fi
exec /usr/bin/tar "$@"
"#,
    );
    write_executable(
        &bin.join("find"),
        b"#!/usr/bin/env bash\nexec /usr/bin/find \"$@\"\n",
    );
    (fixture, bin, temp, argument_log)
}

fn source_archive(fixture: &Path, name: &str, complete: bool, with_symlink: bool) -> PathBuf {
    let tree = fixture.join(format!("{name} tree"));
    let root = tree.join("gh-zero-engine-ministral3-main");
    fs::create_dir_all(root.join("support")).unwrap();
    fs::create_dir_all(root.join("web/frontend")).unwrap();
    write_executable(
        &root.join("support/install.sh"),
        br#"#!/usr/bin/env bash
printf '%s\n' "$@" > "$GRAPH_HORIZON_ARGUMENT_LOG"
exit "${GRAPH_HORIZON_DELEGATE_STATUS:-0}"
"#,
    );
    fs::write(root.join("Cargo.toml"), b"[workspace]\n").unwrap();
    fs::write(root.join("web/frontend/package.json"), b"{}\n").unwrap();
    if complete {
        fs::write(root.join("web/frontend/package-lock.json"), b"{}\n").unwrap();
    }
    if with_symlink {
        std::os::unix::fs::symlink("Cargo.toml", root.join("linked manifest")).unwrap();
    }
    let archive = fixture.join(format!("{name}.tar.gz"));
    let output = Command::new("/usr/bin/tar")
        .args([
            "-czf",
            archive.to_str().unwrap(),
            "-C",
            tree.to_str().unwrap(),
            "gh-zero-engine-ministral3-main",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "tar fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    archive
}

fn run_bootstrap(
    fixture: &Path,
    bin: &Path,
    temp: &Path,
    argument_log: &Path,
    archive: &Path,
    args: &[&str],
) -> Output {
    Command::new("/bin/bash")
        .arg(fixture.join("install.sh"))
        .args(args)
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("GRAPH_HORIZON_TEST_ARCHIVE", archive)
        .env("GRAPH_HORIZON_TEST_TEMP", temp)
        .env("GRAPH_HORIZON_ARGUMENT_LOG", argument_log)
        .output()
        .unwrap()
}

#[test]
fn catalog_contract() {
    let catalog = include_str!("../support/models.tsv");
    let rows = parse_catalog(catalog).expect("approved catalog");
    assert_eq!(
        rows.iter()
            .map(|row| {
                (
                    row.id,
                    row.chat,
                    row.q4_file,
                    row.bytes,
                    row.sha256,
                    row.q8_file,
                )
            })
            .collect::<Vec<_>>(),
        [
            (
                "3b-instruct",
                "instruct",
                "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
                2_147_023_008,
                "9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8",
                "Ministral-3-3B-Instruct-2512-Q8_0.gguf",
            ),
            (
                "3b-reasoning",
                "reasoning",
                "Ministral-3-3B-Reasoning-2512-Q4_K_M.gguf",
                2_147_021_472,
                "7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc",
                "Ministral-3-3B-Reasoning-2512-Q8_0.gguf",
            ),
            (
                "8b-instruct",
                "instruct",
                "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
                5_198_911_904,
                "33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761",
                "Ministral-3-8B-Instruct-2512-Q8_0.gguf",
            ),
            (
                "8b-reasoning",
                "reasoning",
                "Ministral-3-8B-Reasoning-2512-Q4_K_M.gguf",
                5_198_910_368,
                "894aa3645ef8708a81dbe201c26105ce37c4c741252c89c5a78f81b49ac438c6",
                "Ministral-3-8B-Reasoning-2512-Q8_0.gguf",
            ),
            (
                "14b-instruct",
                "instruct",
                "Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
                8_239_593_024,
                "824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613",
                "Ministral-3-14B-Instruct-2512-Q8_0.gguf",
            ),
            (
                "14b-reasoning",
                "reasoning",
                "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf",
                8_239_591_488,
                "fe08ca2158cd7438211ec6a4e5256d31bc980f016e3f5b635fe91fe6848d461c",
                "Ministral-3-14B-Reasoning-2512-Q8_0.gguf",
            ),
        ]
    );
    assert_eq!(rows.len(), 6);
    assert!(rows.iter().all(|row| !Path::new(row.q4_file).is_absolute()));
    assert!(rows.iter().all(|row| !Path::new(row.q8_file).is_absolute()));

    for (case, malformed) in [
        catalog.replacen("\tinstruct\t", "\tunknown\t", 1),
        catalog.replacen("\t2147023008\t", "\t2GB\t", 1),
        catalog.replacen("9ed150d4", "9ED150D4", 1),
        catalog.replacen("9ed150d4", "9ed150d", 1),
        catalog.replacen("3b-instruct", "3b instruct", 1),
        catalog.replacen("3b-instruct", "-3b-instruct", 1),
        catalog.replacen("Ministral-3-3B-Instruct", "$MODEL", 1),
        catalog.replacen("Ministral-3-3B-Instruct", "-Ministral", 1),
        catalog.replacen("\t2147023008\t", "\t\t", 1),
        catalog.replacen(
            "Ministral-3-3B-Instruct-2512-Q8_0.gguf\n",
            "Ministral-3-3B-Instruct-2512-Q8_0.gguf\textra\n",
            1,
        ),
        catalog.replace('\n', "\r\n"),
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(parse_catalog(&malformed), Err(2), "malformed case {case}");
    }

    let first = catalog
        .lines()
        .find(|line| line.starts_with("3b-instruct\t"))
        .unwrap();
    let fields = first.split('\t').collect::<Vec<_>>();
    for duplicate in [
        format!(
            "{}\tinstruct\tunique-q4.gguf\t1\t{}\tunique-q8.gguf",
            fields[0],
            "a".repeat(64)
        ),
        format!(
            "unique-id\tinstruct\t{}\t1\t{}\tunique-q8.gguf",
            fields[2],
            "b".repeat(64)
        ),
        format!(
            "unique-id\tinstruct\tunique-q4.gguf\t1\t{}\tunique-q8.gguf",
            fields[4]
        ),
        format!(
            "unique-id\tinstruct\tunique-q4.gguf\t1\t{}\t{}",
            "c".repeat(64),
            fields[5]
        ),
    ] {
        assert_eq!(parse_catalog(&format!("{catalog}{duplicate}\n")), Err(2));
    }
}

#[test]
fn support_scripts_reject_invalid_values_before_execution() {
    let fixture = fixture_dir("invalid");
    let model = fixture.join("model with spaces.gguf");
    fs::write(&model, b"unchanged model bytes").unwrap();
    let model = model.to_str().unwrap();
    let legacy_vulkan = concat!("vul", "can");

    for (relative, args) in [
        ("support/install.sh", vec!["--backend", "invalid"]),
        ("support/install.sh", vec!["--backend", legacy_vulkan]),
        ("support/install.sh", vec![]),
        (
            "support/profiling/profile.sh",
            vec![
                "--model",
                model,
                "--backend",
                "cpu",
                "--context",
                "0",
                "--kv",
                "f16",
            ],
        ),
        (
            "support/profiling/validate-kv.sh",
            vec!["--backend", "invalid", "--context", "1"],
        ),
        (
            "support/profiling/validate-kv.sh",
            vec!["--backend", legacy_vulkan, "--context", "1"],
        ),
        (
            "support/profiling/profile.sh",
            vec![
                "--model",
                model,
                "--backend",
                legacy_vulkan,
                "--context",
                "1",
                "--kv",
                "f16",
            ],
        ),
        ("support/profiling/validate-weights.sh", vec!["--unknown"]),
        ("support/testing/semantic-check.sh", vec!["--unknown"]),
        (
            "support/testing/parity-check.sh",
            vec![
                "--models-dir",
                model,
                "--model-id",
                "3b-instruct",
                "--backend",
                legacy_vulkan,
                "--kv",
                "f16",
                "--reference-server",
                model,
            ],
        ),
        (
            "support/testing/run-graph-horizon.sh",
            vec![
                "--model",
                model,
                "--backend",
                "cpu",
                "--context",
                "1",
                "--kv",
                "invalid",
            ],
        ),
        (
            "support/testing/run-graph-horizon.sh",
            vec![
                "--model",
                model,
                "--backend",
                legacy_vulkan,
                "--context",
                "1",
                "--kv",
                "f16",
            ],
        ),
    ] {
        let output = script(relative, &args);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{relative} accepted invalid input"
        );
    }

    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn artifact_helpers_support_gnu_and_bsd_tools() {
    let fixture = fixture_dir("artifact helpers");
    let artifact = fixture.join("artifact with spaces");
    let bin = fixture.join("bin");
    fs::create_dir(&bin).unwrap();
    fs::write(&artifact, b"fixture").unwrap();
    let digest = "0123456789abcdef".repeat(4);

    let stat = bin.join("stat");
    let shasum = bin.join("shasum");
    fs::write(
        &stat,
        b"#!/bin/bash\nif [[ \"$1\" == -c ]]; then exit 1; fi\n[[ \"$1 $2\" == '-f %z' ]] || exit 2\nprintf '7\\n'\n",
    )
    .unwrap();
    fs::write(
        &shasum,
        format!(
            "#!/bin/bash\n[[ \"$1 $2\" == '-a 256' ]] || exit 2\nprintf '{}  %s\\n' \"${{!#}}\"\n",
            digest
        ),
    )
    .unwrap();
    for executable in [&stat, &shasum] {
        let mut permissions = fs::metadata(executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(executable, permissions).unwrap();
    }
    let path = bin.display().to_string();
    let command = format!(
        "source \"{}\"; artifact_size \"$1\"; artifact_sha256 \"$1\"",
        repository().join("support/artifact.sh").display()
    );
    let output = Command::new("/bin/bash")
        .args(["-c", &command, "artifact-test", artifact.to_str().unwrap()])
        .env("PATH", path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("7\n{digest}\n")
    );

    let source = fs::read_to_string(repository().join("support/artifact.sh")).unwrap();
    assert!(!source.contains("set -"));
    assert!(source.contains("stat -c %s"));
    assert!(source.contains("stat -f %z"));
    assert!(source.contains("sha256sum"));
    assert!(source.contains("shasum -a 256"));
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn bootstrap_forwards_arguments_and_cleans_temporary_tree() {
    let (fixture, bin, temp, argument_log) = bootstrap_fixture("bootstrap forwarding");
    let archive = source_archive(&fixture, "safe source", true, false);
    let prefix = "/tmp/prefix with spaces";
    let args = ["--backend", "cpu", "--profile", "fast", "--prefix", prefix];
    let output = run_bootstrap(&fixture, &bin, &temp, &argument_log, &archive, &args);
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read_to_string(&argument_log)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        args
    );
    assert!(!temp.exists());

    fs::remove_file(&argument_log).unwrap();
    let output = Command::new("/bin/bash")
        .arg(fixture.join("install.sh"))
        .args(["--backend", "cpu"])
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("GRAPH_HORIZON_TEST_ARCHIVE", &archive)
        .env("GRAPH_HORIZON_TEST_TEMP", &temp)
        .env("GRAPH_HORIZON_ARGUMENT_LOG", &argument_log)
        .env("GRAPH_HORIZON_DELEGATE_STATUS", "2")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(!temp.exists());

    fs::remove_file(&argument_log).unwrap();
    let output = Command::new("/bin/bash")
        .arg(fixture.join("install.sh"))
        .args(["--backend", "cpu"])
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("GRAPH_HORIZON_TEST_ARCHIVE", &archive)
        .env("GRAPH_HORIZON_TEST_TEMP", &temp)
        .env("GRAPH_HORIZON_ARGUMENT_LOG", &argument_log)
        .env("GRAPH_HORIZON_CURL_FAIL", "1")
        .output()
        .unwrap();
    assert_ne!(output.status.code(), Some(0));
    assert!(!argument_log.exists());
    assert!(!temp.exists());
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn bootstrap_rejects_unsafe_or_incomplete_archives() {
    let (fixture, bin, temp, argument_log) = bootstrap_fixture("bootstrap rejection");
    let incomplete = source_archive(&fixture, "incomplete source", false, false);
    let symlink = source_archive(&fixture, "symlink source", true, true);
    for archive in [&incomplete, &symlink] {
        let output = run_bootstrap(
            &fixture,
            &bin,
            &temp,
            &argument_log,
            archive,
            &["--backend", "cpu"],
        );
        assert_ne!(output.status.code(), Some(0));
        assert!(!argument_log.exists());
        assert!(!temp.exists());
    }

    let safe = source_archive(&fixture, "listed source", true, false);
    for (index, members) in [
        "/absolute/member\n",
        "gh-zero-engine-ministral3-main/../escape\n",
        "gh-zero-engine-ministral3-main/./member\n",
        "another-root/member\n",
        "",
    ]
    .into_iter()
    .enumerate()
    {
        let listing = fixture.join(format!("unsafe listing {index}"));
        fs::write(&listing, members).unwrap();
        let output = Command::new("/bin/bash")
            .arg(fixture.join("install.sh"))
            .args(["--backend", "cpu"])
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .env("GRAPH_HORIZON_TEST_ARCHIVE", &safe)
            .env("GRAPH_HORIZON_TEST_TEMP", &temp)
            .env("GRAPH_HORIZON_ARGUMENT_LOG", &argument_log)
            .env("GRAPH_HORIZON_TAR_LIST", &listing)
            .output()
            .unwrap();
        assert_ne!(
            output.status.code(),
            Some(0),
            "unsafe listing {index} accepted"
        );
        assert!(!argument_log.exists());
        assert!(!temp.exists());
    }
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn installer_rejects_unsupported_platform_backend_pairs() {
    let (fixture, root, bin, log) = installer_fixture("installer platform matrix");
    let cases = [
        ("Darwin", "arm64", "cpu", true),
        ("Darwin", "arm64", "vulkan", true),
        ("Darwin", "arm64", "vulkan-hybrid", true),
        ("Darwin", "arm64", "metal", true),
        ("Darwin", "arm64", "metal-hybrid", true),
        ("Linux", "x86_64", "cpu", true),
        ("Linux", "x86_64", "vulkan", true),
        ("Linux", "x86_64", "vulkan-hybrid", true),
        ("Linux", "x86_64", "metal", false),
        ("Linux", "x86_64", "metal-hybrid", false),
        ("Darwin", "x86_64", "cpu", false),
        ("FreeBSD", "x86_64", "cpu", false),
    ];
    for (index, (os, arch, backend, accepted)) in cases.into_iter().enumerate() {
        fs::write(&log, []).unwrap();
        let prefix = fixture.join(format!("prefix {index}"));
        let output = run_installer(
            &fixture,
            &root,
            &bin,
            &log,
            os,
            arch,
            &["--backend", backend, "--prefix", prefix.to_str().unwrap()],
        );
        assert_eq!(
            output.status.success(),
            accepted,
            "{os}/{arch}/{backend}: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if accepted {
            assert_eq!(fs::read_to_string(&log).unwrap().lines().count(), 3);
        } else {
            assert_eq!(output.status.code(), Some(2));
            assert!(fs::read(&log).unwrap().is_empty());
            assert!(
                String::from_utf8_lossy(&output.stderr).contains(&format!("{os}/{arch}/{backend}"))
            );
        }
    }

    fs::write(&log, []).unwrap();
    let prefix = fixture.join("metal tools");
    let output = Command::new("/bin/bash")
        .arg(root.join("support/install.sh"))
        .args(["--backend", "metal", "--prefix", prefix.to_str().unwrap()])
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("HOME", fixture.join("home"))
        .env("GRAPH_HORIZON_TEST_OS", "Darwin")
        .env("GRAPH_HORIZON_TEST_ARCH", "arm64")
        .env("GRAPH_HORIZON_TEST_LOG", &log)
        .env("GRAPH_HORIZON_FIXTURE_ROOT", &root)
        .env("GRAPH_HORIZON_XCRUN_FAIL", "1")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Metal compiler is unavailable"));
    assert!(fs::read(&log).unwrap().is_empty());
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn installer_rejects_unsafe_prefixes_before_build() {
    let (fixture, root, bin, log) = installer_fixture("installer unsafe prefixes");
    for prefix in [
        "",
        "relative",
        "/",
        "////",
        "/tmp/./unsafe",
        "/tmp/../unsafe",
        "/tmp/\nunsafe",
    ] {
        fs::write(&log, []).unwrap();
        let output = run_installer(
            &fixture,
            &root,
            &bin,
            &log,
            "Linux",
            "x86_64",
            &["--backend", "cpu", "--prefix", prefix],
        );
        assert_eq!(
            output.status.code(),
            Some(2),
            "unsafe prefix accepted: {prefix:?}"
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("invalid install prefix"));
        assert!(fs::read(&log).unwrap().is_empty());
    }
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn installer_reports_missing_prerequisite_before_build() {
    let (fixture, root, _, log) = installer_fixture("installer missing prerequisite");
    let isolated = fixture.join("isolated bin");
    fs::create_dir(&isolated).unwrap();
    for tool in ["bash", "uname", "install", "cargo"] {
        write_executable(&isolated.join(tool), b"#!/bin/sh\nexit 0\n");
    }
    let prefix = fixture.join("prefix");
    let output = Command::new("/bin/bash")
        .arg(root.join("support/install.sh"))
        .args(["--backend", "cpu", "--prefix", prefix.to_str().unwrap()])
        .env("PATH", &isolated)
        .env("HOME", fixture.join("home"))
        .env("GRAPH_HORIZON_TEST_LOG", &log)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("npm is required"));
    assert!(!log.exists());
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn installer_reports_missing_path_without_mutating_shells() {
    let (fixture, root, bin, log) = installer_fixture("installer path report");
    let home = fixture.join("home");
    let prefix = fixture.join("prefix with spaces");
    fs::create_dir(&home).unwrap();
    fs::write(home.join(".zshrc"), b"unchanged\n").unwrap();
    let prefix_with_slashes = format!("{}///", prefix.display());
    let output = run_installer(
        &fixture,
        &root,
        &bin,
        &log,
        "Linux",
        "x86_64",
        &["--backend", "cpu", "--prefix", &prefix_with_slashes],
    );
    assert!(
        output.status.success(),
        "stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains(&format!(
        "installed graph-horizon (prefix={}, backend=cpu, profile=release)",
        prefix.display()
    )));
    assert!(stdout.contains(&format!(
        "install: {}/bin is not in PATH; add it manually",
        prefix.display()
    )));
    assert_eq!(fs::read(home.join(".zshrc")).unwrap(), b"unchanged\n");
    assert!(prefix.join("bin/graph-horizon").is_file());
    assert_eq!(
        fs::read_link(prefix.join("bin/gh-zero-engine")).unwrap(),
        Path::new("graph-horizon")
    );
    assert_eq!(
        fs::read_to_string(&log)
            .unwrap()
            .lines()
            .collect::<Vec<_>>(),
        [
            "npm\tci",
            "npm\trun build",
            "cargo\tbuild --locked --no-default-features --features cpu --profile release -p graph-horizon"
        ]
    );
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn installer_requires_profile_and_preflights_metal() {
    let fixture = fixture_dir("installer preflight");
    let bin = fixture.join("bin");
    let mutation = fixture.join("build tool called");
    fs::create_dir(&bin).unwrap();
    fs::write(
        bin.join("uname"),
        b"#!/usr/bin/env bash\nprintf 'Linux\\n'\n",
    )
    .unwrap();
    for tool in ["cargo", "npm"] {
        fs::write(
            bin.join(tool),
            b"#!/usr/bin/env bash\nprintf called > \"$GRAPH_HORIZON_MUTATION_LOG\"\n",
        )
        .unwrap();
    }
    for tool in ["uname", "cargo", "npm"] {
        let mut permissions = fs::metadata(bin.join(tool)).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(bin.join(tool), permissions).unwrap();
    }
    let output = Command::new("/bin/bash")
        .arg(repository().join("support/install.sh"))
        .args(["--backend", "metal"])
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("GRAPH_HORIZON_MUTATION_LOG", &mutation)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("Metal requires macOS on arm64")
    );
    assert!(!mutation.exists());

    let source = fs::read_to_string(repository().join("support/install.sh")).unwrap();
    assert!(source.contains("cpu|vulkan|vulkan-hybrid|metal|metal-hybrid"));
    assert!(source.find("xcrun -f metallib").unwrap() < source.find("npm ci").unwrap());
    assert!(!source.contains("sudo"));
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn support_scripts_preserve_quoted_model_paths_and_model_bytes() {
    let fixture = fixture_dir("quoted");
    let bin = fixture.join("bin");
    fs::create_dir(&bin).unwrap();
    let cargo = bin.join("cargo");
    fs::write(
        &cargo,
        b"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$GRAPH_HORIZON_TEST_ARGS\"\n",
    )
    .unwrap();
    let mut permissions = fs::metadata(&cargo).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&cargo, permissions).unwrap();

    let model = fixture.join("model with spaces.gguf");
    let original = b"immutable GGUF fixture";
    fs::write(&model, original).unwrap();
    let log = fixture.join("arguments");
    let path = format!("{}:/usr/bin:/bin", bin.display());
    let output = Command::new("bash")
        .arg(repository().join("support/profiling/profile.sh"))
        .args([
            "--model",
            model.to_str().unwrap(),
            "--backend",
            "cpu",
            "--context",
            "1",
            "--kv",
            "f16",
        ])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEST_ARGS", &log)
        .output()
        .unwrap();
    assert!(output.status.success());
    let arguments = fs::read_to_string(&log).unwrap();
    assert_eq!(
        arguments
            .lines()
            .filter(|arg| *arg == model.to_str().unwrap())
            .count(),
        1,
        "model path was split or omitted"
    );
    assert_eq!(fs::read(&model).unwrap(), original);

    assert_scripts_quote_model_variables(&repository());
    fs::remove_dir_all(fixture).unwrap();
}

fn assert_scripts_quote_model_variables(root: &Path) {
    for relative in [
        "support/profiling/profile.sh",
        "support/profiling/validate-kv.sh",
        "support/profiling/validate-weights.sh",
        "support/testing/matrix-check.sh",
        "support/testing/parity-check.sh",
        "support/testing/semantic-check.sh",
        "support/testing/run-graph-horizon.sh",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert!(
            source.contains("\"$model\""),
            "{relative} does not quote model"
        );
        for forbidden in ["eval ", "wget ", "git clone"] {
            assert!(
                !source.contains(forbidden),
                "{relative} contains {forbidden}"
            );
        }
        if relative != "support/testing/parity-check.sh" {
            assert!(!source.contains("curl "), "{relative} contains curl");
        }
    }
}

#[test]
fn q4_profiling_contract() {
    let fixture = fixture_dir("q4 profiling");
    let models = fixture.join("-models with spaces");
    let bin = fixture.join("bin");
    let log = fixture.join("calls");
    fs::create_dir(&models).unwrap();
    fs::create_dir(&bin).unwrap();

    let rows = parse_catalog(include_str!("../support/models.tsv")).unwrap();
    for row in &rows {
        fs::write(models.join(row.q4_file), b"read-only model fixture").unwrap();
    }

    let cargo = bin.join("cargo");
    let stat = bin.join("stat");
    let sha256sum = bin.join("sha256sum");
    fs::write(
        &cargo,
        r#"#!/usr/bin/env bash
set -eu
model=""
for argument in "$@"; do
    case "$argument" in *.gguf) model="$argument" ;; esac
done
last="${!#}"
printf '%s\t%s\n' "$model" "$last" >> "$GRAPH_HORIZON_TEST_LOG"
case " $* " in *" --example inspect "*) ;; *) exit 0 ;; esac
if [[ -n "${GRAPH_HORIZON_BAD_INSPECT:-}" && "$model" == *"$GRAPH_HORIZON_BAD_INSPECT"* ]]; then
    printf 'weight_profile: Q4_K_M\n'
    exit 0
fi
printf 'weight_profile: Q4_K_M\n'
case "$model" in
    *-3B-*) printf 'dimensions: blocks=26 hidden=3072 q=4096 k=1024 v=1024 ffn=9216 context=262144\noutput: tied-to-embedding\ntensor_histogram:\n  F32: 53\n  Q4_K: 156\n  Q6_K: 27\n' ;;
    *-8B-*) printf 'dimensions: blocks=34 hidden=4096 q=4096 k=1024 v=1024 ffn=14336 context=262144\noutput: dedicated\ntensor_histogram:\n  F32: 69\n  Q4_K: 205\n  Q6_K: 35\n' ;;
    *-14B-*) printf 'dimensions: blocks=40 hidden=5120 q=4096 k=1024 v=1024 ffn=16384 context=262144\noutput: dedicated\ntensor_histogram:\n  F32: 81\n  Q4_K: 241\n  Q6_K: 41\n' ;;
esac
"#,
    )
    .unwrap();
    fs::write(
        &stat,
        r#"#!/usr/bin/env bash
set -eu
model="${!#}"
if [[ -n "${GRAPH_HORIZON_BAD_SIZE:-}" && "$model" == *"$GRAPH_HORIZON_BAD_SIZE"* ]]; then
    echo 1
    exit 0
fi
case "$model" in
    *3B-Instruct*) echo 2147023008 ;; *3B-Reasoning*) echo 2147021472 ;;
    *8B-Instruct*) echo 5198911904 ;; *8B-Reasoning*) echo 5198910368 ;;
    *14B-Instruct*) echo 8239593024 ;; *14B-Reasoning*) echo 8239591488 ;;
esac
"#,
    )
    .unwrap();
    fs::write(
        &sha256sum,
        r#"#!/usr/bin/env bash
set -eu
model="${!#}"
case "$model" in
    *3B-Instruct*) digest=9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8 ;;
    *3B-Reasoning*) digest=7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc ;;
    *8B-Instruct*) digest=33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761 ;;
    *8B-Reasoning*) digest=894aa3645ef8708a81dbe201c26105ce37c4c741252c89c5a78f81b49ac438c6 ;;
    *14B-Instruct*) digest=824e0f3373e69b84f2cae46fdcb9bd1ebc6ab3bfc7acc125d818b7b8178cc613 ;;
    *14B-Reasoning*) digest=fe08ca2158cd7438211ec6a4e5256d31bc980f016e3f5b635fe91fe6848d461c ;;
esac
printf '%s  %s\n' "$digest" "$model"
"#,
    )
    .unwrap();
    for executable in [&cargo, &stat, &sha256sum] {
        let mut permissions = fs::metadata(executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(executable, permissions).unwrap();
    }

    let path = format!("{}:/usr/bin:/bin", bin.display());
    let model = models.join(rows[0].q4_file);
    let kv = Command::new("bash")
        .arg(repository().join("support/profiling/validate-kv.sh"))
        .args([
            "--model",
            model.to_str().unwrap(),
            "--backend",
            "cpu",
            "--context",
            "4096",
        ])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEST_LOG", &log)
        .output()
        .unwrap();
    assert!(
        kv.status.success(),
        "{}",
        String::from_utf8_lossy(&kv.stderr)
    );
    let calls = fs::read_to_string(&log).unwrap();
    assert_eq!(calls.lines().count(), 2);
    assert!(calls.contains(&format!("{}\tf16", model.display())));
    assert!(calls.contains(&format!("{}\tint8", model.display())));

    fs::write(&log, []).unwrap();
    let weights = Command::new("bash")
        .arg(repository().join("support/profiling/validate-weights.sh"))
        .args(["--models-dir", models.to_str().unwrap()])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEST_LOG", &log)
        .output()
        .unwrap();
    assert!(
        weights.status.success(),
        "{}",
        String::from_utf8_lossy(&weights.stderr)
    );
    let stdout = String::from_utf8(weights.stdout).unwrap();
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.ends_with(": pass"))
            .count(),
        6
    );
    assert!(stdout.contains("summary: pass=6 external_verification=0 total=6"));
    assert_eq!(fs::read_to_string(&log).unwrap().lines().count(), 6);

    fs::write(&log, []).unwrap();
    let mismatch = Command::new("bash")
        .arg(repository().join("support/profiling/validate-weights.sh"))
        .args(["--models-dir", models.to_str().unwrap()])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEST_LOG", &log)
        .env("GRAPH_HORIZON_BAD_SIZE", "3B-Instruct")
        .output()
        .unwrap();
    assert_eq!(mismatch.status.code(), Some(1));
    assert!(
        String::from_utf8(mismatch.stderr)
            .unwrap()
            .contains("3b-instruct byte count mismatch")
    );
    assert!(!log.exists() || fs::read_to_string(&log).unwrap().is_empty());

    fs::remove_file(models.join(rows[1].q4_file)).unwrap();
    fs::write(&log, []).unwrap();
    let missing_weight = Command::new("bash")
        .arg(repository().join("support/profiling/validate-weights.sh"))
        .args(["--models-dir", models.to_str().unwrap()])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEST_LOG", &log)
        .output()
        .unwrap();
    assert!(missing_weight.status.success());
    let stdout = String::from_utf8(missing_weight.stdout).unwrap();
    assert!(
        stdout.contains("3b-reasoning: external verification: artifact is missing or unreadable")
    );
    assert_eq!(fs::read_to_string(&log).unwrap().lines().count(), 5);
    fs::write(models.join(rows[1].q4_file), b"read-only model fixture").unwrap();

    fs::write(&log, []).unwrap();
    let malformed = Command::new("bash")
        .arg(repository().join("support/profiling/validate-weights.sh"))
        .args(["--models-dir", models.to_str().unwrap()])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEST_LOG", &log)
        .env("GRAPH_HORIZON_BAD_INSPECT", "14B-Reasoning")
        .output()
        .unwrap();
    assert_eq!(malformed.status.code(), Some(1));
    assert!(
        String::from_utf8(malformed.stderr)
            .unwrap()
            .contains("14b-reasoning dimensions mismatch")
    );

    let missing = script(
        "support/profiling/validate-kv.sh",
        &[
            "--model",
            "/missing/q4.gguf",
            "--backend",
            "cpu",
            "--context",
            "4096",
        ],
    );
    assert!(missing.status.success());
    assert!(
        String::from_utf8(missing.stdout)
            .unwrap()
            .contains("Q4_K_M cpu: external verification: artifact is missing or unreadable")
    );

    let kv_source =
        fs::read_to_string(repository().join("support/profiling/validate-kv.sh")).unwrap();
    assert!(!kv_source.contains("--model-q8"));
    assert!(!kv_source.contains("--model-q4"));

    let copied_root = fixture.join("malformed catalog");
    fs::create_dir_all(copied_root.join("support/profiling")).unwrap();
    fs::copy(
        repository().join("support/profiling/validate-weights.sh"),
        copied_root.join("support/profiling/validate-weights.sh"),
    )
    .unwrap();
    fs::copy(
        repository().join("support/artifact.sh"),
        copied_root.join("support/artifact.sh"),
    )
    .unwrap();
    fs::write(
        copied_root.join("support/models.tsv"),
        include_str!("../support/models.tsv").replace('\n', "\r\n"),
    )
    .unwrap();
    let bad_catalog = Command::new("bash")
        .arg(copied_root.join("support/profiling/validate-weights.sh"))
        .args(["--models-dir", models.to_str().unwrap()])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEST_LOG", &log)
        .output()
        .unwrap();
    assert_eq!(bad_catalog.status.code(), Some(2));
    assert!(
        String::from_utf8(bad_catalog.stderr)
            .unwrap()
            .contains("catalog error")
    );

    for row in rows {
        assert_eq!(
            fs::read(models.join(row.q4_file)).unwrap(),
            b"read-only model fixture"
        );
    }
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn parity_script_contract() {
    use std::net::TcpListener;
    use std::os::unix::fs::symlink;
    use std::thread;
    use std::time::Duration;

    let fixture = fixture_dir("parity script");
    let models = fixture.join("-models with spaces");
    let bin = fixture.join("bin");
    let temp = fixture.join("owned temp");
    let cargo_log = fixture.join("cargo calls");
    let server_log = fixture.join("server lifecycle");
    let template_log = fixture.join("apply template request");
    fs::create_dir(&models).unwrap();
    fs::create_dir(&bin).unwrap();
    let model = models.join("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf");
    fs::write(&model, b"immutable parity fixture").unwrap();

    let stat = bin.join("stat");
    let sha256sum = bin.join("sha256sum");
    let curl = bin.join("curl");
    let cargo = bin.join("cargo");
    let mktemp = bin.join("mktemp");
    let server = fixture.join("llama server stub");
    fs::write(
        &stat,
        b"#!/usr/bin/env bash\nif [[ \"${GRAPH_HORIZON_BAD_SIZE:-}\" == 1 ]]; then echo 1; else echo 2147023008; fi\n",
    )
    .unwrap();
    fs::write(
        &sha256sum,
        b"#!/usr/bin/env bash\nprintf '9ed150d4367e68df0ac8e1540f6ddc65b42d0ee26378329d1ecbca60f93fc5f8  %s\\n' \"${!#}\"\n",
    )
    .unwrap();
    fs::write(
        &curl,
        r#"#!/usr/bin/env bash
set -eu
url="${!#}"
body=''
while (($#)); do
    if [[ "$1" == --data-binary ]]; then body="$2"; break; fi
    shift
done
case "$url" in
    */health)
        if [[ "${GRAPH_HORIZON_HEALTH_WAIT:-}" == 1 ]]; then /bin/sleep 0.2; exit 1; fi
        exit 0
        ;;
    */apply-template)
        if [[ -n "${GRAPH_HORIZON_APPLY_TEMPLATE_LOG:-}" ]]; then
            printf '%s\n' "$body" > "$GRAPH_HORIZON_APPLY_TEMPLATE_LOG"
        fi
        printf '{"prompt":"oracle prompt"}\n'
        ;;
    */tokenize)
        if [[ "${GRAPH_HORIZON_BAD_HTTP:-}" == tokenize ]]; then printf '{bad json\n'; else printf '{"tokens":[1,2,3]}\n'; fi
        ;;
    */completion) printf '{"tokens":[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]}\n' ;;
esac
"#,
    )
    .unwrap();
    fs::write(
        &cargo,
        r#"#!/usr/bin/env bash
set -eu
printf 'args=%s\nmodel=%s\ncontext=%s\nkv=%s\npercent=%s\nmode=%s\nprompt=%s\ncompletion=%s\n' \
    "$*" "$GRAPH_HORIZON_MODEL" "$GRAPH_HORIZON_CONTEXT" "$GRAPH_HORIZON_KV" \
    "${GRAPH_HORIZON_VRAM_WEIGHTS_PERCENT:-}" "${GRAPH_HORIZON_EXPECTED_MODE:-}" "$GRAPH_HORIZON_REFERENCE_PROMPT_IDS" \
    "$GRAPH_HORIZON_REFERENCE_COMPLETION_IDS" >> "$GRAPH_HORIZON_CARGO_LOG"
if [[ "${GRAPH_HORIZON_MEMORY_FAILURE:-}" == 1 ]]; then
    printf 'Vulkan memory is insufficient: required 2 bytes, available 1 bytes\n' >&2
    exit 1
fi
if [[ "${GRAPH_HORIZON_DEVICE_FAILURE:-}" == 1 ]]; then
    printf 'load selected runtime: Vulkan backend is unavailable\n' >&2
    exit 1
fi
printf 'test result: ok\nministral-parity: local_ids=30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45 oracle_top2=pass\n'
"#,
    )
    .unwrap();
    fs::write(
        &mktemp,
        b"#!/usr/bin/env bash\nmkdir -p -- \"$GRAPH_HORIZON_TEMP_DIR\"\nprintf '%s\\n' \"$GRAPH_HORIZON_TEMP_DIR\"\n",
    )
    .unwrap();
    fs::write(
        &server,
        r#"#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == --version ]]; then
    if [[ "${GRAPH_HORIZON_BAD_REVISION:-}" == 1 ]]; then echo 'llama.cpp old'; else echo 'llama.cpp 13f2b28b0'; fi
    exit 0
fi
printf 'started %s\n' "$$" >> "$GRAPH_HORIZON_SERVER_LOG"
trap 'printf "stopped %s\n" "$$" >> "$GRAPH_HORIZON_SERVER_LOG"; exit 0' TERM INT HUP
while :; do /bin/sleep 0.05; done
"#,
    )
    .unwrap();
    for executable in [&stat, &sha256sum, &curl, &cargo, &mktemp, &server] {
        let mut permissions = fs::metadata(executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(executable, permissions).unwrap();
    }

    let path = format!("{}:/usr/bin:/bin", bin.display());
    let free_port = || {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        port.to_string()
    };
    let base = [
        "--models-dir",
        models.to_str().unwrap(),
        "--model-id",
        "3b-instruct",
        "--backend",
        "vulkan-hybrid",
        "--kv",
        "int8",
        "--reference-server",
        server.to_str().unwrap(),
        "--weights-percent",
        "25",
        "--expect-mode",
        "mixed",
    ];

    let port = free_port();
    let pass = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &port])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEMP_DIR", &temp)
        .env("GRAPH_HORIZON_CARGO_LOG", &cargo_log)
        .env("GRAPH_HORIZON_SERVER_LOG", &server_log)
        .env("GRAPH_HORIZON_APPLY_TEMPLATE_LOG", &template_log)
        .output()
        .unwrap();
    assert!(
        pass.status.success(),
        "{}",
        String::from_utf8_lossy(&pass.stderr)
    );
    let stdout = String::from_utf8(pass.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1);
    assert!(
        stdout.starts_with("pass: model_id=3b-instruct backend=vulkan-hybrid kv=int8"),
        "unexpected parity output: {stdout:?}; stderr={:?}",
        String::from_utf8_lossy(&pass.stderr)
    );
    assert!(
        stdout.contains(
            "prompt_ids=1,2,3 oracle_ids=10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
        )
    );
    assert!(stdout.contains("local_ids=30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45"));
    assert!(!temp.exists());
    let cargo_call = fs::read_to_string(&cargo_log).unwrap();
    assert!(cargo_call.contains(
        "--features vulkan-hybrid --test family_agnostic real_selected_runtime_parity_and_lifecycle"
    ));
    assert!(cargo_call.contains("context=4096\nkv=int8\npercent=25\nmode=mixed"));
    assert!(cargo_call.contains(&format!("model={}", model.display())));
    assert_eq!(
        fs::read_to_string(&template_log).unwrap().trim_end(),
        r#"{"messages":[{"role":"system","content":""},{"role":"user","content":"Quanto fa 17 × 19?"}],"add_generation_prompt":true}"#
    );

    let endpoint_port = free_port();
    let endpoint = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args([
            "--models-dir",
            models.to_str().unwrap(),
            "--model-id",
            "3b-instruct",
            "--backend",
            "metal-hybrid",
            "--kv",
            "f16",
            "--reference-server",
            server.to_str().unwrap(),
            "--reference-port",
            &endpoint_port,
            "--weights-percent",
            "100",
            "--expect-mode",
            "all-metal",
        ])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEMP_DIR", &temp)
        .env("GRAPH_HORIZON_CARGO_LOG", &cargo_log)
        .env("GRAPH_HORIZON_SERVER_LOG", &server_log)
        .output()
        .unwrap();
    assert!(
        endpoint.status.success(),
        "{}",
        String::from_utf8_lossy(&endpoint.stderr)
    );
    assert!(
        String::from_utf8(endpoint.stdout)
            .unwrap()
            .starts_with("pass: model_id=3b-instruct backend=metal-hybrid kv=f16")
    );
    assert!(!temp.exists());

    let port = free_port();
    let malformed = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &port])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEMP_DIR", &temp)
        .env("GRAPH_HORIZON_CARGO_LOG", &cargo_log)
        .env("GRAPH_HORIZON_SERVER_LOG", &server_log)
        .env("GRAPH_HORIZON_APPLY_TEMPLATE_LOG", &template_log)
        .env("GRAPH_HORIZON_BAD_HTTP", "tokenize")
        .output()
        .unwrap();
    assert_eq!(malformed.status.code(), Some(1));
    assert!(
        String::from_utf8(malformed.stderr)
            .unwrap()
            .contains("malformed tokenize response")
    );
    assert!(!temp.exists());

    let revision_port = free_port();
    let revision = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &revision_port])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_BAD_REVISION", "1")
        .output()
        .unwrap();
    assert!(revision.status.success());
    assert_eq!(
        String::from_utf8(revision.stdout).unwrap().trim(),
        "external verification: unsupported llama.cpp revision"
    );

    let device = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &free_port()])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEMP_DIR", &temp)
        .env("GRAPH_HORIZON_CARGO_LOG", &cargo_log)
        .env("GRAPH_HORIZON_SERVER_LOG", &server_log)
        .env("GRAPH_HORIZON_DEVICE_FAILURE", "1")
        .output()
        .unwrap();
    assert!(device.status.success());
    assert_eq!(
        String::from_utf8(device.stdout).unwrap().trim(),
        "external verification: vulkan-hybrid backend unavailable"
    );
    assert!(!temp.exists());

    let server_calls = fs::read_to_string(&server_log).unwrap().lines().count();
    let cargo_calls = fs::read_to_string(&cargo_log).unwrap().lines().count();
    let mismatch = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &free_port()])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_BAD_SIZE", "1")
        .output()
        .unwrap();
    assert_eq!(mismatch.status.code(), Some(1));
    assert!(
        String::from_utf8(mismatch.stderr)
            .unwrap()
            .contains("3b-instruct byte count mismatch")
    );
    assert_eq!(
        fs::read_to_string(&server_log).unwrap().lines().count(),
        server_calls
    );
    assert_eq!(
        fs::read_to_string(&cargo_log).unwrap().lines().count(),
        cargo_calls
    );

    let memory_port = free_port();
    let memory = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &memory_port])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEMP_DIR", &temp)
        .env("GRAPH_HORIZON_CARGO_LOG", &cargo_log)
        .env("GRAPH_HORIZON_SERVER_LOG", &server_log)
        .env("GRAPH_HORIZON_MEMORY_FAILURE", "1")
        .output()
        .unwrap();
    assert!(memory.status.success());
    assert_eq!(
        String::from_utf8(memory.stdout).unwrap().trim(),
        "external verification: insufficient memory for vulkan-hybrid row"
    );
    assert!(!temp.exists());

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let occupied_port = listener.local_addr().unwrap().port().to_string();
    let occupied = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &occupied_port])
        .env("PATH", &path)
        .output()
        .unwrap();
    assert_eq!(occupied.status.code(), Some(2));
    assert!(
        String::from_utf8(occupied.stderr)
            .unwrap()
            .contains("reference port is occupied")
    );
    drop(listener);

    let port = free_port();
    let mut signalled = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &port])
        .env("PATH", &path)
        .env("GRAPH_HORIZON_TEMP_DIR", &temp)
        .env("GRAPH_HORIZON_CARGO_LOG", &cargo_log)
        .env("GRAPH_HORIZON_SERVER_LOG", &server_log)
        .env("GRAPH_HORIZON_HEALTH_WAIT", "1")
        .spawn()
        .unwrap();
    for _ in 0..100 {
        if temp.exists()
            && fs::read_to_string(&server_log)
                .unwrap_or_default()
                .contains("started")
        {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    assert!(temp.exists());
    assert!(
        Command::new("kill")
            .args(["-TERM", &signalled.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(signalled.wait().unwrap().code(), Some(130));
    assert!(!temp.exists());
    assert!(fs::read_to_string(&server_log).unwrap().contains("stopped"));

    let tool_dir = fixture.join("missing curl path");
    fs::create_dir(&tool_dir).unwrap();
    symlink("/usr/bin/dirname", tool_dir.join("dirname")).unwrap();
    symlink("/usr/bin/awk", tool_dir.join("awk")).unwrap();
    let missing_tool_port = free_port();
    let missing_tool = Command::new("/bin/bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &missing_tool_port])
        .env("PATH", &tool_dir)
        .output()
        .unwrap();
    assert!(missing_tool.status.success());
    assert_eq!(
        String::from_utf8(missing_tool.stdout).unwrap().trim(),
        "external verification: curl unavailable"
    );

    let invalid_id = script(
        "support/testing/parity-check.sh",
        &[
            "--models-dir",
            models.to_str().unwrap(),
            "--model-id",
            "unknown",
            "--backend",
            "cpu",
            "--kv",
            "f16",
            "--reference-server",
            server.to_str().unwrap(),
        ],
    );
    assert_eq!(invalid_id.status.code(), Some(2));

    let source = fs::read_to_string(repository().join("support/testing/parity-check.sh")).unwrap();
    for fixed in [
        "--ctx-size 4096",
        "GRAPH_HORIZON_CONTEXT=4096",
        "GRAPH_HORIZON_VRAM_WEIGHTS_PERCENT",
        "GRAPH_HORIZON_EXPECTED_MODE",
        "n_predict:16",
        "--host 127.0.0.1",
        "--offline",
        "--device none",
        "--n-gpu-layers 0",
        "--no-kv-offload",
        "--no-warmup",
        "--ignore-eos",
        "cargo test --locked --release",
        "--exact",
    ] {
        assert!(source.contains(fixed), "missing fixed contract: {fixed}");
    }
    assert!(!source.contains("--context)"));
    assert!(source.contains("--weights-percent)"));
    assert!(!source.contains("eval "));
    assert_eq!(fs::read(&model).unwrap(), b"immutable parity fixture");
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn matrix_runs_seventy_four_exact_rows() {
    use std::collections::HashSet;
    use std::net::TcpListener;

    let fixture = fixture_dir("matrix script");
    let copied_root = fixture.join("copied repository");
    let testing = copied_root.join("support/testing");
    let models = fixture.join("-models with spaces");
    let bin = fixture.join("bin");
    let parity_log = fixture.join("parity calls");
    let inspect_log = fixture.join("inspect calls");
    let server = fixture.join("reference server");
    fs::create_dir_all(&testing).unwrap();
    fs::create_dir_all(&models).unwrap();
    fs::create_dir(&bin).unwrap();
    fs::create_dir_all(copied_root.join("support")).unwrap();
    fs::copy(
        repository().join("support/testing/matrix-check.sh"),
        testing.join("matrix-check.sh"),
    )
    .unwrap();
    fs::write(
        copied_root.join("support/models.tsv"),
        include_str!("../support/models.tsv"),
    )
    .unwrap();
    fs::copy(
        repository().join("support/artifact.sh"),
        copied_root.join("support/artifact.sh"),
    )
    .unwrap();
    fs::write(&server, b"reference fixture").unwrap();

    let rows = parse_catalog(include_str!("../support/models.tsv")).unwrap();
    for row in &rows {
        fs::write(models.join(row.q8_file), b"Q8 rejection fixture").unwrap();
    }

    let parity = testing.join("parity-check.sh");
    fs::write(
        &parity,
        r#"#!/usr/bin/env bash
set -eu
id=""; backend=""; kv=""; percent=""; mode=""
while (($#)); do
    case "$1" in
        --model-id) id="$2"; shift 2 ;; --backend) backend="$2"; shift 2 ;;
        --kv) kv="$2"; shift 2 ;; --weights-percent) percent="$2"; shift 2 ;;
        --expect-mode) mode="$2"; shift 2 ;; *) shift ;;
    esac
done
key="$id:$backend:$kv:$percent:$mode"
printf '%s\n' "$key" >> "$GRAPH_HORIZON_PARITY_LOG"
if [[ "$key" == "${GRAPH_HORIZON_FAIL_KEY:-}" ]]; then echo 'injected failure' >&2; exit 1; fi
if [[ "$key" == "${GRAPH_HORIZON_EXTERNAL_KEY:-}" ]]; then echo 'external verification: injected resource unavailable'; exit 0; fi
ids='3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18'
if [[ "$key" == "${GRAPH_HORIZON_MISMATCH_KEY:-}" ]]; then ids='3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,99'; fi
printf 'pass: model_id=%s backend=%s kv=%s prompt_ids=1 oracle_ids=2 local_ids=%s\n' "$id" "$backend" "$kv" "$ids"
"#,
    )
    .unwrap();
    let cargo = bin.join("cargo");
    fs::write(
        &cargo,
        r#"#!/usr/bin/env bash
set -eu
model="${!#}"
printf '%s\n' "$model" >> "$GRAPH_HORIZON_INSPECT_LOG"
if [[ -n "${GRAPH_HORIZON_BAD_Q8:-}" && "$model" == *"$GRAPH_HORIZON_BAD_Q8"* ]]; then
    echo 'placement: cpu'
    exit 0
fi
echo "E04 unsupported GGUF weight profile 'Q8_0'; supported profile: Q4_K_M" >&2
exit 1
"#,
    )
    .unwrap();
    for executable in [testing.join("matrix-check.sh"), parity, cargo] {
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(executable, permissions).unwrap();
    }

    let matrix = testing.join("matrix-check.sh");
    let path = format!("{}:/usr/bin:/bin", bin.display());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port().to_string();
    drop(listener);
    let base = [
        "--models-dir",
        models.to_str().unwrap(),
        "--reference-server",
        server.to_str().unwrap(),
        "--reference-port",
        &port,
    ];
    let complete = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GRAPH_HORIZON_PARITY_LOG", &parity_log)
        .env("GRAPH_HORIZON_INSPECT_LOG", &inspect_log)
        .output()
        .unwrap();
    assert!(
        complete.status.success(),
        "{}",
        String::from_utf8_lossy(&complete.stderr)
    );
    let stdout = String::from_utf8(complete.stdout).unwrap();
    let statuses = stdout
        .lines()
        .filter(|line| !line.starts_with("summary:"))
        .collect::<Vec<_>>();
    assert_eq!(statuses.len(), 74);
    assert_eq!(
        statuses
            .iter()
            .filter(|line| line.starts_with("q8 "))
            .count(),
        6
    );
    assert_eq!(
        statuses
            .iter()
            .filter(|line| line.starts_with("parity "))
            .count(),
        68
    );
    assert_eq!(statuses.iter().copied().collect::<HashSet<_>>().len(), 74);
    assert!(stdout.contains("summary: pass=74 external_verification=0 failure=0 total=74"));
    assert_eq!(fs::read_to_string(&inspect_log).unwrap().lines().count(), 6);

    let expected = rows
        .iter()
        .flat_map(|row| {
            ["cpu", "vulkan", "vulkan-hybrid", "metal", "metal-hybrid"]
                .into_iter()
                .flat_map(move |backend| {
                    ["f16", "int8"].into_iter().map(move |kv| match backend {
                        "vulkan-hybrid" | "metal-hybrid" => {
                            format!("{}:{backend}:{kv}:25:mixed", row.id)
                        }
                        _ => format!("{}:{backend}:{kv}::", row.id),
                    })
                })
        })
        .chain(
            [("vulkan-hybrid", "all-gpu"), ("metal-hybrid", "all-metal")]
                .into_iter()
                .flat_map(|(backend, all_mode)| {
                    [(all_mode, "100"), ("cpu-only", "0")].into_iter().flat_map(
                        move |(mode, percent)| {
                            ["f16", "int8"].into_iter().map(move |kv| {
                                format!("3b-instruct:{backend}:{kv}:{percent}:{mode}")
                            })
                        },
                    )
                }),
        )
        .collect::<Vec<_>>();
    let calls = fs::read_to_string(&parity_log).unwrap();
    assert_eq!(calls.lines().collect::<Vec<_>>(), expected);

    fs::remove_file(models.join(rows[0].q8_file)).unwrap();
    fs::write(&parity_log, []).unwrap();
    fs::write(&inspect_log, []).unwrap();
    let continued = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GRAPH_HORIZON_PARITY_LOG", &parity_log)
        .env("GRAPH_HORIZON_INSPECT_LOG", &inspect_log)
        .env("GRAPH_HORIZON_EXTERNAL_KEY", "3b-instruct:cpu:f16::")
        .output()
        .unwrap();
    assert!(continued.status.success());
    let stdout = String::from_utf8(continued.stdout).unwrap();
    assert!(stdout.contains("q8 model_id=3b-instruct: external verification"));
    assert!(
        stdout.contains("parity model_id=3b-instruct backend=cpu kv=f16: external verification")
    );
    assert!(stdout.contains("summary: pass=72 external_verification=2 failure=0 total=74"));
    assert_eq!(fs::read_to_string(&parity_log).unwrap().lines().count(), 68);
    fs::write(models.join(rows[0].q8_file), b"Q8 rejection fixture").unwrap();

    fs::write(&parity_log, []).unwrap();
    fs::write(&inspect_log, []).unwrap();
    let stopped = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GRAPH_HORIZON_PARITY_LOG", &parity_log)
        .env("GRAPH_HORIZON_INSPECT_LOG", &inspect_log)
        .env("GRAPH_HORIZON_FAIL_KEY", "3b-instruct:cpu:int8::")
        .output()
        .unwrap();
    assert_eq!(stopped.status.code(), Some(1));
    let stdout = String::from_utf8(stopped.stdout).unwrap();
    assert!(stdout.contains("parity model_id=3b-instruct backend=cpu kv=int8: failure"));
    assert!(stdout.contains("summary: pass=7 external_verification=0 failure=1 total=8"));
    assert_eq!(fs::read_to_string(&parity_log).unwrap().lines().count(), 2);

    fs::write(&parity_log, []).unwrap();
    fs::write(&inspect_log, []).unwrap();
    let mismatch = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GRAPH_HORIZON_PARITY_LOG", &parity_log)
        .env("GRAPH_HORIZON_INSPECT_LOG", &inspect_log)
        .env(
            "GRAPH_HORIZON_MISMATCH_KEY",
            "3b-instruct:vulkan-hybrid:int8:100:all-gpu",
        )
        .output()
        .unwrap();
    assert_eq!(mismatch.status.code(), Some(1));
    let stdout = String::from_utf8(mismatch.stdout).unwrap();
    assert!(
        stdout.contains("backend=vulkan-hybrid kv=int8 weights_percent=100 mode=all-gpu: failure")
    );
    assert!(stdout.contains("summary: pass=67 external_verification=0 failure=1 total=68"));
    assert!(!stdout.contains(models.to_str().unwrap()));

    fs::write(&parity_log, []).unwrap();
    fs::write(&inspect_log, []).unwrap();
    let bad_q8 = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GRAPH_HORIZON_PARITY_LOG", &parity_log)
        .env("GRAPH_HORIZON_INSPECT_LOG", &inspect_log)
        .env("GRAPH_HORIZON_BAD_Q8", "3B-Instruct")
        .output()
        .unwrap();
    assert_eq!(bad_q8.status.code(), Some(1));
    let stdout = String::from_utf8(bad_q8.stdout).unwrap();
    assert!(stdout.contains("q8 model_id=3b-instruct: failure"));
    assert!(stdout.contains("summary: pass=0 external_verification=0 failure=1 total=1"));
    assert!(!parity_log.exists() || fs::read_to_string(&parity_log).unwrap().is_empty());

    fs::write(
        copied_root.join("support/models.tsv"),
        include_str!("../support/models.tsv").replace('\n', "\r\n"),
    )
    .unwrap();
    let malformed = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GRAPH_HORIZON_PARITY_LOG", &parity_log)
        .env("GRAPH_HORIZON_INSPECT_LOG", &inspect_log)
        .output()
        .unwrap();
    assert_eq!(malformed.status.code(), Some(2));
    assert!(
        String::from_utf8(malformed.stderr)
            .unwrap()
            .contains("catalog error")
    );

    let invalid = Command::new(&matrix).arg("--unknown").output().unwrap();
    assert_eq!(invalid.status.code(), Some(2));
    let source = fs::read_to_string(repository().join("support/testing/matrix-check.sh")).unwrap();
    assert!(!source.contains("&\n"));
    assert!(!source.contains("eval "));
    assert!(source.contains("for backend in cpu vulkan vulkan-hybrid metal metal-hybrid"));
    assert!(source.contains("for kv in f16 int8"));
    assert!(source.contains("vulkan-hybrid:vulkan:all-gpu metal-hybrid:metal:all-metal"));
    assert_eq!(fs::read(&server).unwrap(), b"reference fixture");
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn matrix_rejects_homogeneous_endpoint_sequence_mismatch() {
    let source = fs::read_to_string(repository().join("support/testing/matrix-check.sh")).unwrap();
    assert!(source.contains("homogeneous endpoint local ID mismatch"));
    assert!(source.contains("local_ids=([0-9]+(,[0-9]+){15})"));
}

#[test]
fn semantic_script_contract() {
    use std::collections::HashSet;

    let fixture = fixture_dir("semantic script");
    let copied_root = fixture.join("copied repository");
    let testing = copied_root.join("support/testing");
    let models = fixture.join("-models with spaces");
    let bin = fixture.join("bin");
    let calls = fixture.join("cargo calls");
    let tool_calls = fixture.join("tool calls");
    fs::create_dir_all(&testing).unwrap();
    fs::create_dir_all(&models).unwrap();
    fs::create_dir(&bin).unwrap();
    fs::copy(
        repository().join("support/testing/semantic-check.sh"),
        testing.join("semantic-check.sh"),
    )
    .unwrap();
    fs::write(
        copied_root.join("support/models.tsv"),
        include_str!("../support/models.tsv"),
    )
    .unwrap();
    fs::copy(
        repository().join("support/artifact.sh"),
        copied_root.join("support/artifact.sh"),
    )
    .unwrap();

    let rows = parse_catalog(include_str!("../support/models.tsv")).unwrap();
    for row in &rows {
        fs::write(models.join(row.q4_file), b"immutable semantic fixture").unwrap();
    }
    let stat = bin.join("stat");
    let sha256sum = bin.join("sha256sum");
    let cargo = bin.join("cargo");
    fs::write(
        &stat,
        r#"#!/usr/bin/env bash
set -eu
model="${!#}"
printf 'stat\t%s\n' "$model" >> "$SEMANTIC_STUB_TOOL_LOG"
case "$model" in *Instruct*) echo "unexpected instruct stat" >&2; exit 9 ;; esac
if [[ -n "${SEMANTIC_STUB_BAD_SIZE:-}" && "$model" == *"$SEMANTIC_STUB_BAD_SIZE"* ]]; then echo 1; exit 0; fi
case "$model" in
    *3B-Reasoning*) echo 2147021472 ;;
    *8B-Reasoning*) echo 5198910368 ;;
    *14B-Reasoning*) echo 8239591488 ;;
esac
"#,
    )
    .unwrap();
    fs::write(
        &sha256sum,
        r#"#!/usr/bin/env bash
set -eu
model="${!#}"
printf 'sha256sum\t%s\n' "$model" >> "$SEMANTIC_STUB_TOOL_LOG"
case "$model" in *Instruct*) echo "unexpected instruct sha256sum" >&2; exit 9 ;; esac
case "$model" in
    *3B-Reasoning*) digest=7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc ;;
    *8B-Reasoning*) digest=894aa3645ef8708a81dbe201c26105ce37c4c741252c89c5a78f81b49ac438c6 ;;
    *14B-Reasoning*) digest=fe08ca2158cd7438211ec6a4e5256d31bc980f016e3f5b635fe91fe6848d461c ;;
esac
if [[ -n "${SEMANTIC_STUB_BAD_SHA:-}" && "$model" == *"$SEMANTIC_STUB_BAD_SHA"* ]]; then digest=0000000000000000000000000000000000000000000000000000000000000000; fi
printf '%s  %s\n' "$digest" "$model"
"#,
    )
    .unwrap();
    fs::write(
        &cargo,
        r#"#!/usr/bin/env bash
set -eu
printf '%s\t%s\t%s\n' "$GRAPH_HORIZON_MODEL_ID" "$GRAPH_HORIZON_MODEL" "$*" >> "$SEMANTIC_STUB_LOG"
temp=0.7; [[ "${SEMANTIC_STUB_PROTOCOL:-}" != config-mismatch || "$GRAPH_HORIZON_MODEL_ID" != 3b-reasoning ]] || temp=0
printf 'semantic-config: model_id=%s context=4096 max_tokens=4096 temperature=%s top_p=1 top_k=0 min_p=0 repeat_penalty=1 seed=0 kv=f16\n' "$GRAPH_HORIZON_MODEL_ID" "$temp"
if [[ "$GRAPH_HORIZON_MODEL_ID" == "${SEMANTIC_STUB_EXTERNAL_ID:-}" ]]; then
    printf 'semantic-external: model_id=%s reason=no-full-vram-fit\n' "$GRAPH_HORIZON_MODEL_ID"
    exit 0
fi
printf 'semantic-selection: model_id=%s backend=vulkan reason=full-vram-fit probe_mode=all-gpu run_mode=all-gpu cpu_layers=0 gpu_layers=34 cpu_weights=0 cpu_kv=0 cpu_scratch=0 cpu_fixed=0 cpu_staging=0 cpu_crossing=0 cpu_reserve=0 cpu_total=0 gpu_weights=0 gpu_kv=0 gpu_scratch=0 gpu_fixed=0 gpu_staging=0 gpu_crossing=0 gpu_reserve=0 gpu_total=0\n' "$GRAPH_HORIZON_MODEL_ID"
ids=(S01 S02 S03 S04 S06 S07 S08 S09 S10)
critical=4; semantic=9; semantic_status=pass; markers=9; marker_status=pass; execution_status=pass
for case_id in "${ids[@]}"; do
    emitted_id="$case_id"; stop=eos; status=pass; marker=complete; details=
    if [[ "$GRAPH_HORIZON_MODEL_ID" == 3b-reasoning ]]; then
        case "${SEMANTIC_STUB_PROTOCOL:-}" in
            semantic-miss) [[ "$case_id" != S09 && "$case_id" != S10 ]] || { status=fail; semantic=7; semantic_status=fail; details=' reason=semantic-gate-miss excerpt=fixture'; } ;;
            incomplete) [[ "$case_id" != S01 ]] || { stop=context; status=fail; critical=3; semantic=8; semantic_status=fail; details=' reason=incomplete-generation excerpt=fixture'; } ;;
            invalid-marker) [[ "$case_id" != S02 ]] || { status=fail; marker=absent; critical=3; semantic=8; semantic_status=fail; markers=8; marker_status=fail; details=' reason=invalid-reasoning-markers excerpt=[invalid reasoning response omitted]'; } ;;
            engine-error) [[ "$case_id" != S03 ]] || { stop=error; status=fail; critical=3; semantic=8; semantic_status=fail; execution_status=fail; details=' reason=engine-error excerpt=[invalid reasoning response omitted]'; } ;;
            duplicate-case) [[ "$case_id" != S02 ]] || emitted_id=S01 ;;
        esac
    fi
    printf 'semantic-case: model_id=%s case_id=%s status=%s predicate=fixture class=semantic stop=%s prompt_tokens=16 completion_tokens=1 marker_status=%s%s\n' "$GRAPH_HORIZON_MODEL_ID" "$emitted_id" "$status" "$stop" "$marker" "$details"
done
printf 'semantic-summary: model_id=%s critical=%s/4 semantic=%s/9 semantic_status=%s reasoning_format=%s/9 reasoning_format_status=%s execution_status=%s\n' "$GRAPH_HORIZON_MODEL_ID" "$critical" "$semantic" "$semantic_status" "$markers" "$marker_status" "$execution_status"
[[ "${SEMANTIC_STUB_PROTOCOL:-}" == missing-timing && "$GRAPH_HORIZON_MODEL_ID" == 3b-reasoning ]] || printf 'semantic-timing: model_id=%s completed_cases=9 total_ms=100 prefill_ms=40 decode_ms=60\n' "$GRAPH_HORIZON_MODEL_ID"
printf 'internal /secret/path must not escape\n' >&2
[[ "$GRAPH_HORIZON_MODEL_ID" != "${SEMANTIC_STUB_FAIL_ID:-}" ]]
"#,
    )
    .unwrap();
    for executable in [testing.join("semantic-check.sh"), stat, sha256sum, cargo] {
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(executable, permissions).unwrap();
    }

    let runner = testing.join("semantic-check.sh");
    let path = format!("{}:/usr/bin:/bin", bin.display());
    let run = |variables: &[(&str, &str)]| {
        let mut command = Command::new(&runner);
        command
            .args(["--models-dir", models.to_str().unwrap()])
            .env("PATH", &path)
            .env("SEMANTIC_STUB_LOG", &calls)
            .env("SEMANTIC_STUB_TOOL_LOG", &tool_calls);
        for (name, value) in variables {
            command.env(name, value);
        }
        command.output().unwrap()
    };

    let complete = run(&[]);
    assert!(complete.status.success());
    let stdout = String::from_utf8(complete.stdout).unwrap();
    let statuses = stdout
        .lines()
        .filter(|line| line.starts_with("qualification:"))
        .collect::<Vec<_>>();
    assert_eq!(statuses.len(), 6);
    assert_eq!(statuses.iter().copied().collect::<HashSet<_>>().len(), 6);
    assert!(
        stdout.contains("summary: qualified=6 not_qualified=0 external_verification=0 total=6")
    );
    assert!(!stdout.contains("/secret/path"));
    assert_eq!(
        stdout
            .matches("profile=instruct evidence=preserved")
            .count(),
        3
    );
    assert_eq!(
        stdout.matches("profile=reasoning evidence=current").count(),
        3
    );
    assert_eq!(stdout.matches("semantic-config:").count(), 3);
    assert_eq!(stdout.matches("semantic-selection:").count(), 3);
    assert_eq!(stdout.matches("semantic-case:").count(), 27);
    assert_eq!(stdout.matches("case_id=S05").count(), 0);
    assert_eq!(stdout.matches("case_id=S11").count(), 0);
    assert_eq!(stdout.matches("case_id=S12").count(), 0);
    assert_eq!(stdout.matches("semantic-summary:").count(), 3);
    assert_eq!(stdout.matches("semantic-timing:").count(), 3);
    let cargo_calls = fs::read_to_string(&calls).unwrap();
    assert_eq!(cargo_calls.lines().count(), 3);
    for (line, row) in cargo_calls
        .lines()
        .zip(rows.iter().filter(|row| row.chat == "reasoning"))
    {
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(fields[0], row.id);
        assert_eq!(fields[1], models.join(row.q4_file).to_str().unwrap());
        assert!(fields[2].contains(
            "--no-default-features --features vulkan-hybrid --test semantic real_semantic_acceptance"
        ));
        assert!(fields[2].contains("--ignored --nocapture --exact"));
    }
    assert_eq!(fs::read_to_string(&tool_calls).unwrap().lines().count(), 6);

    fs::remove_file(models.join(rows[1].q4_file)).unwrap();
    fs::write(&calls, []).unwrap();
    fs::write(&tool_calls, []).unwrap();
    let missing = run(&[]);
    assert!(missing.status.success());
    let stdout = String::from_utf8(missing.stdout).unwrap();
    assert!(stdout.contains(
        "qualification: model_id=3b-reasoning profile=reasoning evidence=current status=external-verification reason=artifact-missing-or-unreadable critical=not-applicable semantic=not-applicable"
    ));
    assert!(
        stdout.contains("summary: qualified=5 not_qualified=0 external_verification=1 total=6")
    );
    assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 2);
    fs::write(models.join(rows[1].q4_file), b"immutable semantic fixture").unwrap();

    fs::write(&calls, []).unwrap();
    let mismatch = run(&[("SEMANTIC_STUB_BAD_SIZE", "3B-Reasoning")]);
    assert!(mismatch.status.success());
    assert!(
        String::from_utf8(mismatch.stdout)
            .unwrap()
            .contains("reason=byte-count-mismatch")
    );
    assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 2);

    fs::write(&calls, []).unwrap();
    let sha = run(&[("SEMANTIC_STUB_BAD_SHA", "8B-Reasoning")]);
    assert!(sha.status.success());
    assert!(
        String::from_utf8(sha.stdout)
            .unwrap()
            .contains("reason=sha256-mismatch")
    );

    for (protocol, reason) in [
        ("config-mismatch", "configuration-mismatch"),
        ("semantic-miss", "semantic-gate-miss"),
        ("incomplete", "incomplete-generation"),
        ("invalid-marker", "invalid-reasoning-markers"),
        ("engine-error", "engine-error"),
        ("missing-timing", "invalid-validation-protocol"),
        ("duplicate-case", "invalid-validation-protocol"),
    ] {
        fs::write(&calls, []).unwrap();
        let output = run(&[("SEMANTIC_STUB_PROTOCOL", protocol)]);
        assert!(output.status.success(), "{protocol}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(
            stdout.contains(&format!("status=not-qualified reason={reason}")),
            "{protocol}"
        );
        assert!(
            stdout.contains("summary: qualified=5 not_qualified=1 external_verification=0 total=6")
        );
        assert!(!stdout.contains("/secret/path"));
        assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 3);
    }

    fs::write(&calls, []).unwrap();
    let no_vram = run(&[("SEMANTIC_STUB_EXTERNAL_ID", "8b-reasoning")]);
    assert!(no_vram.status.success());
    let stdout = String::from_utf8(no_vram.stdout).unwrap();
    assert!(stdout.contains("reason=no-full-vram-fit"));
    assert!(
        stdout.contains("summary: qualified=5 not_qualified=0 external_verification=1 total=6")
    );

    fs::write(
        copied_root.join("support/models.tsv"),
        include_str!("../support/models.tsv").replace('\n', "\r\n"),
    )
    .unwrap();
    assert_eq!(
        Command::new(&runner)
            .args(["--models-dir", models.to_str().unwrap()])
            .env("PATH", &path)
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    fs::write(
        copied_root.join("support/models.tsv"),
        include_str!("../support/models.tsv"),
    )
    .unwrap();
    let no_cargo = Command::new(&runner)
        .args(["--models-dir", models.to_str().unwrap()])
        .env("PATH", "/usr/bin:/bin")
        .env("SEMANTIC_STUB_TOOL_LOG", &tool_calls)
        .output()
        .unwrap();
    if !Command::new("bash")
        .arg("-lc")
        .arg("PATH=/usr/bin:/bin command -v cargo")
        .status()
        .unwrap()
        .success()
    {
        let stdout = String::from_utf8(no_cargo.stdout).unwrap();
        assert!(stdout.contains("reason=cargo-unavailable"));
        assert!(
            stdout.contains("summary: qualified=3 not_qualified=0 external_verification=3 total=6")
        );
    }
    assert_eq!(
        Command::new(&runner)
            .arg("--unknown")
            .status()
            .unwrap()
            .code(),
        Some(2)
    );
    let source =
        fs::read_to_string(repository().join("support/testing/semantic-check.sh")).unwrap();
    for forbidden in ["curl ", "wget ", "eval ", "--features cpu", "pass=6"] {
        assert!(
            !source.contains(forbidden),
            "semantic runner contains {forbidden}"
        );
    }
    assert!(!source.contains("--features vulkan --test"));
    assert_eq!(source.matches("--features vulkan-hybrid").count(), 1);
    assert!(source.lines().count() <= 160);
    for row in &rows {
        assert_eq!(
            fs::read(models.join(row.q4_file)).unwrap(),
            b"immutable semantic fixture"
        );
    }
    fs::remove_dir_all(fixture).unwrap();
}
