/*
 * Support script acceptance tests
 * Single responsibility: exercise the retained shell scripts as external
 * interfaces, proving early validation, quoted model paths, read-only model
 * handling, and explicit not-verified output without invoking real builds.
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
        "gh-zero support scripts {} {label}",
        std::process::id()
    ));
    if path.exists() {
        fs::remove_dir_all(&path).expect("remove stale fixture");
    }
    fs::create_dir_all(&path).expect("create fixture");
    path
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

    for (relative, args) in [
        ("support/install.sh", vec!["--backend", "invalid"]),
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
        ("support/profiling/validate-weights.sh", vec!["--unknown"]),
        (
            "support/testing/run-ghzero-engine.sh",
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
fn support_scripts_preserve_quoted_model_paths_and_model_bytes() {
    let fixture = fixture_dir("quoted");
    let bin = fixture.join("bin");
    fs::create_dir(&bin).unwrap();
    let cargo = bin.join("cargo");
    fs::write(
        &cargo,
        b"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$GH_ZERO_TEST_ARGS\"\n",
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
        .env("GH_ZERO_TEST_ARGS", &log)
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
        "support/testing/parity-check.sh",
        "support/testing/run-ghzero-engine.sh",
    ] {
        let source = fs::read_to_string(root.join(relative)).unwrap();
        assert!(
            source.contains("\"$model\""),
            "{relative} does not quote model"
        );
        for forbidden in ["eval ", "curl ", "wget ", "git clone"] {
            assert!(
                !source.contains(forbidden),
                "{relative} contains {forbidden}"
            );
        }
    }
}

#[test]
fn support_scripts_report_missing_artifacts_as_not_verified() {
    let output = script(
        "support/profiling/validate-kv.sh",
        &[
            "--model-q8",
            "/missing/q8.gguf",
            "--model-q4",
            "/missing/q4.gguf",
            "--backend",
            "cpu",
            "--context",
            "4096",
        ],
    );
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Q8_0 cpu: not verified"));
    assert!(stdout.contains("Q4_K_M cpu: not verified"));
    assert!(stdout.contains("not verified: no pinned artifact is available"));
}
