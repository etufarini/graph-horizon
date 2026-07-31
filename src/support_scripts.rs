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
