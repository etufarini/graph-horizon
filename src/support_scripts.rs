/*
 * Support script acceptance tests
 * Single responsibility: exercise the retained shell scripts as external
 * interfaces through disposable fixtures, proving early validation, quoted
 * read-only model handling, class-sensitive semantic protocols, and explicit
 * external-verification output without real builds.
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
        ("support/testing/semantic-check.sh", vec!["--unknown"]),
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
        "support/testing/matrix-check.sh",
        "support/testing/parity-check.sh",
        "support/testing/semantic-check.sh",
        "support/testing/run-ghzero-engine.sh",
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
printf '%s\t%s\n' "$model" "$last" >> "$GH_ZERO_TEST_LOG"
case " $* " in *" --example inspect "*) ;; *) exit 0 ;; esac
if [[ -n "${GH_ZERO_BAD_INSPECT:-}" && "$model" == *"$GH_ZERO_BAD_INSPECT"* ]]; then
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
if [[ -n "${GH_ZERO_BAD_SIZE:-}" && "$model" == *"$GH_ZERO_BAD_SIZE"* ]]; then
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
        .env("GH_ZERO_TEST_LOG", &log)
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
        .env("GH_ZERO_TEST_LOG", &log)
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
        .env("GH_ZERO_TEST_LOG", &log)
        .env("GH_ZERO_BAD_SIZE", "3B-Instruct")
        .output()
        .unwrap();
    assert!(mismatch.status.success());
    let stdout = String::from_utf8(mismatch.stdout).unwrap();
    assert!(stdout.contains("3b-instruct: external verification: byte count mismatch"));
    assert!(stdout.contains("summary: pass=5 external_verification=1 total=6"));
    assert_eq!(fs::read_to_string(&log).unwrap().lines().count(), 5);

    fs::remove_file(models.join(rows[1].q4_file)).unwrap();
    fs::write(&log, []).unwrap();
    let missing_weight = Command::new("bash")
        .arg(repository().join("support/profiling/validate-weights.sh"))
        .args(["--models-dir", models.to_str().unwrap()])
        .env("PATH", &path)
        .env("GH_ZERO_TEST_LOG", &log)
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
        .env("GH_ZERO_TEST_LOG", &log)
        .env("GH_ZERO_BAD_INSPECT", "14B-Reasoning")
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
    fs::write(
        copied_root.join("support/models.tsv"),
        include_str!("../support/models.tsv").replace('\n', "\r\n"),
    )
    .unwrap();
    let bad_catalog = Command::new("bash")
        .arg(copied_root.join("support/profiling/validate-weights.sh"))
        .args(["--models-dir", models.to_str().unwrap()])
        .env("PATH", &path)
        .env("GH_ZERO_TEST_LOG", &log)
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
        b"#!/usr/bin/env bash\nif [[ \"${GH_ZERO_BAD_SIZE:-}\" == 1 ]]; then echo 1; else echo 2147023008; fi\n",
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
        if [[ "${GH_ZERO_HEALTH_WAIT:-}" == 1 ]]; then /bin/sleep 0.2; exit 1; fi
        exit 0
        ;;
    */apply-template)
        if [[ -n "${GH_ZERO_APPLY_TEMPLATE_LOG:-}" ]]; then
            printf '%s\n' "$body" > "$GH_ZERO_APPLY_TEMPLATE_LOG"
        fi
        printf '{"prompt":"oracle prompt"}\n'
        ;;
    */tokenize)
        if [[ "${GH_ZERO_BAD_HTTP:-}" == tokenize ]]; then printf '{bad json\n'; else printf '{"tokens":[1,2,3]}\n'; fi
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
printf 'args=%s\nmodel=%s\ncontext=%s\nkv=%s\npercent=%s\nprompt=%s\ncompletion=%s\n' \
    "$*" "$GH_ZERO_MODEL" "$GH_ZERO_CONTEXT" "$GH_ZERO_KV" \
    "$GH_ZERO_VRAM_WEIGHTS_PERCENT" "$GH_ZERO_REFERENCE_PROMPT_IDS" \
    "$GH_ZERO_REFERENCE_COMPLETION_IDS" >> "$GH_ZERO_CARGO_LOG"
if [[ "${GH_ZERO_MEMORY_FAILURE:-}" == 1 ]]; then
    printf 'Vulkan memory is insufficient: required 2 bytes, available 1 bytes\n' >&2
    exit 1
fi
printf 'test result: ok\nministral-parity: local_ids=30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45 oracle_top2=pass\n'
"#,
    )
    .unwrap();
    fs::write(
        &mktemp,
        b"#!/usr/bin/env bash\nmkdir -p -- \"$GH_ZERO_TEMP_DIR\"\nprintf '%s\\n' \"$GH_ZERO_TEMP_DIR\"\n",
    )
    .unwrap();
    fs::write(
        &server,
        r#"#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == --version ]]; then
    if [[ "${GH_ZERO_BAD_REVISION:-}" == 1 ]]; then echo 'llama.cpp old'; else echo 'llama.cpp 13f2b28b0'; fi
    exit 0
fi
printf 'started %s\n' "$$" >> "$GH_ZERO_SERVER_LOG"
trap 'printf "stopped %s\n" "$$" >> "$GH_ZERO_SERVER_LOG"; exit 0' TERM INT HUP
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
        "hybrid",
        "--kv",
        "int8",
        "--reference-server",
        server.to_str().unwrap(),
    ];

    let port = free_port();
    let pass = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &port])
        .env("PATH", &path)
        .env("GH_ZERO_TEMP_DIR", &temp)
        .env("GH_ZERO_CARGO_LOG", &cargo_log)
        .env("GH_ZERO_SERVER_LOG", &server_log)
        .env("GH_ZERO_APPLY_TEMPLATE_LOG", &template_log)
        .output()
        .unwrap();
    assert!(
        pass.status.success(),
        "{}",
        String::from_utf8_lossy(&pass.stderr)
    );
    let stdout = String::from_utf8(pass.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1);
    assert!(stdout.starts_with("pass: model_id=3b-instruct backend=hybrid kv=int8"));
    assert!(
        stdout.contains(
            "prompt_ids=1,2,3 oracle_ids=10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
        )
    );
    assert!(stdout.contains("local_ids=30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45"));
    assert!(!temp.exists());
    let cargo_call = fs::read_to_string(&cargo_log).unwrap();
    assert!(
        cargo_call
            .contains("--features hybrid family::mistral::hybrid::graph::real_ministral_parity")
    );
    assert!(cargo_call.contains("context=4096\nkv=int8\npercent=25"));
    assert!(cargo_call.contains(&format!("model={}", model.display())));
    assert_eq!(
        fs::read_to_string(&template_log).unwrap().trim_end(),
        r#"{"messages":[{"role":"system","content":""},{"role":"user","content":"Quanto fa 17 × 19?"}],"add_generation_prompt":true}"#
    );

    let port = free_port();
    let malformed = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &port])
        .env("PATH", &path)
        .env("GH_ZERO_TEMP_DIR", &temp)
        .env("GH_ZERO_CARGO_LOG", &cargo_log)
        .env("GH_ZERO_SERVER_LOG", &server_log)
        .env("GH_ZERO_APPLY_TEMPLATE_LOG", &template_log)
        .env("GH_ZERO_BAD_HTTP", "tokenize")
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
        .env("GH_ZERO_BAD_REVISION", "1")
        .output()
        .unwrap();
    assert!(revision.status.success());
    assert_eq!(
        String::from_utf8(revision.stdout).unwrap().trim(),
        "external verification: unsupported llama.cpp revision"
    );

    let server_calls = fs::read_to_string(&server_log).unwrap().lines().count();
    let cargo_calls = fs::read_to_string(&cargo_log).unwrap().lines().count();
    let mismatch = Command::new("bash")
        .arg(repository().join("support/testing/parity-check.sh"))
        .args(base)
        .args(["--reference-port", &free_port()])
        .env("PATH", &path)
        .env("GH_ZERO_BAD_SIZE", "1")
        .output()
        .unwrap();
    assert!(mismatch.status.success());
    assert_eq!(
        String::from_utf8(mismatch.stdout).unwrap().trim(),
        "external verification: 3b-instruct byte count mismatch"
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
        .env("GH_ZERO_TEMP_DIR", &temp)
        .env("GH_ZERO_CARGO_LOG", &cargo_log)
        .env("GH_ZERO_SERVER_LOG", &server_log)
        .env("GH_ZERO_MEMORY_FAILURE", "1")
        .output()
        .unwrap();
    assert!(memory.status.success());
    assert_eq!(
        String::from_utf8(memory.stdout).unwrap().trim(),
        "external verification: insufficient memory for hybrid row"
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
        .env("GH_ZERO_TEMP_DIR", &temp)
        .env("GH_ZERO_CARGO_LOG", &cargo_log)
        .env("GH_ZERO_SERVER_LOG", &server_log)
        .env("GH_ZERO_HEALTH_WAIT", "1")
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
        "GH_ZERO_CONTEXT=4096",
        "GH_ZERO_VRAM_WEIGHTS_PERCENT=25",
        "n_predict:16",
        "--host 127.0.0.1",
        "--offline",
        "--device none",
        "--n-gpu-layers 0",
        "--no-kv-offload",
        "--no-warmup",
        "--ignore-eos",
        "--exact",
    ] {
        assert!(source.contains(fixed), "missing fixed contract: {fixed}");
    }
    assert!(!source.contains("--context)"));
    assert!(!source.contains("--vram-weights-percent)"));
    assert!(!source.contains("eval "));
    assert_eq!(fs::read(&model).unwrap(), b"immutable parity fixture");
    fs::remove_dir_all(fixture).unwrap();
}

#[test]
fn matrix_script_contract() {
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
id=""; backend=""; kv=""
while (($#)); do
    case "$1" in
        --model-id) id="$2"; shift 2 ;; --backend) backend="$2"; shift 2 ;;
        --kv) kv="$2"; shift 2 ;; *) shift ;;
    esac
done
key="$id:$backend:$kv"
printf '%s\n' "$key" >> "$GH_ZERO_PARITY_LOG"
if [[ "$key" == "${GH_ZERO_FAIL_KEY:-}" ]]; then echo 'injected failure' >&2; exit 1; fi
if [[ "$key" == "${GH_ZERO_EXTERNAL_KEY:-}" ]]; then echo 'external verification: injected resource unavailable'; exit 0; fi
printf 'pass: model_id=%s backend=%s kv=%s prompt_ids=1 oracle_ids=2 local_ids=3\n' "$id" "$backend" "$kv"
"#,
    )
    .unwrap();
    let cargo = bin.join("cargo");
    fs::write(
        &cargo,
        r#"#!/usr/bin/env bash
set -eu
model="${!#}"
printf '%s\n' "$model" >> "$GH_ZERO_INSPECT_LOG"
if [[ -n "${GH_ZERO_BAD_Q8:-}" && "$model" == *"$GH_ZERO_BAD_Q8"* ]]; then
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
        .env("GH_ZERO_PARITY_LOG", &parity_log)
        .env("GH_ZERO_INSPECT_LOG", &inspect_log)
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
    assert_eq!(statuses.len(), 42);
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
        36
    );
    assert_eq!(statuses.iter().copied().collect::<HashSet<_>>().len(), 42);
    assert!(stdout.contains("summary: pass=42 external_verification=0 failure=0 total=42"));
    assert_eq!(fs::read_to_string(&inspect_log).unwrap().lines().count(), 6);

    let expected = rows
        .iter()
        .flat_map(|row| {
            ["cpu", "vulkan", "hybrid"]
                .into_iter()
                .flat_map(move |backend| {
                    ["f16", "int8"]
                        .into_iter()
                        .map(move |kv| format!("{}:{backend}:{kv}", row.id))
                })
        })
        .collect::<Vec<_>>();
    let calls = fs::read_to_string(&parity_log).unwrap();
    assert_eq!(calls.lines().collect::<Vec<_>>(), expected);

    fs::remove_file(models.join(rows[0].q8_file)).unwrap();
    fs::write(&parity_log, []).unwrap();
    fs::write(&inspect_log, []).unwrap();
    let continued = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GH_ZERO_PARITY_LOG", &parity_log)
        .env("GH_ZERO_INSPECT_LOG", &inspect_log)
        .env("GH_ZERO_EXTERNAL_KEY", "3b-instruct:cpu:f16")
        .output()
        .unwrap();
    assert!(continued.status.success());
    let stdout = String::from_utf8(continued.stdout).unwrap();
    assert!(stdout.contains("q8 model_id=3b-instruct: external verification"));
    assert!(
        stdout.contains("parity model_id=3b-instruct backend=cpu kv=f16: external verification")
    );
    assert!(stdout.contains("summary: pass=40 external_verification=2 failure=0 total=42"));
    assert_eq!(fs::read_to_string(&parity_log).unwrap().lines().count(), 36);
    fs::write(models.join(rows[0].q8_file), b"Q8 rejection fixture").unwrap();

    fs::write(&parity_log, []).unwrap();
    fs::write(&inspect_log, []).unwrap();
    let stopped = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GH_ZERO_PARITY_LOG", &parity_log)
        .env("GH_ZERO_INSPECT_LOG", &inspect_log)
        .env("GH_ZERO_FAIL_KEY", "3b-instruct:cpu:int8")
        .output()
        .unwrap();
    assert_eq!(stopped.status.code(), Some(1));
    let stdout = String::from_utf8(stopped.stdout).unwrap();
    assert!(stdout.contains("parity model_id=3b-instruct backend=cpu kv=int8: failure"));
    assert!(stdout.contains("summary: pass=7 external_verification=0 failure=1 total=8"));
    assert_eq!(fs::read_to_string(&parity_log).unwrap().lines().count(), 2);

    fs::write(&parity_log, []).unwrap();
    fs::write(&inspect_log, []).unwrap();
    let bad_q8 = Command::new(&matrix)
        .args(base)
        .env("PATH", &path)
        .env("GH_ZERO_PARITY_LOG", &parity_log)
        .env("GH_ZERO_INSPECT_LOG", &inspect_log)
        .env("GH_ZERO_BAD_Q8", "3B-Instruct")
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
        .env("GH_ZERO_PARITY_LOG", &parity_log)
        .env("GH_ZERO_INSPECT_LOG", &inspect_log)
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
    assert!(source.contains("for backend in cpu vulkan hybrid"));
    assert!(source.contains("for kv in f16 int8"));
    assert_eq!(fs::read(&server).unwrap(), b"reference fixture");
    fs::remove_dir_all(fixture).unwrap();
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
if [[ -n "${SEMANTIC_STUB_BAD_SIZE:-}" && "$model" == *"$SEMANTIC_STUB_BAD_SIZE"* ]]; then echo 1; exit 0; fi
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
    fs::write(
        &cargo,
        r#"#!/usr/bin/env bash
set -eu
printf '%s\t%s\t%s\n' "$GH_ZERO_MODEL_ID" "$GH_ZERO_MODEL" "$*" >> "$SEMANTIC_STUB_LOG"
if [[ "$GH_ZERO_MODEL_ID" == "${SEMANTIC_STUB_EXTERNAL_ID:-}" ]]; then
    printf 'semantic-external: model_id=%s reason=insufficient RAM\n' "$GH_ZERO_MODEL_ID"
    exit 0
fi
case "$GH_ZERO_MODEL_ID" in
    3b-instruct) backend=vulkan; reason=full-vram-fit; probe=all-gpu; run=all-gpu; cpu_layers=0; gpu_layers=26 ;;
    3b-reasoning) backend=cpu; reason=no-full-vram-fit; probe=mixed; run=cpu-only; cpu_layers=26; gpu_layers=0 ;;
    *) backend=cpu; reason=no-full-vram-fit; probe=cpu-only; run=cpu-only; cpu_layers=34; gpu_layers=0 ;;
esac
case "$GH_ZERO_MODEL_ID" in *reasoning) chat=reasoning ;; *) chat=instruct ;; esac
if [[ "$GH_ZERO_MODEL_ID" == 3b-instruct && "${SEMANTIC_STUB_PROTOCOL:-}" == contradictory ]]; then
    reason=no-full-vram-fit
fi
selection="semantic-selection: model_id=$GH_ZERO_MODEL_ID backend=$backend reason=$reason probe_mode=$probe run_mode=$run cpu_layers=$cpu_layers gpu_layers=$gpu_layers cpu_weights=0 cpu_kv=0 cpu_scratch=0 cpu_fixed=0 cpu_staging=0 cpu_crossing=0 cpu_reserve=0 cpu_total=0 gpu_weights=0 gpu_kv=0 gpu_scratch=0 gpu_fixed=0 gpu_staging=0 gpu_crossing=0 gpu_reserve=0 gpu_total=0"
printf '%s\n' "$selection"
if [[ "$GH_ZERO_MODEL_ID" == 3b-instruct && "${SEMANTIC_STUB_PROTOCOL:-}" == duplicate ]]; then
    printf '%s\n' "$selection"
fi
for case_id in S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12; do
    case "$case_id" in S05|S11|S12) class=conformance ;; *) class=semantic ;; esac
    marker=not-applicable
    [[ "$chat" != reasoning ]] || marker=complete
    [[ "$GH_ZERO_MODEL_ID" != 3b-reasoning || "$case_id" != S12 ]] || marker=absent
    [[ "$GH_ZERO_MODEL_ID" != 3b-reasoning || "${SEMANTIC_STUB_PROTOCOL:-}" != zero-format ]] || marker=absent
    stop=eos
    status=pass
    details=
    if [[ "$GH_ZERO_MODEL_ID" == 3b-instruct && "$case_id" == S01 && "${SEMANTIC_STUB_PROTOCOL:-}" == non-eos ]]; then
        stop=context; status=fail; details=' reason=incomplete-generation excerpt=fixture'
    elif [[ "$GH_ZERO_MODEL_ID" == 3b-instruct && "$class" == conformance && "${SEMANTIC_STUB_PROTOCOL:-}" == zero-conformance ]]; then
        status=fail; details=' reason=fixture-failure excerpt=fixture'
    elif [[ "$GH_ZERO_MODEL_ID" == 3b-reasoning && "$case_id" == S11 && "${SEMANTIC_STUB_PROTOCOL:-}" == conformance-non-eos ]]; then
        stop=context; status=fail; details=' reason=incomplete-generation excerpt=fixture'
    elif [[ "$GH_ZERO_MODEL_ID" == 3b-reasoning && "$case_id" == S11 && "${SEMANTIC_STUB_PROTOCOL:-}" == conformance-non-eos-pass ]]; then
        stop=context
    fi
    emitted_id="$case_id"
    [[ "$GH_ZERO_MODEL_ID" != 3b-instruct || "$case_id" != S02 || "${SEMANTIC_STUB_PROTOCOL:-}" != duplicate-case ]] || emitted_id=S01
    printf 'semantic-case: model_id=%s case_id=%s status=%s predicate=fixture class=%s stop=%s prompt_tokens=16 completion_tokens=1 marker_status=%s%s\n' "$GH_ZERO_MODEL_ID" "$emitted_id" "$status" "$class" "$stop" "$marker" "$details"
done
if [[ "$chat" == reasoning ]]; then
    complete=12
    [[ "$GH_ZERO_MODEL_ID" != 3b-reasoning ]] || complete=11
    [[ "$GH_ZERO_MODEL_ID" != 3b-reasoning || "${SEMANTIC_STUB_PROTOCOL:-}" != marker-total ]] || complete=12
    [[ "$GH_ZERO_MODEL_ID" != 3b-reasoning || "${SEMANTIC_STUB_PROTOCOL:-}" != zero-format ]] || complete=0
    format="$complete/12"; format_status=diagnostic
else
    format=not-applicable; format_status=not-applicable
fi
critical=4; semantic=9; semantic_status=pass; conformance=3
if [[ "$GH_ZERO_MODEL_ID" == 3b-instruct && "${SEMANTIC_STUB_PROTOCOL:-}" == non-eos ]]; then
    critical=3; semantic=8; semantic_status=fail
elif [[ "$GH_ZERO_MODEL_ID" == 3b-instruct && "${SEMANTIC_STUB_PROTOCOL:-}" == zero-conformance ]]; then
    conformance=0
elif [[ "$GH_ZERO_MODEL_ID" == 3b-reasoning && "${SEMANTIC_STUB_PROTOCOL:-}" == conformance-non-eos ]]; then
    conformance=2
fi
printf 'semantic-summary: model_id=%s backend=%s critical=%s/4 semantic=%s/9 semantic_status=%s conformance=%s/3 conformance_status=diagnostic reasoning_format=%s reasoning_format_status=%s\n' "$GH_ZERO_MODEL_ID" "$backend" "$critical" "$semantic" "$semantic_status" "$conformance" "$format" "$format_status"
if [[ "$GH_ZERO_MODEL_ID" != 3b-instruct || "${SEMANTIC_STUB_PROTOCOL:-}" != missing-timing ]]; then
    total=100
    [[ "${SEMANTIC_STUB_PROTOCOL:-}" != malformed || "$GH_ZERO_MODEL_ID" != 3b-instruct ]] || total=invalid
    if [[ "$GH_ZERO_MODEL_ID" == 3b-instruct ]]; then baseline=1506690; performance=pass; else baseline=not-applicable; performance=not-applicable; fi
    printf 'semantic-timing: model_id=%s backend=%s completed_cases=12 total_ms=%s prefill_ms=40 decode_ms=60 baseline_cpu_ms=%s performance_status=%s\n' "$GH_ZERO_MODEL_ID" "$backend" "$total" "$baseline" "$performance"
fi
printf 'internal /secret/path must not escape\n' >&2
[[ "$GH_ZERO_MODEL_ID" != "${SEMANTIC_STUB_FAIL_ID:-}" ]]
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
            .env("SEMANTIC_STUB_LOG", &calls);
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
        .filter(|line| line.starts_with("semantic model_id="))
        .collect::<Vec<_>>();
    assert_eq!(statuses.len(), 6);
    assert_eq!(statuses.iter().copied().collect::<HashSet<_>>().len(), 6);
    assert!(stdout.contains("summary: pass=6 external_verification=0 failure=0 total=6"));
    assert!(!stdout.contains("/secret/path"));
    assert_eq!(stdout.matches("semantic-selection:").count(), 6);
    assert_eq!(stdout.matches("semantic-case:").count(), 72);
    assert_eq!(stdout.matches(" stop=eos ").count(), 72);
    assert_eq!(stdout.matches("marker_status=not-applicable").count(), 36);
    assert_eq!(stdout.matches("marker_status=complete").count(), 35);
    assert_eq!(stdout.matches("marker_status=absent").count(), 1);
    assert_eq!(stdout.matches("semantic-summary:").count(), 6);
    assert_eq!(stdout.matches("semantic-timing:").count(), 6);
    for probe in [
        "probe_mode=all-gpu",
        "probe_mode=mixed",
        "probe_mode=cpu-only",
    ] {
        assert!(stdout.contains(probe));
    }

    for (protocol, expected) in [
        (
            "zero-conformance",
            "semantic-summary: model_id=3b-instruct backend=vulkan critical=4/4 semantic=9/9 semantic_status=pass conformance=0/3 conformance_status=diagnostic reasoning_format=not-applicable reasoning_format_status=not-applicable",
        ),
        (
            "zero-format",
            "semantic-summary: model_id=3b-reasoning backend=cpu critical=4/4 semantic=9/9 semantic_status=pass conformance=3/3 conformance_status=diagnostic reasoning_format=0/12 reasoning_format_status=diagnostic",
        ),
        (
            "conformance-non-eos",
            "semantic-case: model_id=3b-reasoning case_id=S11 status=fail predicate=fixture class=conformance stop=context prompt_tokens=16 completion_tokens=1 marker_status=complete reason=incomplete-generation excerpt=fixture",
        ),
    ] {
        fs::write(&calls, []).unwrap();
        let valid = run(&[("SEMANTIC_STUB_PROTOCOL", protocol)]);
        assert!(valid.status.success(), "{protocol}");
        let stdout = String::from_utf8(valid.stdout).unwrap();
        assert!(stdout.contains(expected), "{protocol}");
        if protocol == "conformance-non-eos" {
            assert!(stdout.contains("conformance=2/3"));
        }
        assert!(stdout.contains("summary: pass=6 external_verification=0 failure=0 total=6"));
        assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 6);
    }

    let cargo_calls = fs::read_to_string(&calls).unwrap();
    assert_eq!(cargo_calls.lines().count(), 6);
    for (line, row) in cargo_calls.lines().zip(&rows) {
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(fields[0], row.id);
        assert_eq!(fields[1], models.join(row.q4_file).to_str().unwrap());
        assert!(fields[2].contains(
            "--no-default-features --features hybrid --test semantic real_semantic_acceptance"
        ));
        assert!(fields[2].contains("--ignored --nocapture --exact"));
    }

    fs::remove_file(models.join(rows[0].q4_file)).unwrap();
    fs::write(&calls, []).unwrap();
    let missing = run(&[]);
    assert!(missing.status.success());
    let stdout = String::from_utf8(missing.stdout).unwrap();
    assert!(stdout.contains(
        "semantic model_id=3b-instruct: external verification: artifact is missing or unreadable"
    ));
    assert!(stdout.contains("summary: pass=5 external_verification=1 failure=0 total=6"));
    assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 5);
    fs::write(models.join(rows[0].q4_file), b"immutable semantic fixture").unwrap();

    fs::write(&calls, []).unwrap();
    let mismatch = run(&[("SEMANTIC_STUB_BAD_SIZE", "3B-Instruct")]);
    assert!(mismatch.status.success());
    assert!(
        String::from_utf8(mismatch.stdout)
            .unwrap()
            .contains("summary: pass=5 external_verification=1 failure=0 total=6")
    );
    assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 5);

    fs::write(&calls, []).unwrap();
    let failure = run(&[("SEMANTIC_STUB_FAIL_ID", "3b-reasoning")]);
    assert_eq!(failure.status.code(), Some(1));
    let stdout = String::from_utf8(failure.stdout).unwrap();
    assert!(stdout.contains("semantic model_id=3b-reasoning: failure"));
    let statuses = stdout
        .lines()
        .filter(|line| line.starts_with("semantic model_id="))
        .collect::<Vec<_>>();
    assert_eq!(statuses.len(), 6);
    assert_eq!(statuses.iter().copied().collect::<HashSet<_>>().len(), 6);
    assert!(stdout.contains("summary: pass=5 external_verification=0 failure=1 total=6"));
    assert!(!stdout.contains("/secret/path"));
    assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 6);

    fs::remove_file(models.join(rows[5].q4_file)).unwrap();
    fs::write(&calls, []).unwrap();
    let mixed = run(&[("SEMANTIC_STUB_FAIL_ID", "3b-reasoning")]);
    assert_eq!(mixed.status.code(), Some(1));
    let stdout = String::from_utf8(mixed.stdout).unwrap();
    assert!(stdout.contains("semantic model_id=3b-reasoning: failure"));
    assert!(stdout.contains(
        "semantic model_id=14b-reasoning: external verification: artifact is missing or unreadable"
    ));
    assert!(stdout.contains("summary: pass=4 external_verification=1 failure=1 total=6"));
    assert_eq!(
        stdout
            .lines()
            .filter(|line| line.starts_with("semantic model_id="))
            .count(),
        6
    );
    assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 5);
    fs::write(models.join(rows[5].q4_file), b"immutable semantic fixture").unwrap();

    fs::write(&calls, []).unwrap();
    let external_ram = run(&[("SEMANTIC_STUB_EXTERNAL_ID", "8b-instruct")]);
    assert!(external_ram.status.success());
    let stdout = String::from_utf8(external_ram.stdout).unwrap();
    assert!(
        stdout.contains("semantic model_id=8b-instruct: external verification: insufficient RAM")
    );
    assert!(stdout.contains("summary: pass=5 external_verification=1 failure=0 total=6"));
    assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 6);

    for (protocol, failed_id) in [
        ("missing-timing", "3b-instruct"),
        ("duplicate", "3b-instruct"),
        ("duplicate-case", "3b-instruct"),
        ("contradictory", "3b-instruct"),
        ("malformed", "3b-instruct"),
        ("non-eos", "3b-instruct"),
        ("conformance-non-eos-pass", "3b-reasoning"),
        ("marker-total", "3b-reasoning"),
    ] {
        fs::write(&calls, []).unwrap();
        let invalid = run(&[("SEMANTIC_STUB_PROTOCOL", protocol)]);
        assert_eq!(invalid.status.code(), Some(1), "{protocol}");
        let stdout = String::from_utf8(invalid.stdout).unwrap();
        assert!(
            stdout.contains(&format!("semantic model_id={failed_id}: failure")),
            "{protocol}"
        );
        assert!(stdout.contains("summary: pass=5 external_verification=0 failure=1 total=6"));
        assert!(!stdout.contains("/secret/path"));
        assert_eq!(fs::read_to_string(&calls).unwrap().lines().count(), 6);
    }

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
    for forbidden in [
        "curl ",
        "wget ",
        "eval ",
        "--features vulkan",
        "--features cpu",
    ] {
        assert!(
            !source.contains(forbidden),
            "semantic runner contains {forbidden}"
        );
    }
    assert_eq!(source.matches("--features hybrid").count(), 1);
    assert!(source.lines().count() < 200);
    for row in &rows {
        assert_eq!(
            fs::read(models.join(row.q4_file)).unwrap(),
            b"immutable semantic fixture"
        );
    }
    fs::remove_dir_all(fixture).unwrap();
}
