/*
 * graph_horizon_engine — repository documentation contract
 * Protects public entry points, runtime-flag coverage, AI development material,
 * and local Markdown links without freezing release prose or historical data.
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("engine crate is inside the workspace")
        .to_path_buf()
}

#[test]
fn docs_contract() {
    let root = repository();
    let readme = fs::read_to_string(root.join("README.md")).expect("root README");
    let engine = fs::read_to_string(root.join("crates/graph_horizon_engine/README.md"))
        .expect("engine README");
    let installation =
        fs::read_to_string(root.join("docs/installation.md")).expect("installation docs");
    let models = fs::read_to_string(root.join("docs/supported-models-and-formats.md"))
        .expect("supported-model docs");
    let runtime = fs::read_to_string(root.join("docs/command-line/runtime-options.md"))
        .expect("runtime option docs");
    let args = fs::read_to_string(root.join("src/app/args.rs")).expect("runtime arguments");

    for entrypoint in [
        "docs/installation.md",
        "docs/command-line/runtime-options.md",
        "docs/supported-models-and-formats.md",
        "docs/web-interface/README.md",
        "docs/engine/backend-support-status.md",
        "docs/project-status/validation-evidence.md",
    ] {
        assert!(
            readme.contains(entrypoint),
            "root README missing entry point: {entrypoint}"
        );
    }

    for protected in [
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        ".skills/deep-clean/SKILL.md",
        ".skills/implementer/SKILL.md",
        ".skills/optimizer/SKILL.md",
        ".skills/planner/SKILL.md",
        ".skills/release/SKILL.md",
        ".skills/reviewer/SKILL.md",
        ".skills/rust_code_ablation/SKILL.md",
        ".skills/safe_optimization_loop/SKILL.md",
        ".skills/safe_optimization_loop/references/correctness-oracle.md",
        ".skills/safe_optimization_loop/references/measuring-prefill-decode.md",
        ".skills/safe_optimization_loop/references/rust-optimization-catalog.md",
    ] {
        assert!(
            root.join(protected).is_file(),
            "protected AI document missing: {protected}"
        );
    }
    let agents = fs::read_to_string(root.join("AGENTS.md")).expect("AI policy");
    assert!(agents.contains("Protected AI Development Material"));
    assert!(agents.contains("Never delete, rename, merge, replace, or consolidate"));
    let ignore = fs::read_to_string(root.join(".gitignore")).expect("ignore policy");
    assert!(ignore.lines().any(|line| line == "/DECISIONS.md"));
    assert!(ignore.lines().any(|line| line == "plans/"));

    let production_args = args.split("#[cfg(test)]").next().unwrap();
    let flag_block = production_args
        .split("const FLAGS")
        .nth(1)
        .and_then(|rest| rest.split("];").next())
        .expect("runtime flag table");
    for line in flag_block.lines() {
        let Some(flag) = line
            .trim()
            .strip_suffix(',')
            .and_then(|line| line.strip_prefix('"'))
            .and_then(|line| line.strip_suffix('"'))
        else {
            continue;
        };
        assert!(
            runtime.contains(&format!("`{flag} <")),
            "runtime option docs missing {flag}"
        );
    }

    let readme_flat = readme.split_whitespace().collect::<Vec<_>>().join(" ");
    let engine_flat = engine.split_whitespace().collect::<Vec<_>>().join(" ");
    let models_flat = models.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(models_flat.contains("Ministral 3 Instruct and Reasoning 2512"));
    assert!(models_flat.contains("Only the public `Q4_K_M` profile is accepted"));
    assert!(!production_args.contains("--think"));
    for removed in ["--mode server", "--provider", "--base-url"] {
        assert!(
            !production_args.contains(removed),
            "removed runtime surface remains: {removed}"
        );
    }
    for unsupported in ["Q5_K_M", "Q6_K_M", "Mistral Small", "24B"] {
        assert!(
            !readme_flat.contains(unsupported) && !engine_flat.contains(unsupported),
            "unsupported public claim: {unsupported}"
        );
    }

    for document in [&readme, &installation] {
        let flat = document.split_whitespace().collect::<Vec<_>>().join(" ");
        for backend in ["vulkan-hybrid", "metal-hybrid", "cuda-hybrid"] {
            let command = format!(
                "https://raw.githubusercontent.com/etufarini/graph-horizon/v0.1.5/install.sh \\ | bash -s -- --backend {backend}"
            );
            assert_eq!(
                flat.matches(&command).count(),
                1,
                "stable quick install must contain exactly one {backend} command"
            );
        }
        for line in document
            .lines()
            .filter(|line| line.contains("raw.githubusercontent.com/etufarini/graph-horizon/"))
        {
            assert!(
                line.contains("/v0.1.5/install.sh"),
                "quick install must use only the v0.1.5 bootstrap: {line}"
            );
        }
    }

    assert_local_markdown_links(&root);
}

fn assert_local_markdown_links(root: &Path) {
    let output = Command::new("git")
        .args([
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.md",
        ])
        .current_dir(root)
        .output()
        .expect("list tracked Markdown");
    assert!(output.status.success());
    for relative in String::from_utf8(output.stdout).unwrap().lines() {
        let path = root.join(relative);
        if !path.exists() {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read Markdown");
        let mut rest = text.as_str();
        while let Some(start) = rest.find("](") {
            rest = &rest[start + 2..];
            let Some(end) = rest.find(')') else {
                break;
            };
            let raw = rest[..end].trim().trim_matches(['<', '>']);
            rest = &rest[end + 1..];
            if raw.is_empty()
                || raw.starts_with('#')
                || raw.starts_with("http://")
                || raw.starts_with("https://")
                || raw.starts_with("mailto:")
            {
                continue;
            }
            let target = raw.split('#').next().unwrap();
            let resolved = path.parent().unwrap().join(target);
            assert!(resolved.exists(), "broken local link in {relative}: {raw}");
        }
    }
}
