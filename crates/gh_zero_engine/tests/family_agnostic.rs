/*
 * gh_zero_engine — final source and family boundary audit
 * Statically enforces the exact K/I exemption lists, the 200 productive-line
 * orchestration limit, the Reasoning documentation contract, and absence of
 * obsolete family/backend domains. Tiny synthetic GGUFs exercise the sole
 * mistral3 architecture gate.
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(feature = "hybrid")]
use gh_zero_engine::{BackendMemory, PlacementReport};
use gh_zero_engine::{Engine, EngineConfig};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repository() -> PathBuf {
    manifest()
        .parent()
        .and_then(Path::parent)
        .expect("engine crate is inside the workspace")
        .to_path_buf()
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn productive_lines(path: &Path) -> usize {
    let text = fs::read_to_string(path).expect("read source");
    let mut in_test = false;
    text.lines()
        .filter(|line| {
            let line = line.trim();
            if line.starts_with("#[cfg(test)]") {
                in_test = true;
            }
            !in_test
                && !line.is_empty()
                && !line.starts_with("//")
                && !line.starts_with("/*")
                && !line.starts_with('*')
        })
        .count()
}

#[test]
fn source_structure() {
    const I: &[&str] = &[
        "src/backend/contract.rs",
        "src/backend/cpu/backend.rs",
        "src/backend/vulkan/backend.rs",
    ];
    const K: &[&str] = &[
        "src/backend/cpu/dequant.rs",
        "src/backend/cpu/kernels/attention/mod.rs",
        "src/backend/cpu/kernels/attention/read_q.rs",
        "src/backend/cpu/kernels/attention/simd.rs",
        "src/backend/cpu/kernels/attention/write_q.rs",
        "src/backend/cpu/kernels/matmul/mod.rs",
        "src/backend/cpu/kernels/matmul/q4k.rs",
        "src/backend/cpu/kernels/matmul/q4k_simd.rs",
        "src/backend/cpu/kernels/matmul/q5k.rs",
        "src/backend/cpu/kernels/matmul/q5k_simd.rs",
        "src/backend/cpu/kernels/matmul/q6k.rs",
        "src/backend/cpu/kernels/matmul/q6k_simd.rs",
        "src/backend/cpu/kernels/matmul/q8_0.rs",
        "src/backend/cpu/kernels/matmul/q8_0_simd.rs",
        "src/backend/vulkan/kernels/attention.rs",
        "src/backend/vulkan/kernels/matmul.rs",
    ];
    const TEST_FIXTURES: &[&str] = &[
        "src/family/mistral/graph/shape.rs",
        "src/family/mistral/vulkan.rs",
    ];
    let mut files = Vec::new();
    collect(&manifest().join("src"), &mut files);
    let mut marked_i = Vec::new();
    let mut marked_k = Vec::new();
    let mut over = Vec::new();
    for path in files {
        let relative = path
            .strip_prefix(manifest())
            .expect("source under manifest")
            .to_string_lossy()
            .replace('\\', "/");
        let text = fs::read_to_string(&path).expect("read source");
        if text.contains("AGENTS deroga I") {
            marked_i.push(relative.clone());
        }
        if text.contains("AGENTS deroga K") {
            marked_k.push(relative.clone());
        }
        if !I.contains(&relative.as_str())
            && !K.contains(&relative.as_str())
            && !TEST_FIXTURES.contains(&relative.as_str())
            && productive_lines(&path) > 200
        {
            over.push(format!("{relative}: {}", productive_lines(&path)));
        }
    }
    marked_i.sort();
    marked_k.sort();
    let mut expected_i = I.iter().map(|value| value.to_string()).collect::<Vec<_>>();
    let mut expected_k = K.iter().map(|value| value.to_string()).collect::<Vec<_>>();
    expected_i.sort();
    expected_k.sort();
    assert_eq!(marked_i, expected_i, "category-I marker list changed");
    assert_eq!(marked_k, expected_k, "category-K marker list changed");
    assert!(
        over.is_empty(),
        "orchestration over 200 lines:\n{}",
        over.join("\n")
    );
}

#[test]
fn removed_surface_scan() {
    let previous_family = concat!("src/family/qwen", "35");
    assert!(
        !manifest().join(previous_family).exists(),
        "{previous_family} must be absent"
    );
    for relative in [
        "src/family/shared",
        "src/resident",
        "src/backend/vulkan/hybrid",
        "src/backend/vulkan/resident",
        "src/backend/vulkan/mem/pool.rs",
        "src/backend/cpu/kernels/ssm",
    ] {
        assert!(
            !manifest().join(relative).exists(),
            "{relative} must be absent"
        );
    }
    for relative in [
        "docs/tools.md",
        "support/testing/resident-golden.sh",
        "crates/gh_zero_engine/examples/prefill_trace.rs",
        "crates/gh_zero_engine/examples/quality.rs",
        "crates/gh_zero_engine/examples/regression.rs",
        "crates/gh_zero_engine/examples/validate.rs",
    ] {
        assert!(
            !repository().join(relative).exists(),
            "{relative} must be absent"
        );
    }
    let old_kv = format!("docs/kv-quant-{}-validation.md", concat!("qwen", "35"));
    assert!(
        !repository().join(&old_kv).exists(),
        "{old_kv} must be absent"
    );

    let identifier = concat!("qwen", "35");
    let scan = Command::new("git")
        .args(["grep", "-Iin", identifier, "--", "."])
        .current_dir(repository())
        .output()
        .expect("run tracked-tree scan");
    assert_eq!(
        scan.status.code(),
        Some(1),
        "previous family remains tracked"
    );
    assert!(scan.stdout.is_empty());
}

#[test]
fn docs_contract() {
    let root = repository();
    let readme = fs::read_to_string(root.join("README.md")).expect("root README");
    let engine =
        fs::read_to_string(root.join("crates/gh_zero_engine/README.md")).expect("engine README");
    let backend = fs::read_to_string(root.join("docs/backend.md")).expect("backend docs");
    let config =
        fs::read_to_string(root.join("docs/configuration.md")).expect("configuration docs");
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("architecture docs");
    let ownership = fs::read_to_string(root.join("crates/gh_zero_engine/src/README.md"))
        .expect("source ownership");
    let support = fs::read_to_string(root.join("support/README.md")).expect("support docs");
    let server = fs::read_to_string(root.join("docs/server.md")).expect("server docs");
    let kv = fs::read_to_string(root.join("docs/kv-quant-mistral-validation.md"))
        .expect("KV validation docs");
    let args = fs::read_to_string(root.join("src/app/args.rs")).expect("runtime arguments");
    let readme_flat = readme.split_whitespace().collect::<Vec<_>>().join(" ");
    let engine_flat = engine.split_whitespace().collect::<Vec<_>>().join(" ");
    let backend_flat = backend.split_whitespace().collect::<Vec<_>>().join(" ");
    let config_flat = config.split_whitespace().collect::<Vec<_>>().join(" ");
    let architecture_flat = architecture
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let support_flat = support.split_whitespace().collect::<Vec<_>>().join(" ");

    for required in [
        "Q8_0",
        "Q4_K_M",
        "3B",
        "8B",
        "14B",
        "compatible/unverified",
        "FP16 activations, FP32 residual/logits, and F16/Q4_K/Q5_K/Q6_K/Q8_0 weights",
        "all-GPU",
        "maximum possible contiguous GPU suffix",
        "all-CPU",
        "Pure Vulkan",
        "CPU-first",
        "do not constitute MoE support",
        "lesser of 32,768 tokens and the GGUF context maximum",
        "262,144 tokens",
        "floor(MemAvailable × 90 / 100)",
        "crates/gh_zero_engine/src/family/mistral/version.rs",
        "Input `#FF5229`, Response `#44BA82`, Secondary `#55B3FB`, and Hint/status `#FFAF01`",
    ] {
        assert!(
            readme_flat.contains(required),
            "root README missing contract phrase: {required}"
        );
    }
    for forbidden in [
        "Questo README",
        "Runtime locale",
        "## Requisiti",
        "## Installazione",
        "## Utilizzo",
        "## Configurazione runtime",
        "## Modelli supportati",
        "## Documentazione",
    ] {
        assert!(
            !readme.contains(forbidden),
            "root README retains superseded Italian prose: {forbidden}"
        );
    }
    assert!(engine_flat.contains("fixture sintetiche"));
    assert!(engine_flat.contains("EngineConfig.context_tokens = None"));
    assert!(engine_flat.contains("floor(MemAvailable × 90 / 100)"));
    assert!(engine_flat.contains("src/family/mistral/version.rs"));
    assert!(backend_flat.contains("floor(MemAvailable × 90 / 100)"));
    assert!(backend_flat.contains("swap"));
    assert!(backend_flat.contains("MemFree"));
    assert!(config_flat.contains("minore tra 32.768 token e il massimo dichiarato dal GGUF"));
    assert!(config_flat.contains("262.144 token"));
    assert!(config_flat.contains("floor(MemAvailable × 90 / 100)"));
    assert!(config_flat.contains("GET /props"));
    assert!(config_flat.contains("../crates/gh_zero_engine/src/family/mistral/version.rs"));
    assert!(ownership.contains("family/mistral/version.rs"));
    assert!(config_flat.contains("--vram-weights-percent 0"));
    assert!(config_flat.contains("sceglie esclusivamente il profilo Cargo"));
    assert!(kv.contains("contesto `4096`"));
    assert!(kv.contains("cpu_layers"));
    assert!(kv.contains("gpu_layers"));
    assert!(readme_flat.contains(
        "Ministral 3 Instruct 2512 support is unchanged for 3B, 8B, and 14B. Reasoning support is limited to Ministral 3 3B Reasoning 2512"
    ));
    assert!(readme_flat.contains(
        "Reasoning output, including `[THINK]` and `[/THINK]`, remains ordinary raw text"
    ));
    assert!(engine_flat.contains("`general.name`. Quest'ultimo seleziona soltanto la policy chat"));
    assert!(engine_flat.contains("`tokenizer.chat_template` non viene eseguito"));
    assert!(engine_flat.contains("un `System` esplicito, anche vuoto, lo sostituisce"));
    assert!(
        engine_flat
            .contains("restano testo raw in `TextDelta`, senza un nuovo evento o canale pubblico")
    );
    assert!(
        architecture_flat
            .contains("`general.name` seleziona soltanto la policy chat dopo il gate capability")
    );
    assert!(ownership.contains("family/mistral/tokenizer/profile.rs"));
    assert!(ownership.contains("family/mistral/parity.rs"));
    assert!(
        config_flat
            .contains("Non esiste né viene aggiunto un flag runtime `--think` o equivalente")
    );
    assert!(
        support_flat
            .contains("parity-check.sh --model PATH --backend cpu|vulkan|hybrid --context 4096")
    );
    for prerequisite in ["`curl`", "`jq`", "`sha256sum`", "`13f2b28b0`"] {
        assert!(
            support.contains(prerequisite),
            "support docs missing parity prerequisite: {prerequisite}"
        );
    }
    for evidence in [
        "2147021472",
        "3652203168",
        "7e9516cc01a039bb3e2d41227cdf388849bc1c942c4624c84567b1684cd9c0fc",
        "3220ac17e246f741f7371e8b0964c399f363a31f9acd2f9a3aacc3bb19fd6466",
        "b9973-13f2b28b0",
        "Quanto fa 17 × 19?",
        "`oracle_top2=pass`",
    ] {
        assert!(
            kv.contains(evidence),
            "KV guide missing evidence: {evidence}"
        );
    }
    let reasoning_matrix = kv
        .split("### Registro 2026-07-26")
        .nth(1)
        .expect("Reasoning matrix section")
        .split("### Regressione emoji e UTF-8")
        .next()
        .expect("Reasoning matrix boundary");
    assert_eq!(
        reasoning_matrix
            .lines()
            .filter(|line| { line.starts_with("| Q4_K_M |") || line.starts_with("| Q8_0 |") })
            .count(),
        12,
        "Reasoning matrix must contain twelve independent rows"
    );
    for line in reasoning_matrix
        .lines()
        .filter(|line| line.starts_with("| Q4_K_M |") || line.starts_with("| Q8_0 |"))
    {
        assert!(
            line.contains("`pass`") && line.contains("`oracle_top2=pass`"),
            "Reasoning matrix row is not a verified pass: {line}"
        );
    }
    assert!(
        !reasoning_matrix.contains("external verification"),
        "completed Reasoning matrix retains an external-verification row"
    );
    let production_args = args.split("#[cfg(test)]").next().unwrap();
    assert!(
        !production_args.contains("--think"),
        "runtime argument table exposes --think"
    );
    assert!(server.contains(
        r#"{"error":{"code":"unsupported_feature","message":"tool calling is not supported","type":"invalid_request_error"}}"#
    ));
    for unsupported_reasoning in [
        "Reasoning 8B support",
        "Reasoning 14B support",
        "supporto Reasoning 8B",
        "supporto Reasoning 14B",
    ] {
        assert!(
            !readme_flat.contains(unsupported_reasoning)
                && !engine_flat.contains(unsupported_reasoning),
            "unsupported Reasoning claim: {unsupported_reasoning}"
        );
    }
    for unsupported in ["Q5_K_M", "Q6_K_M", "Mistral Small", "24B"] {
        assert!(
            !readme_flat.contains(unsupported) && !engine_flat.contains(unsupported),
            "unsupported public claim: {unsupported}"
        );
    }
    assert_local_markdown_links(&root);
}

fn assert_local_markdown_links(root: &Path) {
    let output = Command::new("git")
        .args(["ls-files", "--", "*.md"])
        .current_dir(root)
        .output()
        .expect("list tracked Markdown");
    assert!(output.status.success());
    for relative in String::from_utf8(output.stdout).unwrap().lines() {
        let path = root.join(relative);
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

#[cfg(feature = "hybrid")]
#[test]
fn hybrid_placement_contract() {
    let source = fs::read_to_string(manifest().join("src/family/mistral/hybrid/placement.rs"))
        .expect("hybrid placement source");
    for evidence in [
        "0..=block_count",
        "maximum possible contiguous Vulkan suffix",
        "selects_all_gpu_then_first_fitting_mixed_split",
        "unavailable_or_explicit_zero_gpu_selects_cpu_only",
        "supports_one_layer_suffix_and_heterogeneous_costs",
    ] {
        assert!(
            source.contains(evidence),
            "missing hybrid evidence: {evidence}"
        );
    }
    let report = PlacementReport {
        mode: "mixed",
        cpu_layers: 1,
        gpu_layers: 1,
        cpu: BackendMemory::default(),
        gpu: BackendMemory::default(),
    };
    assert!(report.cpu_layers > 0 && report.gpu_layers > 0);
}

fn push_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn minimal_gguf(architecture: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    push_str(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut bytes, architecture);
    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    bytes
}

#[test]
fn only_mistral3_reaches_the_family_contract() {
    let suffix = format!("{}_{}", std::process::id(), unique());
    let path = std::env::temp_dir().join(format!("gh_zero_{suffix}.gguf"));
    fs::write(&path, minimal_gguf("other")).expect("write fixture");
    let error = Engine::new(&path, EngineConfig::default())
        .err()
        .expect("unsupported architecture")
        .to_string();
    let _ = fs::remove_file(&path);
    assert_eq!(
        error,
        "E03 unsupported architecture 'other'; supported architecture: mistral3"
    );
}

fn unique() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos()
}
