/*
 * graph_horizon_engine — final source and family boundary audit
 * Statically enforces the exact K/I exemption lists, the 200 productive-line
 * orchestration limit, placement-based numeric dispatch, the documentation
 * contract, and absence of obsolete domains. Tiny GGUFs exercise the sole
 * mistral3 architecture gate.
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
use graph_horizon_engine::BackendMemory;
#[cfg(feature = "vulkan-hybrid")]
use graph_horizon_engine::PlacementReport;
use graph_horizon_engine::{
    Engine, EngineConfig, Event, EventSink, KvQuant, Message, Request, Role,
};

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

fn collect(dir: &Path, extension: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, extension, out);
        } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            out.push(path);
        }
    }
}

fn productive_lines(path: &Path) -> usize {
    let text = fs::read_to_string(path).expect("read source");
    let lines = text.lines().collect::<Vec<_>>();
    let mut productive = 0;
    let mut index = 0;
    while index < lines.len() {
        if lines[index] == "#[cfg(test)]" || lines[index].starts_with("#[cfg(all(test,") {
            // Rustfmt leaves top-level items unindented. Skip the complete
            // test-only item, including any additional attributes, without
            // hiding production items that follow it.
            index += 1;
            while index < lines.len()
                && (lines[index].trim().is_empty() || lines[index].starts_with("#["))
            {
                index += 1;
            }
            let block = lines
                .get(index)
                .is_some_and(|line| line.contains('{') && !line.trim_end().ends_with(';'));
            while index < lines.len() {
                let line = lines[index];
                index += 1;
                if (block && line.starts_with('}')) || (!block && line.trim_end().ends_with(';')) {
                    break;
                }
            }
            continue;
        }
        let line = lines[index].trim();
        if !line.is_empty()
            && !line.starts_with("//")
            && !line.starts_with("/*")
            && !line.starts_with('*')
        {
            productive += 1;
        }
        index += 1;
    }
    productive
}

#[test]
fn source_structure() {
    const I: &[&str] = &[
        "src/backend/contract.rs",
        "src/backend/cpu/backend.rs",
        "src/backend/hybrid/contract.rs",
        "src/backend/metal/backend.rs",
        "src/backend/vulkan/backend.rs",
    ];
    const K: &[&str] = &[
        "src/backend/cpu/dequant.rs",
        "src/backend/cpu/kernels/attention/mod.rs",
        "src/backend/cpu/kernels/attention/read_q.rs",
        "src/backend/cpu/kernels/attention/simd.rs",
        "src/backend/cpu/kernels/attention/write.rs",
        "src/backend/cpu/kernels/attention/write_q.rs",
        "src/backend/cpu/kernels/elementwise/activation.rs",
        "src/backend/cpu/kernels/elementwise/embedding.rs",
        "src/backend/cpu/kernels/elementwise/normalization.rs",
        "src/backend/cpu/kernels/elementwise/residual.rs",
        "src/backend/cpu/kernels/elementwise/rope.rs",
        "src/backend/cpu/kernels/matmul/mod.rs",
        "src/backend/cpu/kernels/matmul/q4k.rs",
        "src/backend/cpu/kernels/matmul/q4k_simd.rs",
        "src/backend/cpu/kernels/matmul/q5k.rs",
        "src/backend/cpu/kernels/matmul/q5k_simd.rs",
        "src/backend/cpu/kernels/matmul/q6k.rs",
        "src/backend/cpu/kernels/matmul/q6k_simd.rs",
    ];
    const TEST_FIXTURES: &[&str] = &[
        "src/family/mistral/generation/tests.rs",
        "src/family/mistral/graph/shape.rs",
    ];
    let mut files = Vec::new();
    collect(&manifest().join("src"), "rs", &mut files);
    files.push(manifest().join("build.rs"));
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
        let i_markers = text.matches("AGENTS deroga I:").count();
        let k_markers = text.matches("AGENTS deroga K:").count();
        assert!(i_markers <= 1, "multiple category-I markers: {relative}");
        assert!(k_markers <= 1, "multiple category-K markers: {relative}");
        if i_markers == 1 {
            marked_i.push(relative.clone());
        }
        if k_markers == 1 {
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

    let root = repository();
    let mut root_sources = Vec::new();
    collect(&root.join("src"), "rs", &mut root_sources);
    collect(&root.join("examples"), "rs", &mut root_sources);
    for path in root_sources {
        let relative = path
            .strip_prefix(&root)
            .expect("source under repository")
            .to_string_lossy()
            .replace('\\', "/");
        if relative != "src/support_scripts.rs" && productive_lines(&path) > 200 {
            over.push(format!("{relative}: {}", productive_lines(&path)));
        }
    }
    assert!(
        over.is_empty(),
        "orchestration over 200 lines:\n{}",
        over.join("\n")
    );

    let mut shaders = Vec::new();
    collect(
        &manifest().join("src/backend/metal/shaders"),
        "metal",
        &mut shaders,
    );
    collect(
        &manifest().join("src/backend/vulkan/shaders"),
        "comp",
        &mut shaders,
    );
    for path in shaders {
        let text = fs::read_to_string(&path).expect("read shader");
        assert_eq!(
            text.matches("AGENTS deroga K:").count(),
            1,
            "shader must declare category K exactly once: {}",
            path.display()
        );
    }
}

#[test]
fn hybrid_numeric_dispatch_uses_effective_placement() {
    let metal = manifest().join("src/backend/metal");
    for relative in ["kernels/matmul.rs", "kernels/attention.rs"] {
        let source = fs::read_to_string(metal.join(relative)).expect("Metal dispatcher source");
        assert!(
            !source.contains("feature = \"metal-hybrid\""),
            "{relative} dispatches numerically from a Cargo profile"
        );
        assert!(
            source.contains("mixed_placement"),
            "{relative} does not use effective placement"
        );
    }
    let backend = fs::read_to_string(metal.join("backend.rs")).expect("Metal backend delegator");
    assert_eq!(backend.matches("AGENTS deroga I").count(), 1);
    assert_eq!(backend.matches("self.mixed_placement").count(), 3);
    let contract = fs::read_to_string(manifest().join("src/backend/hybrid/contract.rs"))
        .expect("hybrid device contract");
    assert_eq!(contract.matches("AGENTS deroga I").count(), 1);
    assert!(!contract.contains("mixed_placement"));
    let loader = fs::read_to_string(metal.join("loader.rs")).expect("Metal loader source");
    assert!(loader.contains("selection.layers.start > 0"));
    assert!(loader.contains("!selection.embedding && selection.tail"));
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
        "crates/graph_horizon_engine/src/backend/cpu/kernels/matmul/q8_0.rs",
        "crates/graph_horizon_engine/src/backend/cpu/kernels/matmul/q8_0_simd.rs",
        "crates/graph_horizon_engine/src/backend/vulkan/shaders/embed/embed_q8_0.comp",
        "crates/graph_horizon_engine/src/backend/vulkan/shaders/logits/logits_q8_0.comp",
        "crates/graph_horizon_engine/src/backend/vulkan/shaders/matmul/matmul_q8_0.comp",
    ] {
        assert!(
            !repository().join(relative).exists(),
            "{relative} must be absent"
        );
    }
    let facade = fs::read_to_string(manifest().join("src/lib.rs")).expect("library facade");
    assert!(
        !facade.contains("WeightProfile"),
        "public WeightProfile remains"
    );
    let parser = fs::read_to_string(manifest().join("src/gguf/tensor_index.rs"))
        .expect("GGUF tensor parser");
    for required in [
        "Q8_0,",
        "8 => GgmlType::Q8_0",
        "GgmlType::Q8_0 => \"Q8_0\"",
        "GgmlType::Q8_0 => (32, 34)",
    ] {
        assert!(
            parser.contains(required),
            "Q8 parser support missing: {required}"
        );
    }
    for relative in [
        "docs/tools.md",
        "support/testing/resident-golden.sh",
        "crates/graph_horizon_engine/examples/prefill_trace.rs",
        "crates/graph_horizon_engine/examples/quality.rs",
        "crates/graph_horizon_engine/examples/regression.rs",
        "crates/graph_horizon_engine/examples/validate.rs",
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
    let engine = fs::read_to_string(root.join("crates/graph_horizon_engine/README.md"))
        .expect("engine README");
    let backend = fs::read_to_string(root.join("docs/backend.md")).expect("backend docs");
    let config =
        fs::read_to_string(root.join("docs/configuration.md")).expect("configuration docs");
    let architecture =
        fs::read_to_string(root.join("docs/architecture.md")).expect("architecture docs");
    let ownership = fs::read_to_string(root.join("crates/graph_horizon_engine/src/README.md"))
        .expect("source ownership");
    let support = fs::read_to_string(root.join("support/README.md")).expect("support docs");
    let server = fs::read_to_string(root.join("docs/server.md")).expect("server docs");
    let validation = fs::read_to_string(root.join("VALIDATION.md")).expect("validation register");
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
        "only Ministral 3 2512 `Q4_K_M` GGUF files",
        "Both Instruct and Reasoning are supported at 3B, 8B, and 14B",
        "A Q8 profile is rejected before backend allocation",
        "maximum possible contiguous GPU suffix",
        "Reasoning output, including `[THINK]` and `[/THINK]`, remains ordinary raw text",
        "lesser of 32,768 tokens and the GGUF context maximum",
        "floor(MemAvailable × 90 / 100)",
    ] {
        assert!(
            readme_flat.contains(required),
            "root README missing contract phrase: {required}"
        );
    }
    assert!(engine_flat.contains("suite sintetica"));
    assert!(engine_flat.contains("EngineConfig.context_tokens = None"));
    assert!(engine_flat.contains("floor(MemAvailable × 90 / 100)"));
    assert!(engine_flat.contains("src/family/mistral/version.rs"));
    assert!(engine_flat.contains("Il solo profilo GGUF pubblico è `Q4_K_M`"));
    assert!(engine_flat.contains("`GgmlType::Q8_0`"));
    assert!(backend_flat.contains("Build Backends"));
    assert!(backend_flat.contains("The library crate has no default feature"));
    assert!(
        backend_flat.contains("exactly three numeric backend families: CPU, Vulkan, and Metal")
    );
    assert!(backend_flat.contains("public composition profiles, not numeric backend families"));
    assert!(
        backend_flat.contains(
            "Correct 32-row scratch accounting can change a capacity-bound `AllGpu` split"
        )
    );
    assert!(config_flat.contains("`--context-tokens` requests exactly"));
    assert!(config_flat.contains("GET /props"));
    assert!(ownership.contains("family/mistral/version.rs"));
    assert!(config_flat.contains("`--vram-weights-percent <n>`"));
    assert!(kv.contains("contesto `4096`"));
    assert!(kv.contains("cpu_layers"));
    assert!(kv.contains("gpu_layers"));
    assert!(kv.contains("non dichiara superate prove che non sono state eseguite"));
    assert!(engine_flat.contains("`general.name`. Quest'ultimo seleziona soltanto la policy chat"));
    assert!(engine_flat.contains("`tokenizer.chat_template` non viene eseguito"));
    assert!(engine_flat.contains("un `System` esplicito, anche vuoto, lo sostituisce"));
    assert!(
        engine_flat
            .contains("restano testo raw in `TextDelta`, senza un nuovo evento o canale pubblico")
    );
    assert!(architecture_flat.contains("there is not yet a multi-family registry or dispatcher"));
    assert!(architecture_flat.contains("family/mod.rs"));
    assert!(ownership.contains("family/mistral/tokenizer/profile.rs"));
    assert!(ownership.contains("family/mistral/parity.rs"));
    assert!(support_flat.contains("parity-check.sh --models-dir DIR --model-id ID"));
    assert!(support_flat.contains("Tenta 74 righe seriali"));
    assert!(support_flat.contains("sequenza completa di 16 `local_ids`"));
    let addition = fs::read_to_string(root.join("docs/backend-addition-process.md"))
        .expect("backend addition process");
    assert!(addition.contains("must not select a numeric operation variant"));
    assert!(addition.contains("local to one operation"));
    let performance = fs::read_to_string(root.join("docs/performance-investigation-process.md"))
        .expect("performance process");
    let performance_flat = performance.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(performance.contains("terminal state `keep`"));
    assert!(
        performance_flat
            .contains("`interesting`, `reject`, or `not_verified`, remove the candidate code")
    );
    for prerequisite in ["`curl`", "`jq`", "`sha256sum`", "`13f2b28b0`"] {
        assert!(
            support.contains(prerequisite),
            "support docs missing parity prerequisite: {prerequisite}"
        );
    }
    assert!(
        validation.contains("summary: qualified=6 not_qualified=0 external_verification=0 total=6")
    );
    assert!(validation.contains("summary: pass=1 external_verification=73 failure=0 total=74"));
    for model in ["3b-reasoning", "8b-reasoning", "14b-reasoning"] {
        assert!(
            validation.contains(&format!(
                "model_id={model} profile=reasoning evidence=current status=qualified"
            )),
            "validation register missing current Reasoning row: {model}"
        );
    }
    let production_args = args.split("#[cfg(test)]").next().unwrap();
    assert!(
        !production_args.contains("--think"),
        "runtime argument table exposes --think"
    );
    assert!(server.contains("no tool calling or filesystem access through HTTP"));
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

#[cfg(feature = "vulkan-hybrid")]
#[test]
fn hybrid_placement_contract() {
    let source = fs::read_to_string(manifest().join("src/backend/hybrid/placement/separate.rs"))
        .expect("hybrid placement source");
    for evidence in [
        "first_split..=block_count",
        "selects_all_gpu_mixed_one_layer_suffix_and_cpu_only",
        "context_failure_does_not_reduce_context",
        "model does not fit available RAM and VRAM",
        "context {} does not fit the selected backend",
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

#[test]
#[ignore = "requires an authenticated Ministral model and oracle vectors"]
fn real_selected_runtime_parity_and_lifecycle() {
    let model = std::env::var("GRAPH_HORIZON_MODEL").expect("GRAPH_HORIZON_MODEL required");
    let context = required_usize("GRAPH_HORIZON_CONTEXT");
    assert_eq!(context, 4096, "GRAPH_HORIZON_CONTEXT must be 4096");
    let kv = std::env::var("GRAPH_HORIZON_KV").expect("GRAPH_HORIZON_KV required");
    let kv_quant = KvQuant::parse(&kv).expect("GRAPH_HORIZON_KV must be f16 or int8");
    let percentage = std::env::var("GRAPH_HORIZON_VRAM_WEIGHTS_PERCENT")
        .ok()
        .map(|value| value.parse::<u8>().expect("invalid hybrid percentage"));
    let engine = Engine::new(
        Path::new(&model),
        EngineConfig {
            context_tokens: Some(context),
            vram_weights_percent: percentage,
            kv_quant,
            ..EngineConfig::default()
        },
    )
    .expect("load selected runtime");
    assert_placement(&engine, percentage);

    let prompt = std::env::var("GRAPH_HORIZON_REFERENCE_PROMPT_IDS")
        .expect("GRAPH_HORIZON_REFERENCE_PROMPT_IDS required");
    let completion = std::env::var("GRAPH_HORIZON_REFERENCE_COMPLETION_IDS")
        .expect("GRAPH_HORIZON_REFERENCE_COMPLETION_IDS required");
    let report = graph_horizon_engine::harness::validate_parity(&engine, &prompt, &completion)
        .expect("selected-runtime parity");
    assert_eq!(report.local_ids.len(), 16);
    assert_eq!(report.top_two.len(), 16);

    let mut terminal = Vec::new();
    engine.generate(parity_request(), &mut |event| {
        terminal.push(event);
        true
    });
    assert_eq!(
        terminal
            .iter()
            .filter(|event| matches!(event, Event::Finished(_) | Event::Error(_)))
            .count(),
        1
    );
    assert!(matches!(terminal.last(), Some(Event::Finished(_))));

    let mut cancelled = Cancelled { events: Vec::new() };
    engine.generate(parity_request(), &mut cancelled);
    assert!(cancelled.events.is_empty());
    println!(
        "ministral-parity: local_ids={} oracle_top2=pass crossings={}",
        report
            .local_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        report.crossings
    );
}

fn required_usize(name: &str) -> usize {
    std::env::var(name)
        .unwrap_or_else(|_| panic!("{name} required"))
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be an unsigned decimal"))
}

fn assert_placement(engine: &Engine, percentage: Option<u8>) {
    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    {
        let expected = std::env::var("GRAPH_HORIZON_EXPECTED_MODE")
            .expect("GRAPH_HORIZON_EXPECTED_MODE required for hybrid profiles");
        let placement = engine.placement().expect("hybrid placement report");
        assert_eq!(placement.mode, expected);
        assert!(percentage.is_some());
        if expected == "mixed" {
            assert!(placement.cpu_layers > 0 && placement.gpu_layers > 0);
        }
        if expected == "cpu-only" {
            assert_eq!(placement.gpu, BackendMemory::default());
        }
        for bytes in [placement.cpu, placement.gpu] {
            assert_eq!(
                bytes.total,
                bytes.weights
                    + bytes.kv
                    + bytes.scratch
                    + bytes.fixed
                    + bytes.staging
                    + bytes.crossing
                    + bytes.reserve
            );
        }
    }
    #[cfg(not(any(feature = "vulkan-hybrid", feature = "metal-hybrid")))]
    {
        assert!(percentage.is_none());
        assert!(engine.placement().is_none());
    }
}

fn parity_request() -> Request {
    Request {
        messages: vec![
            Message {
                role: Role::System,
                content: String::new(),
            },
            Message {
                role: Role::User,
                content: "Quanto fa 17 × 19?".into(),
            },
        ],
        sampling: graph_horizon_engine::SamplingParams::greedy(),
        max_tokens: 16,
    }
}

struct Cancelled {
    events: Vec<Event>,
}

impl EventSink for Cancelled {
    fn cancelled(&self) -> bool {
        true
    }

    fn emit(&mut self, event: Event) -> bool {
        self.events.push(event);
        false
    }
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
    let path = std::env::temp_dir().join(format!("graph_horizon_{suffix}.gguf"));
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
