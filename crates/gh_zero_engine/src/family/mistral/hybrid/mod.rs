/*
 * gh_zero_engine — immutable Ministral hybrid placement contract
 * Defines the three legal contiguous placement states and their exact byte
 * report. Arithmetic lives in `weights`/`placement`; allocation and execution
 * consume this value without mutating or reconsidering the selected split.
 */

pub(crate) mod forward;
pub(crate) mod loader;
pub(crate) mod prefill;

pub(crate) use crate::backend::hybrid::BackendBytes;
#[cfg(test)]
pub(crate) use crate::backend::hybrid::HybridMode;
pub(crate) use crate::backend::hybrid::crossing;
pub(crate) type LoadedHybrid =
    crate::backend::hybrid::HybridRuntime<crate::backend::vulkan::VulkanBackend>;
pub(crate) type HybridBackends =
    crate::backend::hybrid::HybridBackends<crate::backend::vulkan::VulkanBackend>;

#[cfg(test)]
mod graph {
    use std::process::Command;

    use crate::api::engine::EngineConfig;
    use crate::api::event::Event;
    use crate::api::message::{Message, Role};
    use crate::api::request::{Request, SamplingParams};
    use crate::backend::Backend;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::cpu::{CpuBuffer, CpuFormat};
    use crate::backend::vulkan::VulkanBackend;
    use crate::family::mistral::graph::prefill;
    use crate::family::mistral::parity;
    use crate::family::mistral::{MistralContract, MistralModel, RuntimeModel, run, template};
    use crate::gguf::loader::GgufFile;
    use crate::kv_cache;
    use crate::kv_cache::scheme::KvQuant;
    use crate::sampling::{self, Rng};

    use super::{HybridMode, crossing, forward};

    #[test]
    fn one_mixed_crossing_and_zero_homogeneous_crossings() {
        crossing::reset_count();
        assert_eq!(crossing::count(), 0);

        let gpu = match VulkanBackend::bare() {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return;
            }
        };
        let source = CpuBuffer::zeroed(16, CpuFormat::F32);
        source.write_f32(&[1.0, -2.0, 3.5, 4.0]);
        let target = gpu.alloc_buffer(16).expect("crossing target");
        crossing::copy(&source, &gpu, &target, 4).expect("mixed crossing");
        assert_eq!(crossing::count(), 1);
        let bytes = gpu.read_bytes(&target, 16).expect("crossed residual");
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, [1.0, -2.0, 3.5, 4.0]);
        gpu.free_buffer(target);

        let short = CpuBuffer::zeroed(4, CpuFormat::F32);
        assert!(crossing::copy(&short, &gpu, &gpu.buffers().scratch.x, 2).is_err());
        assert_eq!(crossing::count(), 1);
    }

    #[test]
    fn graph_routes_have_one_mixed_crossing_site() {
        let decode = include_str!("forward.rs");
        let prefill = include_str!("prefill.rs");
        assert_eq!(decode.matches("crossing::copy(").count(), 1);
        assert_eq!(prefill.matches("crossing::copy(").count(), 1);
        // Homogeneous arms contain no crossing site; mixed prefill invokes its
        // sole site once for every batch, including a final one-row batch.
        crossing::reset_count();
        assert_eq!(crossing::count(), 0);
    }

    #[test]
    #[ignore = "requires pinned 3B Q4_K_M model, target Vulkan budget, and reference CLI"]
    fn hybrid_3b_q4_k_m_mixed_chat() {
        let path = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
        let context = env_usize("GH_ZERO_CONTEXT", 4096);
        let percent = env_usize("GH_ZERO_VRAM_WEIGHTS_PERCENT", 25) as u8;
        let scheme = std::env::var("GH_ZERO_KV")
            .ok()
            .and_then(|value| KvQuant::parse(&value))
            .unwrap_or(KvQuant::F16);
        let file = GgufFile::open(std::path::Path::new(&path)).expect("open pinned GGUF");
        let contract = MistralContract::from_gguf(&file).expect("Ministral contract");
        let messages = vec![Message {
            role: Role::User,
            content: "Hello".into(),
        }];
        let prompt = template::render(&messages, &contract.tokenizer, context).expect("prompt");
        let reference_prompt = reference_ids(&path, "[INST]Hello[/INST]", false);
        assert_eq!(prompt, reference_prompt);
        let reference_first = reference_first(&path, context);

        let cpu = MistralModel::<CpuBackend>::load(&file, context).expect("CPU model");
        let cpu_kv = kv_cache::alloc_shape(
            &cpu.backend,
            cpu.config.block_count,
            context,
            cpu.config.kv_head_count,
            cpu.config.key_length,
            cpu.config.value_length,
            scheme,
        )
        .expect("CPU KV");
        prefill::prefill_with(&cpu.backend, &cpu.config, &cpu_kv, &prompt, || Ok(()))
            .expect("CPU prefill");
        let cpu_first = cpu
            .backend
            .read_argmax(&cpu.backend.buffers().logits, cpu.config.vocab_size)
            .expect("CPU first token");
        kv_cache::free(&cpu.backend, cpu_kv);
        drop(cpu);

        let runtime = RuntimeModel::load(
            &file,
            &EngineConfig {
                context_tokens: Some(context),
                vram_weights_percent: Some(percent),
                kv_quant: scheme,
                ..EngineConfig::default()
            },
        )
        .expect("hybrid model");
        let plan = &runtime.backend.plan;
        println!(
            "mode={} cpu_layers={} gpu_layers={} cpu_bytes={} gpu_bytes={}",
            plan.mode.name(),
            plan.cpu_layers,
            plan.gpu_layers,
            plan.cpu.total,
            plan.gpu.total
        );
        assert_eq!(plan.mode, HybridMode::Mixed);
        assert!(plan.cpu_layers > 0 && plan.gpu_layers > 0);

        let hybrid_kv =
            forward::RequestKv::new(&runtime.backend, &runtime.config, context, runtime.scheme)
                .expect("hybrid KV");
        super::prefill::run(&hybrid_kv, &runtime.config, &prompt, || Ok(()))
            .expect("hybrid prefill");
        let mut hybrid_logits =
            forward::read_logits(&hybrid_kv, runtime.config.vocab_size).expect("hybrid logits");
        let hybrid_first = sampling::sample(
            &mut hybrid_logits,
            &SamplingParams::greedy(),
            &prompt,
            &mut Rng::new(0),
        );
        drop(hybrid_kv);
        println!(
            "kv={} prompt_ids={prompt:?} reference_first={reference_first} cpu_first={cpu_first} hybrid_first={hybrid_first}",
            scheme.name()
        );
        assert_eq!(cpu_first, reference_first);
        assert_eq!(hybrid_first, reference_first);

        let mut events = Vec::new();
        run::generate(
            &runtime,
            Request {
                messages,
                sampling: SamplingParams::greedy(),
                // One token is enough to cross the public lifecycle boundary
                // without deliberately truncating a later UTF-8 code point.
                max_tokens: 1,
            },
            &mut |event| {
                events.push(event);
                true
            },
        );
        assert!(matches!(events.last(), Some(Event::Finished(_))));
        assert!(!events.iter().any(|event| matches!(event, Event::Error(_))));
    }

    #[test]
    #[ignore = "requires an approved Ministral model, mixed Vulkan placement, and oracle IDs"]
    fn real_ministral_parity() {
        let path = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
        let context = std::env::var("GH_ZERO_CONTEXT")
            .expect("GH_ZERO_CONTEXT required")
            .parse::<usize>()
            .expect("GH_ZERO_CONTEXT must be an unsigned decimal");
        assert_eq!(context, parity::CONTEXT, "GH_ZERO_CONTEXT must be 4096");
        let percent = std::env::var("GH_ZERO_VRAM_WEIGHTS_PERCENT")
            .expect("GH_ZERO_VRAM_WEIGHTS_PERCENT required")
            .parse::<u8>()
            .expect("GH_ZERO_VRAM_WEIGHTS_PERCENT must be an unsigned decimal");
        assert_eq!(
            percent, 25,
            "GH_ZERO_VRAM_WEIGHTS_PERCENT must be exactly 25"
        );
        let scheme = std::env::var("GH_ZERO_KV")
            .ok()
            .and_then(|value| KvQuant::parse(&value))
            .expect("GH_ZERO_KV must be f16 or int8");
        let reference = parity::reference_vectors();

        match crate::backend::vulkan::hybrid_device() {
            Ok(Some(device)) => drop(device),
            Ok(None) => {
                println!("external verification: Vulkan backend unavailable");
                return;
            }
            Err(error) => panic!("Vulkan initialization failed: {error}"),
        }
        let file = GgufFile::open(std::path::Path::new(&path)).expect("open approved GGUF");
        let contract = MistralContract::from_gguf(&file).expect("Ministral contract");
        let prompt = template::render(&parity::conversation(), &contract.tokenizer, context)
            .expect("Ministral prompt");
        parity::assert_exact("prompt IDs", &prompt, &reference.prompt);

        let runtime = RuntimeModel::load(
            &file,
            &EngineConfig {
                context_tokens: Some(context),
                vram_weights_percent: Some(percent),
                kv_quant: scheme,
                ..EngineConfig::default()
            },
        )
        .expect("hybrid model");
        let plan = &runtime.backend.plan;
        assert_eq!(plan.mode, HybridMode::Mixed, "mixed placement required");
        assert!(plan.cpu_layers > 0 && plan.gpu_layers > 0);
        let kv =
            forward::RequestKv::new(&runtime.backend, &runtime.config, context, runtime.scheme)
                .expect("hybrid KV");
        super::prefill::run(&kv, &runtime.config, &prompt, || Ok(())).expect("hybrid prefill");
        let mut actual = Vec::with_capacity(parity::TOKEN_COUNT);
        for step in 0..parity::TOKEN_COUNT {
            let token =
                forward::read_argmax(&kv, runtime.config.vocab_size).expect("hybrid argmax");
            actual.push(token);
            if step + 1 < parity::TOKEN_COUNT {
                forward::token(&kv, &runtime.config, token, prompt.len() + step)
                    .expect("hybrid decode");
            }
        }
        drop(kv);
        assert_eq!(actual.len(), parity::TOKEN_COUNT);
        let teacher =
            forward::RequestKv::new(&runtime.backend, &runtime.config, context, runtime.scheme)
                .expect("hybrid teacher KV");
        super::prefill::run(&teacher, &runtime.config, &prompt, || Ok(()))
            .expect("hybrid teacher prefill");
        let mut top2 = Vec::with_capacity(parity::TOKEN_COUNT);
        for (step, &token) in reference.completion.iter().enumerate() {
            top2.push(
                forward::read_topk(&teacher, runtime.config.vocab_size, 2)
                    .expect("hybrid top two")
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect(),
            );
            if step + 1 < parity::TOKEN_COUNT {
                forward::token(&teacher, &runtime.config, token, prompt.len() + step)
                    .expect("hybrid teacher decode");
            }
        }
        println!(
            "mode={} cpu_layers={} gpu_layers={} kv={} prompt_ids={prompt:?} reference_completion_ids={:?} hybrid_ids={actual:?} teacher_top2={top2:?}",
            plan.mode.name(),
            plan.cpu_layers,
            plan.gpu_layers,
            scheme.name(),
            reference.completion
        );
        parity::assert_oracle_top2(&top2, &reference.completion);
        println!(
            "ministral-parity: local_ids={} oracle_top2=pass",
            parity::csv(&actual)
        );
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn reference_first(model: &str, context: usize) -> u32 {
        let binary =
            std::env::var("GH_ZERO_REFERENCE_CLI").expect("GH_ZERO_REFERENCE_CLI required");
        let output = Command::new("timeout")
            .arg("300")
            .arg(binary)
            .args([
                "-m",
                model,
                "-p",
                "[INST]Hello[/INST]",
                "-c",
                &context.to_string(),
                "-n",
                "1",
                "--temp",
                "0",
                "--top-k",
                "1",
                "--top-p",
                "1",
                "--min-p",
                "0",
                "--repeat-penalty",
                "1",
                "--no-display-prompt",
                "--no-warmup",
                "--no-perf",
                "-no-cnv",
                "--no-jinja",
            ])
            .output()
            .expect("run llama-completion");
        assert!(output.status.success(), "llama-completion failed");
        reference_ids(
            model,
            std::str::from_utf8(&output.stdout)
                .expect("reference UTF-8")
                .trim_end(),
            true,
        )[0]
    }

    fn reference_ids(model: &str, text: &str, no_bos: bool) -> Vec<u32> {
        let binary = std::env::var("GH_ZERO_REFERENCE_TOKENIZE")
            .expect("GH_ZERO_REFERENCE_TOKENIZE required");
        let mut command = Command::new(binary);
        command.args(["--log-disable", "--ids"]);
        if no_bos {
            command.arg("--no-bos");
        }
        let output = command
            .args(["-m", model, "-p", text])
            .output()
            .expect("run llama-tokenize");
        assert!(output.status.success(), "llama-tokenize failed");
        std::str::from_utf8(&output.stdout)
            .expect("token IDs UTF-8")
            .split_whitespace()
            .filter_map(|word| {
                word.trim_matches(|c| c == '[' || c == ']' || c == ',')
                    .parse()
                    .ok()
            })
            .collect()
    }
}
