/*
 * gh_zero_engine — real Ministral Vulkan parity test
 * Owns family-specific real GGUF checks, including the shared exact-vector
 * Reasoning parity contract. It is compiled only for Vulkan tests and does not
 * alter runtime family selection.
 */

use std::process::Command;

use crate::api::message::{Message, Role};
use crate::backend::Backend;
use crate::backend::vulkan::VulkanBackend;
use crate::gguf::loader::GgufFile;
use crate::kv_cache::scheme::KvQuant;

use super::{MistralConfig, MistralContract, MistralModel, template};
use super::{parity, version};

#[test]
fn cpu_vulkan_parity_shared_graph_shape_traces_versioned_rows() {
    use crate::family::mistral::graph::shape::{ShapeBackend, config};
    use crate::family::mistral::version::REFERENCE_ROWS;
    use crate::kv_cache;

    for row in REFERENCE_ROWS {
        let cfg = config(row.0, row.1, row.2);
        let cpu_trace = ShapeBackend::new(&cfg, false);
        let cpu_kv = kv_cache::alloc_shape(
            &cpu_trace,
            cfg.block_count,
            1,
            cfg.kv_head_count,
            cfg.key_length,
            cfg.value_length,
            KvQuant::F16,
        )
        .unwrap();
        crate::family::mistral::graph::forward::token(&cpu_trace, &cfg, &cpu_kv, 3, 0).unwrap();
        kv_cache::free(&cpu_trace, cpu_kv);

        let vulkan_trace = ShapeBackend::new(&cfg, false);
        let vulkan_kv = kv_cache::alloc_shape(
            &vulkan_trace,
            cfg.block_count,
            1,
            cfg.kv_head_count,
            cfg.key_length,
            cfg.value_length,
            KvQuant::F16,
        )
        .unwrap();
        crate::family::mistral::graph::forward::token(&vulkan_trace, &cfg, &vulkan_kv, 3, 0)
            .unwrap();
        kv_cache::free(&vulkan_trace, vulkan_kv);
        assert_eq!(cpu_trace.trace(), vulkan_trace.trace());
    }
}

#[test]
fn cpu_vulkan_parity_shared_graph_prefill_and_mixed_weights() {
    use crate::backend::buffers::{Buffers, LayerWeights, Scratch, WeightSet};
    use crate::backend::vulkan::buffers::{GpuBuffer, WeightFormat};
    use crate::family::mistral::graph::prefill;
    use crate::kv_cache;

    fn config() -> MistralConfig {
        MistralConfig {
            block_count: 1,
            context_length: 8,
            embedding_length: 256,
            feed_forward_length: 256,
            head_count: 2,
            kv_head_count: 1,
            key_length: 128,
            value_length: 128,
            q_width: 256,
            k_width: 128,
            v_width: 128,
            attention_width: 256,
            rope_dimension: 128,
            rope_freq_base: 10_000.0,
            rms_epsilon: 0.00001,
            yarn_factor: 2.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_log_multiplier: 0.1,
            yarn_original_context: 8,
            attention_temperature_scale: 1.1,
            vocab_size: 32,
            bos_id: 1,
            eos_id: 2,
        }
    }

    fn f16_bits(value: f32) -> [u8; 2] {
        crate::kv_cache::int8::f32_to_f16(value).to_le_bytes()
    }

    fn norm(backend: &VulkanBackend, dim: usize) -> GpuBuffer {
        let bytes: Vec<u8> = (0..dim).flat_map(|_| f16_bits(1.0)).collect();
        upload(backend, bytes, WeightFormat::F16)
    }

    fn weight(backend: &VulkanBackend, input: usize, output: usize, seed: usize) -> GpuBuffer {
        let (format, block_bytes) = if seed.is_multiple_of(2) {
            (WeightFormat::Q6K, 210)
        } else {
            (WeightFormat::Q4K, 144)
        };
        let mut bytes = vec![0u8; output * (input / 256) * block_bytes];
        for chunk in bytes.chunks_exact_mut(block_bytes) {
            match format {
                WeightFormat::Q4K => {
                    chunk[0..2].copy_from_slice(&f16_bits(0.02));
                    chunk[2..4].copy_from_slice(&f16_bits(0.005));
                    chunk[4..12].fill(1);
                    for (i, byte) in chunk[16..].iter_mut().enumerate() {
                        *byte = ((i + seed) as u8 & 0x0f) | ((((i + seed + 3) as u8) & 0x0f) << 4);
                    }
                }
                WeightFormat::Q6K => {
                    for (i, byte) in chunk[..192].iter_mut().enumerate() {
                        *byte = (i + seed) as u8;
                    }
                    chunk[192..208].fill(1);
                    chunk[208..210].copy_from_slice(&f16_bits(0.01));
                }
                _ => unreachable!(),
            }
        }
        upload(backend, bytes, format)
    }

    fn upload(backend: &VulkanBackend, bytes: Vec<u8>, format: WeightFormat) -> GpuBuffer {
        let mut buffer = backend.alloc_buffer(bytes.len() as u64).unwrap();
        buffer.quant = format;
        backend.upload_bytes(&buffer, &bytes).unwrap();
        buffer
    }

    fn backend(cfg: &MistralConfig) -> color_eyre::eyre::Result<VulkanBackend> {
        let mut backend = VulkanBackend::bare()?;
        let z = |n| backend.alloc_buffer(n as u64 * 2).unwrap();
        let layer = LayerWeights {
            attn_norm: norm(&backend, cfg.embedding_length),
            attn_q: weight(&backend, cfg.embedding_length, cfg.q_width, 1),
            attn_k: weight(&backend, cfg.embedding_length, cfg.k_width, 2),
            attn_v: weight(&backend, cfg.embedding_length, cfg.v_width, 3),
            attn_output: weight(&backend, cfg.attention_width, cfg.embedding_length, 4),
            ffn_norm: norm(&backend, cfg.embedding_length),
            ffn_gate: weight(&backend, cfg.embedding_length, cfg.feed_forward_length, 5),
            ffn_up: weight(&backend, cfg.embedding_length, cfg.feed_forward_length, 6),
            ffn_down: weight(&backend, cfg.feed_forward_length, cfg.embedding_length, 7),
        };
        let buffers = Buffers {
            weights: WeightSet {
                token_embd: Some(weight(&backend, cfg.embedding_length, cfg.vocab_size, 8)),
                output_norm: Some(norm(&backend, cfg.embedding_length)),
                output: None,
                layers: vec![layer],
            },
            scratch: Scratch {
                x: backend.alloc_buffer((cfg.embedding_length * 4) as u64)?,
                normed: z(cfg.embedding_length),
                q: z(cfg.q_width),
                k: z(cfg.k_width),
                v: z(cfg.v_width),
                attn: z(cfg.attention_width),
                proj: z(cfg.embedding_length),
                gate: z(cfg.feed_forward_length),
                up: z(cfg.feed_forward_length),
                act: z(cfg.feed_forward_length),
                ffn_out: z(cfg.embedding_length),
            },
            logits: backend.alloc_buffer((cfg.vocab_size * 4) as u64)?,
        };
        backend.replace_test_buffers(buffers, cfg.vocab_size)?;
        Ok(backend)
    }

    fn logits(cfg: &MistralConfig, scheme: KvQuant, batched: bool) -> Option<Vec<f32>> {
        let backend = match backend(cfg) {
            Ok(backend) => backend,
            Err(_) => {
                eprintln!("external verification: no Vulkan device");
                return None;
            }
        };
        let kv = kv_cache::alloc_shape(
            &backend,
            cfg.block_count,
            cfg.context_length,
            cfg.kv_head_count,
            cfg.key_length,
            cfg.value_length,
            scheme,
        )
        .unwrap();
        let prompt = [1, 3, 4, 5, 6];
        if batched {
            prefill::prefill_with(&backend, cfg, &kv, &prompt, || Ok(())).unwrap();
        } else {
            for (pos, &token) in prompt.iter().enumerate() {
                super::graph::forward::token(&backend, cfg, &kv, token, pos).unwrap();
            }
        }
        let result = backend
            .read_logits(&backend.buffers().logits, cfg.vocab_size)
            .unwrap();
        kv_cache::free(&backend, kv);
        Some(result)
    }

    let cfg = config();
    for scheme in [KvQuant::F16, KvQuant::Int8] {
        let Some(sequential) = logits(&cfg, scheme, false) else {
            return;
        };
        let batched = logits(&cfg, scheme, true).unwrap();
        let max = sequential
            .iter()
            .zip(&batched)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("shared_graph_{} max_err={max}", scheme.name());
        assert!(max <= 0.05, "shared graph prefill error {max}");
        assert_eq!(
            argmax(&sequential),
            argmax(&batched),
            "shared graph greedy id"
        );
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap()
}

#[test]
#[ignore]
fn real_greedy_parity() {
    let model = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
    let context = std::env::var("GH_ZERO_CONTEXT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4096);
    let scheme = std::env::var("GH_ZERO_KV")
        .ok()
        .and_then(|value| KvQuant::parse(&value))
        .unwrap_or(KvQuant::F16);
    let prompt_text = "[INST]Hello[/INST]";
    let file = GgufFile::open(std::path::Path::new(&model)).expect("open GGUF");
    let contract = MistralContract::from_gguf(&file).expect("Ministral contract");
    let messages = [Message {
        role: Role::User,
        content: "Hello".into(),
    }];
    let prompt = template::render(&messages, &contract.tokenizer, context).expect("prompt");
    let reference_prompt = reference_prompt_ids(&model, prompt_text);
    assert_eq!(prompt, reference_prompt);
    let reference_first = reference_first_id(&model, prompt_text, context);

    let model = match MistralModel::<VulkanBackend>::load(&file, context) {
        Ok(model) => model,
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.starts_with("Vulkan memory is insufficient: required ")
                    || msg
                        == format!(
                            "context {context} does not fit the selected backend; context was not reduced"
                        ),
                "{msg}"
            );
            println!("vulkan real parity expected refusal: {msg}");
            return;
        }
    };
    let kv = crate::kv_cache::alloc_shape(
        &model.backend,
        model.config.block_count,
        context,
        model.config.kv_head_count,
        model.config.key_length,
        model.config.value_length,
        scheme,
    )
    .expect("Vulkan KV");
    crate::family::mistral::graph::prefill::prefill_with(
        &model.backend,
        &model.config,
        &kv,
        &prompt,
        || Ok(()),
    )
    .expect("Vulkan prefill");
    let got = model
        .backend
        .read_argmax(&model.backend.buffers().logits, model.config.vocab_size)
        .expect("Vulkan greedy token");
    crate::kv_cache::free(&model.backend, kv);
    println!(
        "profile=Q4_K_M kv={} prompt_ids={prompt:?} reference_first={reference_first} vulkan_first={got}",
        scheme.name()
    );
    assert_eq!(got, reference_first);
}

#[test]
#[ignore = "requires an approved Reasoning model, Vulkan, and oracle IDs"]
fn real_reasoning_parity() {
    let path = std::env::var("GH_ZERO_MODEL").expect("GH_ZERO_MODEL required");
    let context = std::env::var("GH_ZERO_CONTEXT")
        .expect("GH_ZERO_CONTEXT required")
        .parse::<usize>()
        .expect("GH_ZERO_CONTEXT must be an unsigned decimal");
    assert_eq!(context, parity::CONTEXT, "GH_ZERO_CONTEXT must be 4096");
    let scheme = std::env::var("GH_ZERO_KV")
        .ok()
        .and_then(|value| KvQuant::parse(&value))
        .expect("GH_ZERO_KV must be f16 or int8");
    let reference = parity::reference_vectors();
    let file = GgufFile::open(std::path::Path::new(&path)).expect("open approved GGUF");
    let contract = MistralContract::from_gguf(&file).expect("Reasoning contract");
    let prompt = template::render(
        &[Message {
            role: Role::User,
            content: parity::USER_CONTENT.into(),
        }],
        &contract.tokenizer,
        context,
    )
    .expect("Reasoning prompt");
    parity::assert_exact("prompt IDs", &prompt, &reference.prompt);

    let model = match MistralModel::<VulkanBackend>::load(&file, context) {
        Ok(model) => model,
        Err(error) if error.to_string() == "Vulkan backend is unavailable" => {
            println!("external verification: Vulkan backend unavailable");
            return;
        }
        Err(error) => panic!("Vulkan load failed: {error}"),
    };
    let actual = vulkan_completion(&model, &prompt, context, scheme);
    let top2 = vulkan_teacher_top2(&model, &prompt, context, scheme, &reference.completion);
    println!(
        "profile=Q4_K_M release={} kv={} prompt_ids={prompt:?} reference_completion_ids={:?} vulkan_ids={actual:?} teacher_top2={top2:?}",
        version::RELEASE,
        scheme.name(),
        reference.completion
    );
    parity::assert_oracle_top2(&top2, &reference.completion);
    println!(
        "reasoning-parity: local_ids={} oracle_top2=pass",
        parity::csv(&actual)
    );
}

fn vulkan_completion(
    model: &MistralModel<VulkanBackend>,
    prompt: &[u32],
    context: usize,
    scheme: KvQuant,
) -> Vec<u32> {
    let kv = crate::kv_cache::alloc_shape(
        &model.backend,
        model.config.block_count,
        context,
        model.config.kv_head_count,
        model.config.key_length,
        model.config.value_length,
        scheme,
    )
    .expect("Vulkan KV");
    let result = (|| {
        crate::family::mistral::graph::prefill::prefill_with(
            &model.backend,
            &model.config,
            &kv,
            prompt,
            || Ok(()),
        )?;
        let mut ids = Vec::with_capacity(parity::TOKEN_COUNT);
        for step in 0..parity::TOKEN_COUNT {
            let token = model
                .backend
                .read_argmax(&model.backend.buffers().logits, model.config.vocab_size)?;
            ids.push(token);
            if step + 1 < parity::TOKEN_COUNT {
                crate::family::mistral::graph::forward::token(
                    &model.backend,
                    &model.config,
                    &kv,
                    token,
                    prompt.len() + step,
                )?;
            }
        }
        color_eyre::eyre::Ok(ids)
    })();
    crate::kv_cache::free(&model.backend, kv);
    result.expect("Vulkan completion")
}

fn vulkan_teacher_top2(
    model: &MistralModel<VulkanBackend>,
    prompt: &[u32],
    context: usize,
    scheme: KvQuant,
    reference: &[u32],
) -> Vec<Vec<u32>> {
    let kv = crate::kv_cache::alloc_shape(
        &model.backend,
        model.config.block_count,
        context,
        model.config.kv_head_count,
        model.config.key_length,
        model.config.value_length,
        scheme,
    )
    .expect("Vulkan teacher KV");
    let result = (|| {
        crate::family::mistral::graph::prefill::prefill_with(
            &model.backend,
            &model.config,
            &kv,
            prompt,
            || Ok(()),
        )?;
        let mut top2 = Vec::with_capacity(reference.len());
        for (step, &token) in reference.iter().enumerate() {
            top2.push(
                model
                    .backend
                    .read_topk(&model.backend.buffers().logits, model.config.vocab_size, 2)?
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect(),
            );
            if step + 1 < reference.len() {
                crate::family::mistral::graph::forward::token(
                    &model.backend,
                    &model.config,
                    &kv,
                    token,
                    prompt.len() + step,
                )?;
            }
        }
        color_eyre::eyre::Ok(top2)
    })();
    crate::kv_cache::free(&model.backend, kv);
    result.expect("Vulkan teacher forcing")
}

fn reference_prompt_ids(model: &str, prompt: &str) -> Vec<u32> {
    let bin =
        std::env::var("GH_ZERO_REFERENCE_TOKENIZE").expect("GH_ZERO_REFERENCE_TOKENIZE required");
    let out = Command::new(bin)
        .args(["--log-disable", "--ids", "-m", model, "-p", prompt])
        .output()
        .expect("run llama-tokenize");
    assert!(out.status.success(), "llama-tokenize failed");
    parse_ids(std::str::from_utf8(&out.stdout).expect("utf8 token ids"))
}

fn reference_first_id(model: &str, prompt: &str, context: usize) -> u32 {
    let bin = std::env::var("GH_ZERO_REFERENCE_CLI").expect("GH_ZERO_REFERENCE_CLI required");
    let out = Command::new("timeout")
        .arg("300")
        .arg(bin)
        .args([
            "-m",
            model,
            "-p",
            prompt,
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
    assert!(out.status.success(), "llama-completion failed");
    let text = std::str::from_utf8(&out.stdout)
        .expect("utf8 completion")
        .trim_end();
    let tokenizer =
        std::env::var("GH_ZERO_REFERENCE_TOKENIZE").expect("GH_ZERO_REFERENCE_TOKENIZE required");
    let ids = Command::new(tokenizer)
        .args([
            "--log-disable",
            "--ids",
            "--no-bos",
            "-m",
            model,
            "-p",
            text,
        ])
        .output()
        .expect("tokenize reference completion");
    assert!(ids.status.success(), "reference tokenization failed");
    parse_ids(std::str::from_utf8(&ids.stdout).expect("utf8 token ids"))[0]
}

fn parse_ids(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .filter_map(|word| {
            word.trim_matches(|c| c == '[' || c == ']' || c == ',')
                .parse()
                .ok()
        })
        .collect()
}
