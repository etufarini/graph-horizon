/*
 * graph_horizon_engine — Ministral generation tests
 * Verifies lifecycle, cancellation, cleanup, tokenization, and numeric parity.
 * Test fixtures own concrete backends; production generation remains neutral.
 */

use super::*;
use crate::api::event::Event;
use crate::backend::Backend;
use crate::family::mistral::MistralConfig;
use crate::family::mistral::MistralModel;
use crate::family::mistral::graph::shape::{ShapeBackend, config};
use crate::family::mistral::graph::{forward, prefill};
use crate::kv_cache::scheme::KvQuant;
use crate::kv_cache::{self, Kv};
use color_eyre::eyre::bail;
use std::cell::Cell;
use std::rc::Rc;

struct Request {
    prompt: Vec<u32>,
    max_tokens: usize,
    context: usize,
    kv_quant: crate::kv_cache::scheme::KvQuant,
}

#[derive(Default)]
struct Events {
    cancelled: bool,
    values: Vec<Event>,
}

impl EventSink for Events {
    fn cancelled(&self) -> bool {
        self.cancelled
    }

    fn emit(&mut self, event: Event) -> bool {
        self.values.push(event);
        !self.cancelled
    }
}

#[test]
fn error_matrix_e18_e19_has_one_terminal_or_silent_cancel() {
    let stats = GenerationStats {
        prompt_tokens: 2,
        completion_tokens: 1,
        prefill_ms: 3,
        decode_ms: 4,
    };
    let mut success = Events::default();
    let mut terminal = Terminal::new(&mut success);
    assert!(terminal.delta("x".into()));
    terminal.finish(stats);
    terminal.finish(stats);
    assert_eq!(
        success.values,
        [Event::TextDelta("x".into()), Event::Finished(stats)]
    );

    let mut failure = Events::default();
    let mut terminal = Terminal::new(&mut failure);
    terminal.fail();
    terminal.finish(stats);
    assert_eq!(failure.values, [Event::Error("generation failed".into())]);

    let mut cancelled = Events {
        cancelled: true,
        values: Vec::new(),
    };
    let mut terminal = Terminal::new(&mut cancelled);
    terminal.fail();
    terminal.finish(stats);
    assert!(cancelled.values.is_empty());
}

#[test]
fn api_cancellation_discards_incomplete_utf8() {
    let mut decoder = TextDecoder::default();
    assert_eq!(decoder.push(&[0xe2, 0x82]), None);
    drop(decoder);
}

#[test]
fn reasoning_output_tags_remain_text_deltas() {
    let mut events = Events::default();
    let mut terminal = Terminal::new(&mut events);
    let mut decoder = TextDecoder::default();
    for bytes in [
        b"[TH".as_slice(),
        b"INK]x[/".as_slice(),
        b"THINK]".as_slice(),
    ] {
        if let Some(text) = decoder.push(bytes) {
            assert!(terminal.delta(text));
        }
    }
    decoder.finish();
    terminal.finish(GenerationStats {
        prompt_tokens: 1,
        completion_tokens: 3,
        prefill_ms: 0,
        decode_ms: 0,
    });

    let text = events
        .values
        .iter()
        .filter_map(|event| match event {
            Event::TextDelta(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(text, "[THINK]x[/THINK]");
    assert!(!text.contains('\u{fffd}'));
    assert_eq!(
        events
            .values
            .iter()
            .filter(|event| matches!(event, Event::Finished(_)))
            .count(),
        1
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Boundary {
    BeforePrefill,
    BeforeDecode,
}

trait Cancel {
    fn cancelled(&self, boundary: Boundary) -> bool;
}

struct NeverCancel;

impl Cancel for NeverCancel {
    fn cancelled(&self, _: Boundary) -> bool {
        false
    }
}

struct Tracker(Rc<Cell<usize>>);

impl Tracker {
    fn new() -> Self {
        Self(Rc::new(Cell::new(0)))
    }

    fn live(&self) -> usize {
        self.0.get()
    }
}

struct OwnedKv<'a, B: Backend> {
    backend: &'a B,
    kv: Option<Kv<B::Buffer>>,
    live: Rc<Cell<usize>>,
}

impl<'a, B: Backend> OwnedKv<'a, B> {
    fn new(backend: &'a B, cfg: &MistralConfig, req: &Request, tracker: &Tracker) -> Result<Self> {
        let kv = kv_cache::alloc_shape(
            backend,
            cfg.block_count,
            req.context,
            cfg.kv_head_count,
            cfg.key_length,
            cfg.value_length,
            req.kv_quant,
        )?;
        tracker.0.set(tracker.0.get() + 1);
        Ok(Self {
            backend,
            kv: Some(kv),
            live: Rc::clone(&tracker.0),
        })
    }
}

impl<B: Backend> Drop for OwnedKv<'_, B> {
    fn drop(&mut self) {
        if let Some(kv) = self.kv.take() {
            kv_cache::free(self.backend, kv);
            self.live.set(self.live.get() - 1);
        }
    }
}

fn generate<B: Backend, C: Cancel>(
    model: &MistralModel<B>,
    req: &Request,
    cancel: &C,
    tracker: &Tracker,
    fail_at: Option<Boundary>,
) -> Result<Vec<u32>> {
    generate_with_prefill(model, req, cancel, tracker, fail_at, true)
}

fn generate_with_prefill<B: Backend, C: Cancel>(
    model: &MistralModel<B>,
    req: &Request,
    cancel: &C,
    tracker: &Tracker,
    fail_at: Option<Boundary>,
    batched: bool,
) -> Result<Vec<u32>> {
    let required = req
        .prompt
        .len()
        .checked_add(req.max_tokens.saturating_sub(1));
    if req.prompt.is_empty()
        || req.prompt.len() > req.context
        || required.is_none_or(|n| n > req.context)
    {
        bail!("E11 invalid generation request");
    }
    let owned = OwnedKv::new(&model.backend, &model.config, req, tracker)?;
    if batched {
        prefill::prefill_with(
            &model.backend,
            &model.config,
            owned.kv.as_ref().unwrap(),
            &req.prompt,
            || boundary(cancel, fail_at, Boundary::BeforePrefill),
        )?;
    } else {
        boundary(cancel, fail_at, Boundary::BeforePrefill)?;
        for (pos, &token) in req.prompt.iter().enumerate() {
            forward::token(
                &model.backend,
                &model.config,
                owned.kv.as_ref().unwrap(),
                token,
                pos,
            )?;
        }
    }
    let mut tokens = Vec::new();
    for step in 0..req.max_tokens {
        boundary(cancel, fail_at, Boundary::BeforeDecode)?;
        let next = model
            .backend
            .read_argmax(&model.backend.buffers().logits, model.config.vocab_size)?;
        tokens.push(next);
        let pos = req.prompt.len() + step;
        if step + 1 < req.max_tokens && pos + 1 < req.context {
            forward::token(
                &model.backend,
                &model.config,
                owned.kv.as_ref().unwrap(),
                next,
                pos,
            )?;
        }
    }
    Ok(tokens)
}

fn boundary<C: Cancel>(cancel: &C, fail: Option<Boundary>, at: Boundary) -> Result<()> {
    if cancel.cancelled(at) || fail == Some(at) {
        bail!("generation stopped");
    }
    Ok(())
}

struct CancelAt(Boundary);

impl Cancel for CancelAt {
    fn cancelled(&self, boundary: Boundary) -> bool {
        self.0 == boundary
    }
}

fn model() -> MistralModel<ShapeBackend> {
    let config = config(1, 4096, 512);
    model_with(config.clone(), ShapeBackend::new(&config, false))
}

fn model_with(config: MistralConfig, backend: ShapeBackend) -> MistralModel<ShapeBackend> {
    MistralModel { backend, config }
}

fn request(max_tokens: usize) -> Request {
    Request {
        prompt: vec![1, 2],
        max_tokens,
        context: 8,
        kv_quant: KvQuant::F16,
    }
}

#[test]
fn prompt_overflow_fails_before_allocation() {
    let model = model();
    let tracker = Tracker::new();
    let mut req = request(1);
    req.context = 1;
    assert!(generate(&model, &req, &NeverCancel, &tracker, None).is_err());
    assert_eq!(tracker.live(), 0);
}

#[test]
fn zero_tokens_prefills_and_releases_kv() {
    let model = model();
    let tracker = Tracker::new();
    assert!(generate(&model, &request(0), &NeverCancel, &tracker, None).is_ok());
    assert_eq!(tracker.live(), 0);
}

#[test]
fn error_matrix_e18_e19_failures_and_cancellation_release_kv() {
    for fail_at in [Some(Boundary::BeforePrefill), Some(Boundary::BeforeDecode)] {
        let model = model();
        let tracker = Tracker::new();
        assert!(generate(&model, &request(1), &NeverCancel, &tracker, fail_at).is_err());
        assert_eq!(tracker.live(), 0);
    }
    let model = model();
    let tracker = Tracker::new();
    assert!(
        generate(
            &model,
            &request(1),
            &CancelAt(Boundary::BeforeDecode),
            &tracker,
            None
        )
        .is_err()
    );
    assert_eq!(tracker.live(), 0);
}

#[test]
fn allocation_and_numeric_failures_release_every_buffer() {
    for allocation in [0, 1, 4] {
        let config = config(1, 4096, 512);
        let backend = ShapeBackend::new(&config, false);
        backend.fail_allocation(allocation);
        let model = model_with(config, backend);
        assert!(generate(&model, &request(0), &NeverCancel, &Tracker::new(), None).is_err());
        assert_eq!(model.backend.live_allocations(), 0);
    }

    let model = model();
    model.backend.fail_embed();
    assert!(generate(&model, &request(0), &NeverCancel, &Tracker::new(), None).is_err());
    assert_eq!(model.backend.live_allocations(), 0);
}

struct CancelCall {
    boundary: Boundary,
    call: usize,
    seen: Cell<usize>,
}

impl Cancel for CancelCall {
    fn cancelled(&self, boundary: Boundary) -> bool {
        if boundary != self.boundary {
            return false;
        }
        let seen = self.seen.get();
        self.seen.set(seen + 1);
        seen == self.call
    }
}

#[test]
fn cancellation_is_checked_before_each_batch_and_decode_step() {
    for (boundary, call, mut req) in [
        (Boundary::BeforePrefill, 1, request(0)),
        (Boundary::BeforeDecode, 1, request(3)),
    ] {
        if boundary == Boundary::BeforePrefill {
            req.prompt = vec![1; prefill::BATCH_ROWS + 1];
            req.context = req.prompt.len();
        }
        let model = model();
        let cancel = CancelCall {
            boundary,
            call,
            seen: Cell::new(0),
        };
        assert!(generate(&model, &req, &cancel, &Tracker::new(), None).is_err());
        assert_eq!(cancel.seen.get(), 2);
        assert_eq!(model.backend.live_allocations(), 0);
    }
}

#[cfg(any(feature = "cpu", feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) mod numeric {
    use crate::backend::Backend;
    use crate::backend::buffers::{Buffers, LayerWeights, Scratch, WeightSet};
    use crate::backend::cpu::{CpuBackend, CpuBuffer, CpuFormat};
    use crate::family::mistral::MistralConfig;
    use crate::family::mistral::graph::{forward, prefill};
    use crate::kv_cache;
    use crate::kv_cache::scheme::KvQuant;

    fn cfg() -> MistralConfig {
        MistralConfig {
            block_count: 1,
            context_length: 16,
            embedding_length: 8,
            feed_forward_length: 8,
            head_count: 2,
            kv_head_count: 1,
            key_length: 4,
            value_length: 2,
            q_width: 8,
            k_width: 4,
            v_width: 2,
            attention_width: 4,
            rope_dimension: 4,
            rope_freq_base: 10_000.0,
            rms_epsilon: 0.00001,
            yarn_factor: 2.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_log_multiplier: 0.1,
            yarn_original_context: 8,
            attention_temperature_scale: 1.1,
            vocab_size: 8,
            bos_id: 1,
            eos_id: 2,
        }
    }

    fn f16(values: &[f32]) -> CpuBuffer {
        let buffer = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buffer.write_f16_from_f32(values);
        buffer
    }

    fn matrix(input: usize, output: usize, seed: usize) -> CpuBuffer {
        f16(&(0..input * output)
            .map(|i| ((i + seed) % 11) as f32 * 0.025 - 0.12)
            .collect::<Vec<_>>())
    }

    #[derive(Clone, Copy)]
    enum Profile {
        Dense,
        Mixed,
    }

    fn weight(input: usize, output: usize, seed: usize, profile: Profile) -> CpuBuffer {
        match profile {
            Profile::Dense => matrix(input, output, seed),
            Profile::Mixed if seed.is_multiple_of(2) => {
                quant(input, output, seed, CpuFormat::Q6_K, 210)
            }
            Profile::Mixed => quant(input, output, seed, CpuFormat::Q4_K, 144),
        }
    }

    fn quant(
        input: usize,
        output: usize,
        seed: usize,
        format: CpuFormat,
        block_bytes: usize,
    ) -> CpuBuffer {
        // All retained K-quants use one 256-value super-block.
        let block = 256;
        let mut bytes = vec![0u8; output * (input / block) * block_bytes];
        for chunk in bytes.chunks_exact_mut(block_bytes) {
            match format {
                CpuFormat::Q4_K => {
                    chunk[0..2].copy_from_slice(&f16_bits(0.02));
                    chunk[2..4].copy_from_slice(&f16_bits(0.005));
                    chunk[4..12].fill(1);
                    for (i, byte) in chunk[16..].iter_mut().enumerate() {
                        *byte = ((i + seed) as u8 & 0x0f) | ((((i + seed + 3) as u8) & 0x0f) << 4);
                    }
                }
                CpuFormat::Q6_K => {
                    for (i, byte) in chunk[..192].iter_mut().enumerate() {
                        *byte = (i + seed) as u8;
                    }
                    chunk[192..208].fill(1);
                    chunk[208..210].copy_from_slice(&f16_bits(0.01));
                }
                _ => unreachable!(),
            }
        }
        CpuBuffer::from_bytes(bytes, format)
    }

    fn f16_bits(value: f32) -> [u8; 2] {
        crate::backend::cpu::buffer::f32_to_f16(value).to_le_bytes()
    }

    fn backend(cfg: &MistralConfig, dedicated: bool, profile: Profile) -> CpuBackend {
        let norm = || f16(&vec![1.0; cfg.embedding_length]);
        let layer = LayerWeights {
            attn_norm: norm(),
            attn_q: weight(cfg.embedding_length, cfg.q_width, 1, profile),
            attn_k: weight(cfg.embedding_length, cfg.k_width, 2, profile),
            attn_v: weight(cfg.embedding_length, cfg.v_width, 3, profile),
            attn_output: weight(cfg.attention_width, cfg.embedding_length, 4, profile),
            ffn_norm: norm(),
            ffn_gate: weight(cfg.embedding_length, cfg.feed_forward_length, 5, profile),
            ffn_up: weight(cfg.embedding_length, cfg.feed_forward_length, 6, profile),
            ffn_down: weight(cfg.feed_forward_length, cfg.embedding_length, 7, profile),
        };
        let token_embd = weight(cfg.embedding_length, cfg.vocab_size, 8, profile);
        let output = dedicated.then(|| weight(cfg.embedding_length, cfg.vocab_size, 9, profile));
        let z16 = |n| CpuBuffer::zeroed(n * 2, CpuFormat::F16);
        CpuBackend::from_buffers(Buffers {
            weights: WeightSet {
                token_embd: Some(token_embd),
                output_norm: Some(norm()),
                output,
                layers: vec![layer],
            },
            scratch: Scratch {
                x: CpuBuffer::zeroed(cfg.embedding_length * 4, CpuFormat::F32),
                normed: z16(cfg.embedding_length),
                q: z16(cfg.q_width),
                k: z16(cfg.k_width),
                v: z16(cfg.v_width),
                attn: z16(cfg.attention_width),
                proj: z16(cfg.embedding_length),
                gate: z16(cfg.feed_forward_length),
                up: z16(cfg.feed_forward_length),
                act: z16(cfg.feed_forward_length),
                ffn_out: z16(cfg.embedding_length),
            },
            logits: CpuBuffer::zeroed(cfg.vocab_size * 4, CpuFormat::F32),
        })
    }

    fn logits(
        cfg: &MistralConfig,
        scheme: KvQuant,
        prompt: &[u32],
        batched: bool,
        dedicated: bool,
        profile: Profile,
    ) -> Vec<f32> {
        let backend = backend(cfg, dedicated, profile);
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
        if batched {
            prefill::prefill_with(&backend, cfg, &kv, prompt, || Ok(())).unwrap();
        } else {
            for (pos, &token) in prompt.iter().enumerate() {
                forward::token(&backend, cfg, &kv, token, pos).unwrap();
            }
        }
        let result = backend
            .read_logits(&backend.buffers().logits, cfg.vocab_size)
            .unwrap();
        kv_cache::free(&backend, kv);
        result
    }

    #[test]
    pub(crate) fn batched_final_logits_match_sequential_for_both_kv_schemes() {
        let cfg = cfg();
        for (scheme, dedicated) in [
            (KvQuant::F16, false),
            (KvQuant::F16, true),
            (KvQuant::Int8, false),
            (KvQuant::Int8, true),
        ] {
            for prompt in [&[3u32][..], &[1, 3, 4, 5, 6][..]] {
                let sequential = logits(&cfg, scheme, prompt, false, dedicated, Profile::Dense);
                let batched = logits(&cfg, scheme, prompt, true, dedicated, Profile::Dense);
                let max = sequential
                    .iter()
                    .zip(&batched)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "scheme={} rows={} max_error={max}",
                    scheme.name(),
                    prompt.len()
                );
                assert!(max <= 0.05, "prefill error {max}");
            }
        }
    }

    #[test]
    pub(crate) fn quantized_profiles_drive_decode_and_short_prefill() {
        let mut cfg = cfg();
        cfg.embedding_length = 256;
        cfg.feed_forward_length = 256;
        cfg.head_count = 2;
        cfg.kv_head_count = 1;
        cfg.key_length = 128;
        cfg.value_length = 128;
        cfg.q_width = 256;
        cfg.k_width = 128;
        cfg.v_width = 128;
        cfg.attention_width = 256;
        cfg.rope_dimension = 128;
        cfg.vocab_size = 32;
        for scheme in [KvQuant::F16, KvQuant::Int8] {
            for dedicated in [false, true] {
                let prompt = [1, 3, 4, 5, 6];
                let sequential = logits(&cfg, scheme, &prompt, false, dedicated, Profile::Mixed);
                let batched = logits(&cfg, scheme, &prompt, true, dedicated, Profile::Mixed);
                let max = sequential
                    .iter()
                    .zip(&batched)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(sequential.iter().all(|v| v.is_finite()));
                assert!(batched.iter().all(|v| v.is_finite()));
                assert!(max <= 0.05, "quantized prefill error {max}");
            }
        }
    }
}
