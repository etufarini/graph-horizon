/*
 * graph_horizon_engine — Ministral graph shape recorder tests
 * Test-only backend that records the shared graph without allocating tensor
 * payloads. It validates exact extents and operation order for the approved
 * 3B/8B/14B rows; it is not compiled into production builds.
 */

use std::cell::{Cell, RefCell};

use color_eyre::eyre::{Result, bail};

use crate::backend::Backend;
use crate::backend::buffers::{Buffers, LayerWeights, Scratch, WeightSet};
use crate::backend::rope::{RopeRole, Yarn};
#[cfg(any(feature = "cpu", feature = "vulkan"))]
use crate::backend::source::WeightSource;
use crate::family::mistral::MistralConfig;
use crate::family::mistral::graph::forward;
use crate::family::mistral::version::{
    ATTENTION_WIDTH, HEAD_COUNT, K_WIDTH, KEY_LENGTH, KV_HEAD_COUNT, MAX_CONTEXT, Q_WIDTH,
    REFERENCE_ROWS, ROPE_DIMENSION, V_WIDTH, VALUE_LENGTH,
};
#[cfg(any(feature = "cpu", feature = "vulkan"))]
use crate::gguf::loader::GgufFile;
#[cfg(any(feature = "cpu", feature = "vulkan"))]
use crate::gguf::metadata::ModelMetadata;
use crate::kv_cache::scheme::KvQuant;
use crate::kv_cache::{self, Kv};

#[derive(Clone)]
pub(crate) struct ShapeBuffer {
    bytes: u64,
    matrix: Option<(u32, u32)>,
    name: String,
}

impl ShapeBuffer {
    fn plain(name: impl Into<String>, bytes: u64) -> Self {
        Self {
            bytes,
            matrix: None,
            name: name.into(),
        }
    }

    fn matrix(name: impl Into<String>, input: usize, output: usize) -> Self {
        Self {
            bytes: 0,
            matrix: Some((input as u32, output as u32)),
            name: name.into(),
        }
    }
}

pub(crate) struct ShapeBackend {
    buffers: Buffers<ShapeBuffer>,
    trace: RefCell<Vec<String>>,
    live: Cell<usize>,
    allocations: Cell<usize>,
    fail_allocation: Cell<Option<usize>>,
    fail_embed: Cell<bool>,
}

pub(crate) struct ShapeEncoder;

impl ShapeBackend {
    pub(crate) fn new(cfg: &MistralConfig, dedicated: bool) -> Self {
        let f16 = |name, n| ShapeBuffer::plain(name, n as u64 * 2);
        let matrix = |name, input, output| ShapeBuffer::matrix(name, input, output);
        let layers = (0..cfg.block_count)
            .map(|i| LayerWeights {
                attn_norm: f16(format!("l{i}.attn_norm"), cfg.embedding_length),
                attn_q: matrix(format!("l{i}.q"), cfg.embedding_length, cfg.q_width),
                attn_k: matrix(format!("l{i}.k"), cfg.embedding_length, cfg.k_width),
                attn_v: matrix(format!("l{i}.v"), cfg.embedding_length, cfg.v_width),
                attn_output: matrix(
                    format!("l{i}.attn_out"),
                    cfg.attention_width,
                    cfg.embedding_length,
                ),
                ffn_norm: f16(format!("l{i}.ffn_norm"), cfg.embedding_length),
                ffn_gate: matrix(
                    format!("l{i}.gate"),
                    cfg.embedding_length,
                    cfg.feed_forward_length,
                ),
                ffn_up: matrix(
                    format!("l{i}.up"),
                    cfg.embedding_length,
                    cfg.feed_forward_length,
                ),
                ffn_down: matrix(
                    format!("l{i}.down"),
                    cfg.feed_forward_length,
                    cfg.embedding_length,
                ),
            })
            .collect();
        let token_embd = matrix(
            "token_embd".to_string(),
            cfg.embedding_length,
            cfg.vocab_size,
        );
        let output =
            dedicated.then(|| matrix("output".to_string(), cfg.embedding_length, cfg.vocab_size));
        let scratch = Scratch {
            x: ShapeBuffer::plain("x", cfg.embedding_length as u64 * 4),
            normed: f16("normed".to_string(), cfg.embedding_length),
            q: f16("q".to_string(), cfg.q_width),
            k: f16("k".to_string(), cfg.k_width),
            v: f16("v".to_string(), cfg.v_width),
            attn: f16("attn".to_string(), cfg.attention_width),
            proj: f16("proj".to_string(), cfg.embedding_length),
            gate: f16("gate".to_string(), cfg.feed_forward_length),
            up: f16("up".to_string(), cfg.feed_forward_length),
            act: f16("act".to_string(), cfg.feed_forward_length),
            ffn_out: f16("ffn_out".to_string(), cfg.embedding_length),
        };
        Self {
            buffers: Buffers {
                weights: WeightSet {
                    token_embd: Some(token_embd),
                    output_norm: Some(f16("output_norm".to_string(), cfg.embedding_length)),
                    output,
                    layers,
                },
                scratch,
                logits: ShapeBuffer::plain("logits", cfg.vocab_size as u64 * 4),
            },
            trace: RefCell::new(Vec::new()),
            live: Cell::new(0),
            allocations: Cell::new(0),
            fail_allocation: Cell::new(None),
            fail_embed: Cell::new(false),
        }
    }

    fn push(&self, operation: impl Into<String>) {
        self.trace.borrow_mut().push(operation.into());
    }

    pub(crate) fn trace(&self) -> Vec<String> {
        self.trace.borrow().clone()
    }

    pub(crate) fn fail_allocation(&self, allocation: usize) {
        self.fail_allocation.set(Some(allocation));
    }

    pub(crate) fn fail_embed(&self) {
        self.fail_embed.set(true);
    }

    pub(crate) fn live_allocations(&self) -> usize {
        self.live.get()
    }
}

impl Backend for ShapeBackend {
    type Buffer = ShapeBuffer;
    type Encoder = ShapeEncoder;

    #[cfg(any(feature = "cpu", feature = "vulkan"))]
    fn load(
        _meta: &ModelMetadata,
        _ws: &dyn WeightSource,
        _gguf: &GgufFile,
        _context: usize,
    ) -> Result<Self> {
        bail!("shape backend is constructed from validated dimensions")
    }

    fn buffers(&self) -> &Buffers<ShapeBuffer> {
        &self.buffers
    }

    fn alloc_buffer(&self, bytes: u64) -> Result<ShapeBuffer> {
        let allocation = self.allocations.get();
        self.allocations.set(allocation + 1);
        if self.fail_allocation.get() == Some(allocation) {
            bail!("shape backend: injected allocation failure");
        }
        self.live.set(self.live.get() + 1);
        Ok(ShapeBuffer::plain("kv", bytes))
    }

    fn free_buffer(&self, _buf: ShapeBuffer) {
        self.live.set(self.live.get() - 1);
    }

    fn view(&self, buf: &ShapeBuffer, offset: u64, len: u64) -> ShapeBuffer {
        assert!(offset.checked_add(len).is_some_and(|end| end <= buf.bytes));
        ShapeBuffer::plain(format!("{}[{offset}]", buf.name), len)
    }

    fn min_buffer_offset_alignment(&self) -> u64 {
        1
    }

    fn kv_write(
        &self,
        _enc: &ShapeEncoder,
        kv: &Kv<ShapeBuffer>,
        k: &ShapeBuffer,
        v: &ShapeBuffer,
        _k_payload_offset: u64,
        _v_payload_offset: u64,
        _k_meta_offset: u64,
        _v_meta_offset: u64,
        vectors: u32,
    ) -> Result<()> {
        assert_eq!(k.bytes, vectors as u64 * kv.head_dim as u64 * 2);
        assert_eq!(v.bytes, vectors as u64 * kv.value_dim as u64 * 2);
        self.push("kv_write");
        Ok(())
    }

    fn begin(&self) -> Result<ShapeEncoder> {
        self.push("begin");
        Ok(ShapeEncoder)
    }

    fn submit(&self, _enc: ShapeEncoder) -> Result<()> {
        self.push("submit");
        Ok(())
    }

    fn embed(
        &self,
        _enc: &ShapeEncoder,
        x: &ShapeBuffer,
        weight: &ShapeBuffer,
        _token: u32,
        embd: u32,
    ) -> Result<()> {
        if self.fail_embed.get() {
            bail!("shape backend: injected numeric failure");
        }
        assert_eq!(x.bytes, embd as u64 * 4);
        assert_eq!(weight.matrix.unwrap().0, embd);
        self.push("embed");
        Ok(())
    }

    fn matmul(
        &self,
        _enc: &ShapeEncoder,
        out: &ShapeBuffer,
        a: &ShapeBuffer,
        weight: &ShapeBuffer,
        input: u32,
        output: u32,
    ) {
        assert!(a.bytes >= input as u64 * 2);
        assert!(out.bytes >= output as u64 * 2);
        assert_eq!(weight.matrix, Some((input, output)), "{}", weight.name);
        self.push(weight.name.clone());
    }

    fn logits(
        &self,
        _enc: &ShapeEncoder,
        out: &ShapeBuffer,
        x: &ShapeBuffer,
        weight: &ShapeBuffer,
        input: u32,
        output: u32,
    ) {
        assert_eq!(x.bytes, input as u64 * 2);
        assert_eq!(out.bytes, output as u64 * 4);
        assert_eq!(weight.matrix, Some((input, output)));
        self.push(format!("logits:{}", weight.name));
    }

    fn rmsnorm_x(
        &self,
        _enc: &ShapeEncoder,
        out: &ShapeBuffer,
        x: &ShapeBuffer,
        weight: &ShapeBuffer,
        dim: u32,
        _eps: f32,
        rows: u32,
    ) {
        assert!(x.bytes >= dim as u64 * rows as u64 * 2);
        assert!(out.bytes >= dim as u64 * rows as u64 * 2);
        assert_eq!(weight.bytes, dim as u64 * 2);
        self.push(weight.name.clone());
    }

    fn rope_yarn(
        &self,
        _enc: &ShapeEncoder,
        x: &ShapeBuffer,
        heads: u32,
        head_dim: u32,
        _pos: u32,
        _yarn: &Yarn,
        role: RopeRole,
    ) -> Result<()> {
        assert_eq!(x.bytes, heads as u64 * head_dim as u64 * 2);
        self.push(match role {
            RopeRole::Query => "rope_q",
            RopeRole::Key => "rope_k",
        });
        Ok(())
    }

    fn silu_mul(
        &self,
        _enc: &ShapeEncoder,
        out: &ShapeBuffer,
        gate: &ShapeBuffer,
        up: &ShapeBuffer,
        n: u32,
    ) {
        assert_eq!(out.bytes, n as u64 * 2);
        assert_eq!(gate.bytes, out.bytes);
        assert_eq!(up.bytes, out.bytes);
        self.push("silu_mul");
    }

    fn residual_add(&self, _enc: &ShapeEncoder, x: &ShapeBuffer, y: &ShapeBuffer, n: u32) {
        assert_eq!(x.bytes, n as u64 * 4);
        assert_eq!(y.bytes, n as u64 * 2);
        self.push("residual");
    }

    fn attention_decode(
        &self,
        _enc: &ShapeEncoder,
        out: &ShapeBuffer,
        q: &ShapeBuffer,
        kv: &Kv<ShapeBuffer>,
        q_heads: u32,
        _pos: u32,
        _layer: u32,
    ) {
        assert_eq!(q.bytes, q_heads as u64 * kv.head_dim as u64 * 2);
        assert_eq!(out.bytes, q_heads as u64 * kv.value_dim as u64 * 2);
        self.push("attention");
    }

    fn attention_prefill(
        &self,
        _enc: &ShapeEncoder,
        _out: &ShapeBuffer,
        _q: &ShapeBuffer,
        _kv: &Kv<ShapeBuffer>,
        _q_heads: u32,
        _base: u32,
        _n: u32,
        _layer: u32,
    ) {
        self.push("attention_prefill");
    }

    fn read_logits(&self, _logits: &ShapeBuffer, vocab: usize) -> Result<Vec<f32>> {
        Ok(vec![0.0; vocab])
    }

    fn read_argmax(&self, _logits: &ShapeBuffer, _vocab: usize) -> Result<u32> {
        Ok(0)
    }

    fn read_topk(
        &self,
        _logits: &ShapeBuffer,
        _vocab: usize,
        _k: usize,
    ) -> Result<Vec<(u32, f32)>> {
        Ok(Vec::new())
    }
}

pub(crate) fn config(blocks: usize, hidden: usize, ffn: usize) -> MistralConfig {
    MistralConfig {
        block_count: blocks,
        context_length: MAX_CONTEXT,
        embedding_length: hidden,
        feed_forward_length: ffn,
        head_count: HEAD_COUNT,
        kv_head_count: KV_HEAD_COUNT,
        key_length: KEY_LENGTH,
        value_length: VALUE_LENGTH,
        q_width: Q_WIDTH,
        k_width: K_WIDTH,
        v_width: V_WIDTH,
        attention_width: ATTENTION_WIDTH,
        rope_dimension: ROPE_DIMENSION,
        rope_freq_base: 1_000_000.0,
        rms_epsilon: 0.00001,
        yarn_factor: 8.0,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        yarn_log_multiplier: 0.1,
        yarn_original_context: 32_768,
        attention_temperature_scale: 1.0,
        vocab_size: 32,
        bos_id: 1,
        eos_id: 2,
    }
}

#[test]
fn versioned_rows_follow_one_shared_operation_order() {
    for row in REFERENCE_ROWS {
        let (blocks, hidden, ffn, dedicated) = row;
        let cfg = config(blocks, hidden, ffn);
        let backend = ShapeBackend::new(&cfg, dedicated);
        let kv = kv_cache::alloc_shape(
            &backend,
            cfg.block_count,
            cfg.context_length,
            cfg.kv_head_count,
            cfg.key_length,
            cfg.value_length,
            KvQuant::F16,
        )
        .unwrap();
        forward::token(&backend, &cfg, &kv, 3, 0).unwrap();
        kv_cache::free(&backend, kv);

        let trace = backend.trace();
        assert_eq!(trace.first().map(String::as_str), Some("begin"));
        assert_eq!(trace.get(1).map(String::as_str), Some("embed"));
        for layer in 0..cfg.block_count {
            let start = 2 + layer * 16;
            let expected = [
                format!("l{layer}.attn_norm"),
                format!("l{layer}.q"),
                format!("l{layer}.k"),
                format!("l{layer}.v"),
                "rope_q".into(),
                "rope_k".into(),
                "kv_write".into(),
                "attention".into(),
                format!("l{layer}.attn_out"),
                "residual".into(),
                format!("l{layer}.ffn_norm"),
                format!("l{layer}.gate"),
                format!("l{layer}.up"),
                "silu_mul".into(),
                format!("l{layer}.down"),
                "residual".into(),
            ];
            assert_eq!(&trace[start..start + 16], &expected);
        }
        let tail = 2 + cfg.block_count * 16;
        assert_eq!(trace[tail], "output_norm");
        assert_eq!(
            trace[tail + 1],
            if dedicated {
                "logits:output"
            } else {
                "logits:token_embd"
            }
        );
        assert_eq!(trace[tail + 2], "submit");
    }
}

#[test]
#[should_panic]
fn hidden_width_q_buffer_is_rejected_before_recording_projection() {
    let (blocks, hidden, ffn, _) = REFERENCE_ROWS[0];
    let cfg = config(blocks, hidden, ffn);
    let mut backend = ShapeBackend::new(&cfg, false);
    backend.buffers.scratch.q.bytes = cfg.embedding_length as u64 * 2;
    let kv = kv_cache::alloc_shape(
        &backend,
        cfg.block_count,
        cfg.context_length,
        cfg.kv_head_count,
        cfg.key_length,
        cfg.value_length,
        KvQuant::F16,
    )
    .unwrap();
    let _ = forward::token(&backend, &cfg, &kv, 3, 0);
}
