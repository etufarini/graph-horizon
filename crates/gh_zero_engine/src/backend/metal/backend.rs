/*
 * gh_zero_engine — Metal Backend trait implementation
 * Contains one thin `impl Backend` delegating ownership, command lifecycle,
 * readback, and operation dispatch. Algorithms and loading remain outside.
 */

use color_eyre::eyre::{Result, bail};

use super::exec::{dispatch, readback};
use super::pipeline::Kernel;
use super::{MetalBackend, MetalBuffer, MetalEncoder, MetalFormat};
use crate::backend::Backend;
use crate::backend::buffers::Buffers;
use crate::backend::source::WeightSource;
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;

// AGENTS deroga I: singolo `impl Backend for MetalBackend` di delegatori sottili.
impl Backend for MetalBackend {
    type Buffer = MetalBuffer;
    type Encoder = MetalEncoder;

    fn load(
        _meta: &ModelMetadata,
        _source: &dyn WeightSource,
        _file: &GgufFile,
        _context: usize,
    ) -> Result<Self> {
        bail!("metal: model allocation failed")
    }

    fn buffers(&self) -> &Buffers<MetalBuffer> {
        &self.buffers
    }

    fn alloc_buffer(&self, bytes: u64) -> Result<MetalBuffer> {
        MetalBuffer::allocate(&self.device, bytes, MetalFormat::Raw)
    }

    fn free_buffer(&self, buffer: MetalBuffer) {
        drop(buffer);
    }

    fn view(&self, buffer: &MetalBuffer, offset: u64, len: u64) -> MetalBuffer {
        buffer
            .view(offset, len)
            .expect("validated Metal buffer view")
    }

    fn min_buffer_offset_alignment(&self) -> u64 {
        4
    }

    fn begin(&self) -> Result<MetalEncoder> {
        MetalEncoder::begin(&self.device)
    }

    fn submit(&self, encoder: MetalEncoder) -> Result<()> {
        encoder.submit()
    }

    fn read_logits(&self, logits: &MetalBuffer, vocab: usize) -> Result<Vec<f32>> {
        readback::logits(logits, vocab)
    }

    fn read_argmax(&self, logits: &MetalBuffer, vocab: usize) -> Result<u32> {
        readback::argmax(logits, vocab)
    }

    fn read_topk(&self, logits: &MetalBuffer, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        readback::topk(logits, vocab, k)
    }

    fn kv_write(
        &self,
        encoder: &MetalEncoder,
        kv: &crate::kv_cache::Kv<MetalBuffer>,
        k: &MetalBuffer,
        v: &MetalBuffer,
        _k_payload_offset: u64,
        _v_payload_offset: u64,
        _k_meta_offset: u64,
        _v_meta_offset: u64,
        _vectors: u32,
    ) -> Result<()> {
        dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::KvWrite,
            &[k, v, &kv.k, &kv.v],
            &[],
            [1, 1, 1],
        )
    }

    fn embed(
        &self,
        encoder: &MetalEncoder,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        _token: u32,
        _embd: u32,
    ) -> Result<()> {
        dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::Embedding,
            &[weight, x],
            &[],
            [1, 1, 1],
        )
    }

    fn matmul(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        _in_dim: u32,
        _out_dim: u32,
    ) {
        let _ = dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::Matmul,
            &[input, weight, out],
            &[],
            [1, 1, 1],
        );
    }

    fn logits(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        _in_dim: u32,
        _out_dim: u32,
    ) {
        let _ = dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::Matmul,
            &[input, weight, out],
            &[],
            [1, 1, 1],
        );
    }

    fn rmsnorm_x(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        _dim: u32,
        _eps: f32,
        _rows: u32,
    ) {
        let _ = dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::Rmsnorm,
            &[input, weight, out],
            &[],
            [1, 1, 1],
        );
    }

    fn rope_yarn(
        &self,
        encoder: &MetalEncoder,
        input: &MetalBuffer,
        _heads: u32,
        _head_dim: u32,
        _position: u32,
        _yarn: &crate::backend::rope::Yarn,
        _role: crate::backend::rope::RopeRole,
    ) -> Result<()> {
        dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::Rope,
            &[input],
            &[],
            [1, 1, 1],
        )
    }

    fn silu_mul(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        _n: u32,
    ) {
        let _ = dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::SiluMul,
            &[gate, up, out],
            &[],
            [1, 1, 1],
        );
    }

    fn residual_add(&self, encoder: &MetalEncoder, x: &MetalBuffer, y: &MetalBuffer, _n: u32) {
        let _ = dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::ResidualAdd,
            &[x, y],
            &[],
            [1, 1, 1],
        );
    }

    fn attention_decode(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        q: &MetalBuffer,
        kv: &crate::kv_cache::Kv<MetalBuffer>,
        _q_heads: u32,
        _position: u32,
        _layer: u32,
    ) {
        let _ = dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::Attention,
            &[q, &kv.k, &kv.v, out],
            &[],
            [1, 1, 1],
        );
    }

    fn attention_prefill(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        q: &MetalBuffer,
        kv: &crate::kv_cache::Kv<MetalBuffer>,
        _q_heads: u32,
        _base: u32,
        _n: u32,
        _layer: u32,
    ) {
        let _ = dispatch::encode(
            encoder,
            &self.pipelines,
            Kernel::Attention,
            &[q, &kv.k, &kv.v, out],
            &[],
            [1, 1, 1],
        );
    }
}
