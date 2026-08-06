/*
 * graph_horizon_engine — Metal Backend trait implementation
 * Contains one thin `impl Backend` delegating ownership, command lifecycle,
 * readback, and operation dispatch. Features control availability; immutable
 * effective placement reaches only Metal's two approved operation exceptions.
 */

use color_eyre::eyre::Result;

use super::exec::readback;
use super::kernels;
use super::{MetalBackend, MetalBuffer, MetalEncoder, MetalFormat};
use crate::backend::Backend;
use crate::backend::buffers::Buffers;

// AGENTS deroga I: singolo `impl Backend for MetalBackend` di delegatori sottili.
impl Backend for MetalBackend {
    type Buffer = MetalBuffer;
    type Encoder = MetalEncoder;

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
        kernels::argmax::read(&self.device, &self.pipelines, logits, &self.reduce, vocab)
    }

    fn read_topk(&self, logits: &MetalBuffer, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        kernels::topk::read(
            &self.device,
            &self.pipelines,
            logits,
            &self.reduce,
            vocab,
            k,
        )
    }

    fn kv_write(
        &self,
        encoder: &MetalEncoder,
        kv: &crate::kv_cache::Kv<MetalBuffer>,
        k: &MetalBuffer,
        v: &MetalBuffer,
        k_payload_offset: u64,
        v_payload_offset: u64,
        k_meta_offset: u64,
        v_meta_offset: u64,
        vectors: u32,
    ) -> Result<()> {
        kernels::kv_write::encode(
            encoder,
            &self.pipelines,
            kv,
            k,
            v,
            k_payload_offset,
            v_payload_offset,
            k_meta_offset,
            v_meta_offset,
            vectors,
        )
    }

    fn embed(
        &self,
        encoder: &MetalEncoder,
        x: &MetalBuffer,
        weight: &MetalBuffer,
        token: u32,
        embd: u32,
    ) -> Result<()> {
        kernels::embedding::encode(encoder, &self.pipelines, x, weight, token, embd)
    }

    fn matmul(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        in_dim: u32,
        out_dim: u32,
    ) {
        let _ = kernels::matmul::encode(
            encoder,
            &self.pipelines,
            out,
            input,
            weight,
            in_dim,
            out_dim,
            false,
        );
    }

    fn matmul_batched(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        in_dim: u32,
        out_dim: u32,
        rows: u32,
    ) {
        let _ = kernels::matmul::encode_batched(
            encoder,
            &self.pipelines,
            out,
            input,
            weight,
            in_dim,
            out_dim,
            rows,
            self.mixed_placement,
        );
    }

    fn logits(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        in_dim: u32,
        out_dim: u32,
    ) {
        let _ = kernels::matmul::encode(
            encoder,
            &self.pipelines,
            out,
            input,
            weight,
            in_dim,
            out_dim,
            true,
        );
    }

    fn rmsnorm_x(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        dim: u32,
        eps: f32,
        rows: u32,
    ) {
        let _ = kernels::normalization::encode(
            encoder,
            &self.pipelines,
            out,
            input,
            weight,
            dim,
            eps,
            rows,
        );
    }

    fn rope_yarn(
        &self,
        encoder: &MetalEncoder,
        input: &MetalBuffer,
        heads: u32,
        head_dim: u32,
        position: u32,
        yarn: &crate::backend::rope::Yarn,
        role: crate::backend::rope::RopeRole,
    ) -> Result<()> {
        kernels::rope::encode(
            encoder,
            &self.pipelines,
            input,
            heads,
            head_dim,
            position,
            yarn,
            role,
        )
    }

    fn silu_mul(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        n: u32,
    ) {
        let _ = kernels::silu_mul::encode(encoder, &self.pipelines, out, gate, up, n);
    }

    fn residual_add(&self, encoder: &MetalEncoder, x: &MetalBuffer, y: &MetalBuffer, n: u32) {
        let _ = kernels::residual_add::encode(encoder, &self.pipelines, x, y, n);
    }

    fn attention_decode(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        q: &MetalBuffer,
        kv: &crate::kv_cache::Kv<MetalBuffer>,
        q_heads: u32,
        position: u32,
        layer: u32,
    ) {
        let _ = kernels::attention::encode(
            encoder,
            &self.pipelines,
            out,
            q,
            kv,
            q_heads,
            position,
            1,
            layer,
            self.mixed_placement,
        );
    }

    fn attention_prefill(
        &self,
        encoder: &MetalEncoder,
        out: &MetalBuffer,
        q: &MetalBuffer,
        kv: &crate::kv_cache::Kv<MetalBuffer>,
        q_heads: u32,
        base: u32,
        n: u32,
        layer: u32,
    ) {
        let _ = kernels::attention::encode(
            encoder,
            &self.pipelines,
            out,
            q,
            kv,
            q_heads,
            base,
            n,
            layer,
            self.mixed_placement,
        );
    }
}
