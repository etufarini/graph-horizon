/*
 * graph_horizon_engine — thin CUDA Backend trait delegation.
 * Ownership, validation, launches, and numeric work remain in their focused
 * modules; void operations only add the mandated first-error latch.
 */

use color_eyre::eyre::Result;

use super::exec::readback;
use super::kernels;
use super::{CudaBackend, CudaBuffer, CudaEncoder, CudaFormat};
use crate::backend::Backend;
use crate::backend::buffers::Buffers;

// AGENTS deroga I: singolo `impl Backend for CudaBackend` di delegatori sottili.
impl Backend for CudaBackend {
    type Buffer = CudaBuffer;
    type Encoder = CudaEncoder;

    fn buffers(&self) -> &Buffers<CudaBuffer> {
        &self.buffers
    }

    fn alloc_buffer(&self, bytes: u64) -> Result<CudaBuffer> {
        CudaBuffer::allocate(&self.device, bytes, CudaFormat::Raw)
    }

    fn free_buffer(&self, buffer: CudaBuffer) {
        drop(buffer);
    }

    fn view(&self, buffer: &CudaBuffer, offset: u64, len: u64) -> CudaBuffer {
        buffer
            .view(offset, len)
            .expect("validated CUDA buffer view")
    }

    fn min_buffer_offset_alignment(&self) -> u64 {
        2
    }

    fn begin(&self) -> Result<CudaEncoder> {
        Ok(CudaEncoder::begin(&self.device))
    }

    fn submit(&self, encoder: CudaEncoder) -> Result<()> {
        encoder.submit()
    }

    fn read_logits(&self, logits: &CudaBuffer, vocab: usize) -> Result<Vec<f32>> {
        readback::logits(&self.device, logits, vocab)
    }

    fn read_argmax(&self, logits: &CudaBuffer, vocab: usize) -> Result<u32> {
        kernels::argmax::read(&self.device, &self.module, logits, &self.reduce, vocab)
    }

    fn read_topk(&self, logits: &CudaBuffer, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        kernels::topk::read(&self.device, &self.module, logits, &self.reduce, vocab, k)
    }

    fn kv_write(
        &self,
        encoder: &CudaEncoder,
        kv: &crate::kv_cache::Kv<CudaBuffer>,
        k: &CudaBuffer,
        v: &CudaBuffer,
        k_payload_offset: u64,
        v_payload_offset: u64,
        k_meta_offset: u64,
        v_meta_offset: u64,
        vectors: u32,
    ) -> Result<()> {
        kernels::kv_write::encode(
            encoder,
            &self.module,
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
        encoder: &CudaEncoder,
        out: &CudaBuffer,
        weight: &CudaBuffer,
        token: u32,
        width: u32,
    ) -> Result<()> {
        kernels::embedding::encode(encoder, &self.module, out, weight, token, width)
    }

    fn matmul(
        &self,
        encoder: &CudaEncoder,
        out: &CudaBuffer,
        input: &CudaBuffer,
        weight: &CudaBuffer,
        input_width: u32,
        output_width: u32,
    ) {
        encoder.latch(kernels::matmul::encode(
            encoder,
            &self.module,
            out,
            input,
            weight,
            input_width,
            output_width,
            false,
        ));
    }

    fn matmul_batched(
        &self,
        encoder: &CudaEncoder,
        out: &CudaBuffer,
        input: &CudaBuffer,
        weight: &CudaBuffer,
        input_width: u32,
        output_width: u32,
        rows: u32,
    ) {
        encoder.latch(kernels::matmul::encode_batched(
            encoder,
            &self.module,
            out,
            input,
            weight,
            input_width,
            output_width,
            rows,
        ));
    }

    fn logits(
        &self,
        encoder: &CudaEncoder,
        out: &CudaBuffer,
        input: &CudaBuffer,
        weight: &CudaBuffer,
        input_width: u32,
        output_width: u32,
    ) {
        encoder.latch(kernels::matmul::encode(
            encoder,
            &self.module,
            out,
            input,
            weight,
            input_width,
            output_width,
            true,
        ));
    }

    fn rmsnorm_x(
        &self,
        encoder: &CudaEncoder,
        out: &CudaBuffer,
        input: &CudaBuffer,
        weight: &CudaBuffer,
        width: u32,
        epsilon: f32,
        rows: u32,
    ) {
        encoder.latch(kernels::normalization::encode(
            encoder,
            &self.module,
            out,
            input,
            weight,
            width,
            epsilon,
            rows,
        ));
    }

    fn rope_yarn(
        &self,
        encoder: &CudaEncoder,
        values: &CudaBuffer,
        heads: u32,
        head_dim: u32,
        position: u32,
        yarn: &crate::backend::rope::Yarn,
        role: crate::backend::rope::RopeRole,
    ) -> Result<()> {
        kernels::rope::encode(
            encoder,
            &self.module,
            values,
            heads,
            head_dim,
            position,
            yarn,
            role,
        )
    }

    fn silu_mul(
        &self,
        encoder: &CudaEncoder,
        out: &CudaBuffer,
        gate: &CudaBuffer,
        up: &CudaBuffer,
        length: u32,
    ) {
        encoder.latch(kernels::silu_mul::encode(
            encoder,
            &self.module,
            out,
            gate,
            up,
            length,
        ));
    }

    fn residual_add(&self, encoder: &CudaEncoder, x: &CudaBuffer, y: &CudaBuffer, length: u32) {
        encoder.latch(kernels::residual_add::encode(
            encoder,
            &self.module,
            x,
            y,
            length,
        ));
    }

    fn attention_decode(
        &self,
        encoder: &CudaEncoder,
        out: &CudaBuffer,
        query: &CudaBuffer,
        kv: &crate::kv_cache::Kv<CudaBuffer>,
        q_heads: u32,
        position: u32,
        layer: u32,
    ) {
        encoder.latch(kernels::attention::decode(
            encoder,
            &self.module,
            out,
            query,
            kv,
            q_heads,
            position,
            layer,
        ));
    }

    fn attention_prefill(
        &self,
        encoder: &CudaEncoder,
        out: &CudaBuffer,
        query: &CudaBuffer,
        kv: &crate::kv_cache::Kv<CudaBuffer>,
        q_heads: u32,
        base: u32,
        rows: u32,
        layer: u32,
    ) {
        encoder.latch(kernels::attention::prefill(
            encoder,
            &self.module,
            out,
            query,
            kv,
            q_heads,
            base,
            rows,
            layer,
        ));
    }
}
