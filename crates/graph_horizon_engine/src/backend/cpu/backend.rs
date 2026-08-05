/*
 * graph_horizon_engine — CPU Backend implementation
 * This file contains exactly the single `Backend for CpuBackend` delegator.
 * Concrete state and construction live in `cpu/mod.rs`; this implementation
 * owns no graph order or kernel algorithm and forwards each operation to the
 * focused buffer, dispatch, kernel, readback, or weight module.
*/

use color_eyre::eyre::Result;

use crate::backend::Backend;
use crate::backend::buffers::Buffers;
#[cfg(any(test, not(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))))]
use crate::backend::source::WeightSource;
#[cfg(any(test, not(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))))]
use crate::gguf::loader::GgufFile;
#[cfg(any(test, not(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))))]
use crate::gguf::metadata::ModelMetadata;

use super::buffer::{CpuBuffer, CpuFormat};
#[cfg(any(test, not(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))))]
use super::weights;
use super::{CpuBackend, CpuEncoder, dispatch, kernels, readback};

// AGENTS deroga I: singolo `impl Backend for CpuBackend` di delegatori sottili.
impl Backend for CpuBackend {
    type Buffer = CpuBuffer;
    type Encoder = CpuEncoder;

    #[cfg(any(test, not(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))))]
    fn load(
        meta: &ModelMetadata,
        ws: &dyn WeightSource,
        gguf: &GgufFile,
        context: usize,
    ) -> Result<Self> {
        let buffers = weights::load(meta, ws, gguf, context)?;
        Ok(CpuBackend { buffers })
    }

    fn buffers(&self) -> &Buffers<CpuBuffer> {
        &self.buffers
    }

    // KV cache storage is FP16 (kv_cache::alloc passes elems*2 bytes).
    fn alloc_buffer(&self, bytes: u64) -> Result<CpuBuffer> {
        Ok(CpuBuffer::zeroed(bytes as usize, CpuFormat::F16))
    }

    // Dropping the buffer frees its backing Vec.
    fn free_buffer(&self, buf: CpuBuffer) {
        drop(buf);
    }

    // Aliases `buf`'s storage over [offset_bytes, offset_bytes + len_bytes). The
    // view shares the parent's bytes (Arc) and must not be freed (it only drops a
    // reference count). Byte→byte, no cast.
    fn view(&self, buf: &CpuBuffer, offset_bytes: u64, len_bytes: u64) -> CpuBuffer {
        buf.view(offset_bytes as usize, len_bytes as usize)
    }

    // Views slice a plain Vec: any byte offset is legal, so no alignment to honor.
    fn min_buffer_offset_alignment(&self) -> u64 {
        1
    }

    fn begin(&self) -> Result<CpuEncoder> {
        Ok(CpuEncoder)
    }

    fn submit(&self, _enc: CpuEncoder) -> Result<()> {
        Ok(())
    }

    fn read_logits(&self, logits: &CpuBuffer, vocab: usize) -> Result<Vec<f32>> {
        readback::read_logits(logits, vocab)
    }

    fn read_argmax(&self, logits: &CpuBuffer, vocab: usize) -> Result<u32> {
        readback::read_argmax(logits, vocab)
    }

    fn read_topk(&self, logits: &CpuBuffer, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        readback::read_topk(logits, vocab, k)
    }

    #[allow(clippy::too_many_arguments)]
    fn kv_write(
        &self,
        _enc: &CpuEncoder,
        kv: &crate::kv_cache::Kv<CpuBuffer>,
        k: &CpuBuffer,
        v: &CpuBuffer,
        k_payload_offset: u64,
        v_payload_offset: u64,
        k_meta_offset: u64,
        v_meta_offset: u64,
        vectors: u32,
    ) -> Result<()> {
        kernels::attention::kv_write(
            kv,
            k,
            v,
            k_payload_offset,
            v_payload_offset,
            k_meta_offset,
            v_meta_offset,
            vectors as usize,
        );
        Ok(())
    }

    fn embed(
        &self,
        _enc: &CpuEncoder,
        x: &CpuBuffer,
        token_embd: &CpuBuffer,
        token: u32,
        embd: u32,
    ) -> Result<()> {
        kernels::elementwise::embed(x, token_embd, token as usize, embd as usize);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        _enc: &CpuEncoder,
        out: &CpuBuffer,
        a: &CpuBuffer,
        w: &CpuBuffer,
        in_dim: u32,
        out_dim: u32,
    ) {
        dispatch::matmul(out, a, w, in_dim, out_dim);
    }

    // Batched prefill matmul: dequant each weight row once, reuse across the N
    // prompt tokens (the per-token `matmul` above re-reads the weights N times).
    #[allow(clippy::too_many_arguments)]
    fn matmul_batched(
        &self,
        _enc: &CpuEncoder,
        out: &CpuBuffer,
        a: &CpuBuffer,
        w: &CpuBuffer,
        in_dim: u32,
        out_dim: u32,
        n: u32,
    ) {
        dispatch::matmul_batched(out, a, w, in_dim, out_dim, n);
    }

    #[allow(clippy::too_many_arguments)]
    fn logits(
        &self,
        _enc: &CpuEncoder,
        out: &CpuBuffer,
        x: &CpuBuffer,
        w: &CpuBuffer,
        in_dim: u32,
        out_dim: u32,
    ) {
        dispatch::logits(out, x, w, in_dim, out_dim);
    }

    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_x(
        &self,
        _enc: &CpuEncoder,
        out: &CpuBuffer,
        x: &CpuBuffer,
        w: &CpuBuffer,
        dim: u32,
        eps: f32,
        rows: u32,
    ) {
        kernels::elementwise::rmsnorm_x(out, x, w, dim as usize, eps, rows as usize);
    }

    fn rope_yarn(
        &self,
        _enc: &CpuEncoder,
        x: &CpuBuffer,
        heads: u32,
        head_dim: u32,
        pos: u32,
        yarn: &crate::backend::rope::Yarn,
        role: crate::backend::rope::RopeRole,
    ) -> Result<()> {
        kernels::elementwise::rope_yarn(
            x,
            heads as usize,
            head_dim as usize,
            pos as usize,
            yarn,
            role,
        )
    }

    fn silu_mul(
        &self,
        _enc: &CpuEncoder,
        out: &CpuBuffer,
        gate: &CpuBuffer,
        up: &CpuBuffer,
        n: u32,
    ) {
        kernels::elementwise::silu_mul(out, gate, up, n as usize);
    }

    fn residual_add(&self, _enc: &CpuEncoder, x: &CpuBuffer, y: &CpuBuffer, n: u32) {
        kernels::elementwise::residual_add(x, y, n as usize);
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_decode(
        &self,
        _enc: &CpuEncoder,
        out: &CpuBuffer,
        q: &CpuBuffer,
        kv: &crate::kv_cache::Kv<CpuBuffer>,
        q_heads: u32,
        pos: u32,
        layer: u32,
    ) {
        kernels::attention::attention_decode(
            out,
            q,
            kv,
            q_heads as usize,
            pos as usize,
            layer as usize,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill(
        &self,
        _enc: &CpuEncoder,
        out: &CpuBuffer,
        q: &CpuBuffer,
        kv: &crate::kv_cache::Kv<CpuBuffer>,
        q_heads: u32,
        base: u32,
        n: u32,
        layer: u32,
    ) {
        kernels::attention::attention_prefill(
            out,
            q,
            kv,
            q_heads as usize,
            base as usize,
            n as usize,
            layer as usize,
        );
    }
}
