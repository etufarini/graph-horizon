/*
 * graph_horizon_engine — Vulkan backend trait implementation
 * Contains the single `Backend for VulkanBackend` implementation. Each method
 * delegates trait operations to the state and domain functions owned elsewhere;
 * this file defines no backend state, budget policy, trace setting, or resource
 * lifecycle.
*/

// Every `unsafe {}` in the vulkan subtree must carry a `// SAFETY:` stating its real
// invariant (handle lifetime, map'd-pointer size/alignment, validated GGUF dims). The
// deny is module-scoped and inherited by all submodules; new undocumented unsafe breaks
// the build rather than slipping in silently.
#![deny(clippy::undocumented_unsafe_blocks)]

use super::exec::{dispatch, readback};

use ash::vk;
use color_eyre::eyre::Result;

use crate::backend::Backend;
use crate::backend::buffers::Buffers;
#[cfg(feature = "vulkan")]
use crate::backend::source::WeightSource;
#[cfg(feature = "vulkan")]
use crate::gguf::loader::GgufFile;
#[cfg(feature = "vulkan")]
use crate::gguf::metadata::ModelMetadata;

use super::VulkanBackend;
use super::buffers::GpuBuffer;
#[cfg(feature = "vulkan")]
use super::device::Device;
use super::kernels::attention::{
    attention_decode, attention_decode_int8, attention_prefill, attention_prefill_int8, kv_write,
    kv_write_int8,
};
use super::kernels::elementwise::{residual_add, rmsnorm_x};
use super::kernels::fused;
use super::kernels::matmul::logits;
use super::kernels::reduce;

// AGENTS deroga I: singolo `impl Backend for VulkanBackend` di delegatori sottili.
impl Backend for VulkanBackend {
    type Buffer = GpuBuffer;
    type Encoder = vk::CommandBuffer;

    #[cfg(feature = "vulkan")]
    fn load(
        meta: &ModelMetadata,
        ws: &dyn WeightSource,
        gguf: &GgufFile,
        context: usize,
    ) -> Result<Self> {
        let dev = Device::init().map_err(super::init::pure_loader_unavailable)?;
        Self::load_inner(dev, meta, ws, gguf, context)
    }

    fn buffers(&self) -> &Buffers<GpuBuffer> {
        &self.buf
    }

    fn alloc_buffer(&self, bytes: u64) -> Result<GpuBuffer> {
        GpuBuffer::alloc(&self.dev, bytes, false)
    }

    fn free_buffer(&self, buf: GpuBuffer) {
        buf.destroy(&self.dev);
    }

    // Aliases `buf`'s storage over [offset_bytes, offset_bytes + len_bytes).
    // The returned handle shares the parent's VkBuffer/memory and must not be
    // freed (see GpuBuffer::view / the `offset` field comment).
    fn view(&self, buf: &GpuBuffer, offset_bytes: u64, len_bytes: u64) -> GpuBuffer {
        buf.view(offset_bytes, len_bytes)
    }

    // The device's required start alignment for a storage-buffer binding offset.
    fn min_buffer_offset_alignment(&self) -> u64 {
        self.dev.min_storage_buffer_offset_alignment
    }

    fn kv_write(
        &self,
        enc: &vk::CommandBuffer,
        kv: &crate::kv_cache::Kv<GpuBuffer>,
        k: &GpuBuffer,
        v: &GpuBuffer,
        k_payload_offset: u64,
        v_payload_offset: u64,
        k_meta_offset: u64,
        _v_meta_offset: u64,
        vectors: u32,
    ) -> Result<()> {
        // One branch per call on the load-time scheme (I2).
        match kv.scheme {
            // f16: the byte offset maps back to the historic element offset
            // (2 bytes/element); same pure-copy kernel as before the refactor.
            crate::kv_cache::scheme::KvQuant::F16 => kv_write(
                &self.dev,
                &self.reg,
                *enc,
                &kv.k,
                &kv.v,
                k,
                v,
                (k_payload_offset / 2) as u32,
                vectors * kv.head_dim as u32,
            ),
            crate::kv_cache::scheme::KvQuant::Int8 => kv_write_int8(
                &self.dev,
                &self.reg,
                *enc,
                &kv.k,
                &kv.v,
                k,
                v,
                k_payload_offset as u32,
                k_meta_offset as u32,
                vectors,
                kv.head_dim as u32,
            ),
        }
        // Current Vulkan attention shaders use equal K/V head widths. Mistral
        // artifacts satisfy this; CPU owns the validated unequal-width oracle.
        debug_assert_eq!(k_payload_offset, v_payload_offset);
        debug_assert_eq!(kv.head_dim, kv.value_dim);
        Ok(())
    }

    fn begin(&self) -> Result<vk::CommandBuffer> {
        self.dev.begin_commands()
    }

    fn submit(&self, enc: vk::CommandBuffer) -> Result<()> {
        // Test-only instrumentation: counts model-forward submits so lifecycle tests
        // can assert exactly one `backend.submit` per token (I2). The logits readback
        // uses `read_buffer` (a device-level `submit_wait`), so it is NOT counted here.
        #[cfg(test)]
        super::record_submit();
        self.dev.submit_wait(enc)
    }

    // Mark the next dispatch as not requiring its trailing barrier (see the trait
    // doc and `device::Device::skip_next_barrier`). One-shot, consumed by the next
    // recorded kernel.
    fn no_barrier(&self) {
        self.dev
            .skip_next_barrier
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    // Embedding lookup into the FP32 residual stream. FP16 token_embd is a
    // widening copy; Q4_K/Q5_K/Q6_K are dequantized on the fly (Q6_K covers the
    // tied-embedding lm_head of the Q4_K_M/Q6_K causal models).
    fn embed(
        &self,
        enc: &vk::CommandBuffer,
        x: &GpuBuffer,
        token_embd: &GpuBuffer,
        token: u32,
        embd: u32,
    ) -> Result<()> {
        dispatch::embed(self, enc, x, token_embd, token, embd)
    }

    fn matmul(
        &self,
        enc: &vk::CommandBuffer,
        out: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        in_dim: u32,
        out_dim: u32,
    ) {
        dispatch::matmul(self, enc, out, a, w, in_dim, out_dim);
    }

    fn matmul_batched(
        &self,
        enc: &vk::CommandBuffer,
        out: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        in_dim: u32,
        out_dim: u32,
        n: u32,
    ) {
        dispatch::matmul_batched(self, enc, out, a, w, in_dim, out_dim, n);
    }

    fn logits(
        &self,
        enc: &vk::CommandBuffer,
        out: &GpuBuffer,
        x: &GpuBuffer,
        w: &GpuBuffer,
        in_dim: u32,
        out_dim: u32,
    ) {
        logits(&self.dev, &self.reg, *enc, out, x, w, in_dim, out_dim);
    }

    fn rmsnorm_x(
        &self,
        enc: &vk::CommandBuffer,
        out: &GpuBuffer,
        x: &GpuBuffer,
        w: &GpuBuffer,
        dim: u32,
        eps: f32,
        rows: u32,
    ) {
        rmsnorm_x(&self.dev, &self.reg, *enc, out, x, w, dim, eps, rows);
    }

    fn rope_yarn(
        &self,
        enc: &vk::CommandBuffer,
        x: &GpuBuffer,
        heads: u32,
        head_dim: u32,
        pos: u32,
        yarn: &crate::backend::rope::Yarn,
        role: crate::backend::rope::RopeRole,
    ) -> Result<()> {
        super::kernels::elementwise::rope_yarn(
            &self.dev,
            &self.reg,
            *enc,
            x,
            heads,
            head_dim,
            yarn.rope_dim as u32,
            pos,
            yarn.freq_base,
            yarn.factor,
            yarn.beta_fast,
            yarn.beta_slow,
            yarn.original_context as u32,
            yarn.post_scale(role, pos as usize),
        );
        Ok(())
    }

    fn silu_mul(
        &self,
        enc: &vk::CommandBuffer,
        out: &GpuBuffer,
        gate: &GpuBuffer,
        up: &GpuBuffer,
        n: u32,
    ) {
        fused::silu_mul(&self.dev, &self.reg, *enc, out, gate, up, n);
    }

    fn residual_add(&self, enc: &vk::CommandBuffer, x: &GpuBuffer, y: &GpuBuffer, n: u32) {
        residual_add(&self.dev, &self.reg, *enc, x, y, n);
    }

    fn attention_decode(
        &self,
        enc: &vk::CommandBuffer,
        out: &GpuBuffer,
        q: &GpuBuffer,
        kv: &crate::kv_cache::Kv<GpuBuffer>,
        q_heads: u32,
        pos: u32,
        layer: u32,
    ) {
        match kv.scheme {
            crate::kv_cache::scheme::KvQuant::F16 => attention_decode(
                &self.dev,
                &self.reg,
                *enc,
                out,
                q,
                &kv.k,
                &kv.v,
                &self.reduce,
                &self.mmvq_ds,
                kv.head_dim as u32,
                kv.kv_heads as u32,
                q_heads,
                pos,
                layer,
                kv.context as u32,
            ),
            crate::kv_cache::scheme::KvQuant::Int8 => attention_decode_int8(
                &self.dev,
                &self.reg,
                *enc,
                out,
                q,
                &kv.k,
                &kv.v,
                kv.head_dim as u32,
                kv.kv_heads as u32,
                q_heads,
                pos,
                layer,
                kv.context as u32,
                kv.meta_base() as u32,
            ),
        }
    }

    fn attention_prefill(
        &self,
        enc: &vk::CommandBuffer,
        out: &GpuBuffer,
        q: &GpuBuffer,
        kv: &crate::kv_cache::Kv<GpuBuffer>,
        q_heads: u32,
        base: u32,
        n: u32,
        layer: u32,
    ) {
        match kv.scheme {
            crate::kv_cache::scheme::KvQuant::F16 => attention_prefill(
                &self.dev,
                &self.reg,
                *enc,
                out,
                q,
                &kv.k,
                &kv.v,
                kv.head_dim as u32,
                kv.kv_heads as u32,
                q_heads,
                base,
                n,
                layer,
                kv.context as u32,
            ),
            crate::kv_cache::scheme::KvQuant::Int8 => attention_prefill_int8(
                &self.dev,
                &self.reg,
                *enc,
                out,
                q,
                &kv.k,
                &kv.v,
                kv.head_dim as u32,
                kv.kv_heads as u32,
                q_heads,
                base,
                n,
                layer,
                kv.context as u32,
                kv.meta_base() as u32,
            ),
        }
    }

    fn read_logits(&self, logits: &GpuBuffer, vocab: usize) -> Result<Vec<f32>> {
        readback::read_logits(self, logits, vocab)
    }

    // Device-side argmax/top-k: delegate to `reduce`, which dispatches the shader,
    // copies the minimal result into `logits_host` and merges (top-k) on the host.
    fn read_argmax(&self, logits: &GpuBuffer, vocab: usize) -> Result<u32> {
        reduce::argmax(
            &self.dev,
            &self.reg,
            logits,
            &self.reduce,
            &self.logits_host,
            vocab,
        )
    }

    fn read_topk(&self, logits: &GpuBuffer, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
        reduce::topk(
            &self.dev,
            &self.reg,
            logits,
            &self.reduce,
            &self.logits_host,
            vocab,
            k,
        )
    }
}
