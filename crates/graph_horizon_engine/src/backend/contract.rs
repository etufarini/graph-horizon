/*
 * graph_horizon_engine — backend contract
 * This file contains the single model-agnostic `Backend` trait. It names the
 * operations the dense graph may request, keeps Q/K/V dimensions explicit, and
 * leaves resource ownership, feature selection, CPU/Vulkan modules and numeric
 * kernels to sibling files.
*/

// AGENTS deroga I: definizione del trait Backend — singolo costrutto irriducibile.

use color_eyre::eyre::Result;

use super::buffers;

// The backend boundary for the forward path. `Buffer` is an opaque handle that
// carries its own format; `Encoder` is a per-position recording session. The
// per-request KV cache lives in `kv_cache`, layered over the buffer primitives
// below. `Sized` (one backend per build) so the crate is generic-monomorphized,
// never `dyn`.
pub(crate) trait Backend: Sized {
    type Buffer;
    type Encoder;

    // Access to weights/scratch/logits.
    fn buffers(&self) -> &buffers::Buffers<Self::Buffer>;

    // Raw buffer primitives backing the per-request KV cache (owned by the
    // `kv_cache` module): allocate at the start, free always (even on error).
    fn alloc_buffer(&self, bytes: u64) -> Result<Self::Buffer>;
    fn free_buffer(&self, buf: Self::Buffer);

    // Returns a handle that aliases `buf`'s storage over the window
    // `[offset_bytes, offset_bytes + len_bytes)`. It does not allocate and does
    // not copy: the returned handle shares the parent's backing storage, so a
    // write through the view is observed through the parent at the same window
    // and vice versa.
    //
    // Ownership invariant: a view MUST NOT be passed to `free_buffer` (nor
    // otherwise freed/destroyed). It owns only a logical reference to the
    // parent's storage; the parent stays the sole owner of the allocation.
    //
    // Contract: `offset_bytes + len_bytes <= buf.size`. Violating it is a caller
    // bug (it would bind past the buffer on Vulkan / panic on a CPU slice). The
    // prefill path (graph::prefill) computes and validates these offsets from
    // known dimensions before constructing any view, so this method does not add
    // a per-view runtime error path. Intended for transient use in the batched
    // prefill path, not for persistent state.
    fn view(&self, buf: &Self::Buffer, offset_bytes: u64, len_bytes: u64) -> Self::Buffer;

    // Minimum byte alignment a sub-buffer (`view`) binding offset must satisfy on
    // this backend. The batched prefill (`graph::prefill`) checks its N-wide row
    // strides against this before building any row-view, so a misaligned binding
    // never reaches `view`. Vulkan returns the device's
    // `min_storage_buffer_offset_alignment`; the CPU backend slices a plain `Vec`
    // and has no such constraint, so it returns `1`.
    fn min_buffer_offset_alignment(&self) -> u64;

    // Write the current token's k/v into the caches (`vectors` vectors of
    // `kv.head_dim` values each), quantizing per `kv.scheme`. KV quantization is
    // selected at load via `Kv::scheme`; backends branch once per call.
    // Per-role payload and metadata offsets are byte origins precomputed by
    // `kv_cache`; K and V may have different validated logical widths.
    #[allow(clippy::too_many_arguments)]
    fn kv_write(
        &self,
        enc: &Self::Encoder,
        kv: &crate::kv_cache::Kv<Self::Buffer>,
        k: &Self::Buffer,
        v: &Self::Buffer,
        k_payload_offset: u64,
        v_payload_offset: u64,
        k_meta_offset: u64,
        v_meta_offset: u64,
        vectors: u32,
    ) -> Result<()>;

    // Open a position's recording session; submit and wait (synchronous).
    fn begin(&self) -> Result<Self::Encoder>;
    fn submit(&self, enc: Self::Encoder) -> Result<()>;

    // Hint that the NEXT recorded kernel is independent of the one that follows it,
    // so the backend may omit the inter-dispatch barrier after it. Used to coalesce
    // a run of mutually-independent dispatches (e.g. Q/K/V or gate/up projections)
    // behind a single barrier instead of one each. Purely a scheduling hint: it
    // never changes the math (a backend that ignores it stays correct), so the CPU
    // reference — which records nothing and runs sequentially — leaves it a no-op.
    fn no_barrier(&self) {}

    // Dense graph operations record onto the encoder using buffer handles and dimensions.
    // Embedding lookup (row copy), widening F16→FP32 into the residual `x`.
    fn embed(
        &self,
        enc: &Self::Encoder,
        x: &Self::Buffer,
        token_embd: &Self::Buffer,
        token: u32,
        embd: u32,
    ) -> Result<()>;
    // y = W·a; the format of `w` picks the kernel (quant stays internal).
    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        enc: &Self::Encoder,
        out: &Self::Buffer,
        a: &Self::Buffer,
        w: &Self::Buffer,
        in_dim: u32,
        out_dim: u32,
    );
    // Batched `y = W·a` over `n` tokens packed token-major: `a` is `[n][in_dim]`
    // and `out` is `[n][out_dim]` (FP16 rows, contiguous per token). The default
    // loops the per-token `matmul` over row-views, so every backend is correct
    // without an override and the result is bit-identical to the per-token path.
    // The CPU backend overrides it to dequantize each weight row ONCE and reuse it
    // across all `n` activation columns — the prefill bandwidth/dequant
    // amortization (the per-token `matmul` re-reads the whole weight matrix per
    // token). `in_dim`/`out_dim` are the single-token dims; `n` the prompt length.
    #[allow(clippy::too_many_arguments)]
    fn matmul_batched(
        &self,
        enc: &Self::Encoder,
        out: &Self::Buffer,
        a: &Self::Buffer,
        w: &Self::Buffer,
        in_dim: u32,
        out_dim: u32,
        n: u32,
    ) {
        let a_stride = in_dim as u64 * 2; // FP16 activation row
        let o_stride = out_dim as u64 * 2; // FP16 output row
        // The n per-token matmuls write disjoint output rows from disjoint inputs, so
        // they are mutually independent: only the LAST needs its trailing barrier (the
        // next consumer reads all n rows). Eliding the n-1 intermediate barriers avoids
        // n-1 full pipeline drains per batched projection — the bulk of the prefill
        // barrier overhead — without changing the math (`no_barrier` is a no-op on the
        // CPU reference, which records nothing).
        for i in 0..n as u64 {
            let ai = self.view(a, i * a_stride, a_stride);
            let oi = self.view(out, i * o_stride, o_stride);
            if i + 1 < n as u64 {
                self.no_barrier();
            }
            self.matmul(enc, &oi, &ai, w, in_dim, out_dim);
        }
    }
    // Same as matmul but the output is the FP32 vocab-sized logits.
    #[allow(clippy::too_many_arguments)]
    fn logits(
        &self,
        enc: &Self::Encoder,
        out: &Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        in_dim: u32,
        out_dim: u32,
    );
    // RMSNorm reading the FP32 residual stream (hidden-state norm).
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_x(
        &self,
        enc: &Self::Encoder,
        out: &Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        dim: u32,
        eps: f32,
        rows: u32,
    );
    // Ministral YaRN variant. The role is explicit because only Q receives the
    // post-RoPE attention-temperature scale; K remains neutral.
    #[allow(clippy::too_many_arguments)]
    fn rope_yarn(
        &self,
        enc: &Self::Encoder,
        x: &Self::Buffer,
        heads: u32,
        head_dim: u32,
        pos: u32,
        yarn: &crate::backend::rope::Yarn,
        role: crate::backend::rope::RopeRole,
    ) -> Result<()>;

    // Rotates packed Q/K rows. The default preserves backend portability while
    // eliding barriers between disjoint rows; GPU backends may override it with
    // one dispatch per role.
    #[allow(clippy::too_many_arguments)]
    fn rope_yarn_batched(
        &self,
        enc: &Self::Encoder,
        q: &Self::Buffer,
        k: &Self::Buffer,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        base: u32,
        rows: u32,
        yarn: &crate::backend::rope::Yarn,
    ) -> Result<()> {
        if rows == 0 {
            return Err(color_eyre::eyre::eyre!("rope: empty batch"));
        }
        let q_stride = u64::from(q_heads)
            .checked_mul(u64::from(head_dim))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| color_eyre::eyre::eyre!("rope: buffer size overflow"))?;
        let k_stride = u64::from(kv_heads)
            .checked_mul(u64::from(head_dim))
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| color_eyre::eyre::eyre!("rope: buffer size overflow"))?;
        for row in 0..rows {
            let position = base
                .checked_add(row)
                .ok_or_else(|| color_eyre::eyre::eyre!("rope: position overflow"))?;
            let q_offset = u64::from(row)
                .checked_mul(q_stride)
                .ok_or_else(|| color_eyre::eyre::eyre!("rope: buffer size overflow"))?;
            let k_offset = u64::from(row)
                .checked_mul(k_stride)
                .ok_or_else(|| color_eyre::eyre::eyre!("rope: buffer size overflow"))?;
            let q_row = self.view(q, q_offset, q_stride);
            let k_row = self.view(k, k_offset, k_stride);
            self.no_barrier();
            self.rope_yarn(
                enc,
                &q_row,
                q_heads,
                head_dim,
                position,
                yarn,
                crate::backend::rope::RopeRole::Query,
            )?;
            if row + 1 < rows {
                self.no_barrier();
            }
            self.rope_yarn(
                enc,
                &k_row,
                kv_heads,
                head_dim,
                position,
                yarn,
                crate::backend::rope::RopeRole::Key,
            )?;
        }
        Ok(())
    }

    // Fused `SiLU(gate) ⊙ up`. The intermediate is rounded to FP16 before the
    // product, preserving the graph's sequential FP16-memory formula:
    // `out[i] = f16(f16(silu(gate[i])) * up[i])`. Backend tests compare this
    // operation with that sequential numeric reference.
    fn silu_mul(
        &self,
        enc: &Self::Encoder,
        out: &Self::Buffer,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        n: u32,
    );

    fn residual_add(&self, enc: &Self::Encoder, x: &Self::Buffer, y: &Self::Buffer, n: u32);
    // Causal GQA attention of the current token over cached positions 0..=pos.
    // The cache handle carries the buffers, the shape (head_dim/kv_heads/
    // context) and the scheme; backends branch on `kv.scheme` once per call and
    // derive metadata origins from the cache's per-role layout.
    #[allow(clippy::too_many_arguments)]
    fn attention_decode(
        &self,
        enc: &Self::Encoder,
        out: &Self::Buffer,
        q: &Self::Buffer,
        kv: &crate::kv_cache::Kv<Self::Buffer>,
        q_heads: u32,
        pos: u32,
        layer: u32,
    );

    // Causal GQA attention for N query in a single dispatch: the generalization
    // of `attention_decode` over a row of queries, sharing the same online-softmax
    // primitive. `q`/`out` are N-wide, laid out `[n][q_heads][head_dim]` (FP16);
    // the KV cache is the layer's `[layer][token][kv_head][head_dim]` cache, which
    // the caller must have populated for positions `base..base+n` before this call.
    // For each row `i in 0..n` the query at absolute position `base+i` attends
    // `t in 0..=base+i`; scale is `1/sqrt(head_dim)`, GQA maps query head h to kv
    // head `h / (q_heads/kv_heads)`. With `n == 1` and `base == pos` it degenerates
    // to `attention_decode(pos)`.
    #[allow(clippy::too_many_arguments)]
    fn attention_prefill(
        &self,
        enc: &Self::Encoder,
        out: &Self::Buffer,
        q: &Self::Buffer,
        kv: &crate::kv_cache::Kv<Self::Buffer>,
        q_heads: u32,
        base: u32,
        n: u32,
        layer: u32,
    );

    // Copy the FP32 logits to host.
    fn read_logits(&self, logits: &Self::Buffer, vocab: usize) -> Result<Vec<f32>>;

    // Device-side reductions of the FP32 logits for the decode hot-path: instead
    // of copying the whole vocab to host every token (`read_logits`), reduce on the
    // device and read back only what sampling needs — the argmax index (greedy) or
    // the k highest candidates (top-k). The Vulkan backend runs a real reduction
    // shader; the CPU backend scans the host-resident buffer (no transfer to hide).
    //
    // Tiebreak invariant (shared with `sampling.rs`, a property of this trait):
    //   - `read_argmax`: on equal logit value the LOWEST index wins (strict `>`,
    //     first occurrence) — identical to `sampling::sample`'s greedy argmax.
    //   - `read_topk`: returns exactly `min(k, vocab)` pairs under the total order
    //     `(logit desc, index asc)`, IN that order — identical comparator to
    //     `sampling.rs`: `b.1.total_cmp(&a.1).then(a.0.cmp(&b.0))`. The values are
    //     the raw FP32 logits (the caller applies temperature), so the selection is
    //     bit-exact with the host path even on ties at the k/k+1 border.
    fn read_argmax(&self, logits: &Self::Buffer, vocab: usize) -> Result<u32>;
    fn read_topk(&self, logits: &Self::Buffer, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>>;
}
