/*
 * graph_horizon_engine — request-session KV storage
 * Owns the `Kv` type (the key/value buffers plus the shape and the runtime
 * scheme needed to index them) and orchestrates its lifecycle as a thin layer
 * over the backend: allocate both buffers transactionally, append tokens, and
 * free both buffers exactly once. Every GPU action goes through the `Backend` primitives
 * (`alloc_buffer`/`free_buffer`/`kv_write`); all layout arithmetic and the
 * position invariant live in `layout`, the per-scheme byte sizes in `scheme`.
 * The scheme is chosen once at load (I1) and fixed before any alloc; backends
 * branch on it once per call (I2). Each buffer holds a payload region followed
 * by a metadata region. K and V retain their own logical head widths and their
 * own checked payload/metadata offsets. A homogeneous GPU session may transfer
 * ownership to one serialized model cache; errors and final model drop still
 * release it. No raw math and no concrete GPU API here.
*/

#[cfg(any(
    feature = "cpu",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    test
))]
pub(crate) mod int8;
pub(crate) mod layout;
pub(crate) mod scheme;

use color_eyre::eyre::Result;

use crate::backend::Backend;
use crate::kv_cache::scheme::{KvQuant, KvRole};

// Request-session KV storage: the key/value buffers plus the shape and scheme
// needed to index them. Allocation, append, and exact-once free are below.
pub(crate) struct Kv<Buf> {
    pub k: Buf,
    pub v: Buf,
    pub scheme: KvQuant,
    pub block_count: usize,
    pub context: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub value_dim: usize,
}

impl<Buf> Kv<Buf> {
    // Byte base of the metadata region (the payload region of all layers
    // precedes it; role-independent since payload sizes are). The single
    // layout truth backends read instead of re-deriving the region split (D6).
    #[cfg(any(
        feature = "vulkan",
        feature = "vulkan-hybrid",
        all(test, any(feature = "cpu", feature = "metal-hybrid"))
    ))]
    pub(crate) fn meta_base(&self) -> u64 {
        self.meta_base_for(KvRole::Key)
    }

    pub(crate) fn meta_base_for(&self, role: KvRole) -> u64 {
        let dim = match role {
            KvRole::Key => self.head_dim,
            KvRole::Value => self.value_dim,
        };
        layout::meta_base(
            self.scheme,
            self.block_count,
            self.context,
            self.kv_heads,
            dim,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn alloc_shape<B: Backend>(
    backend: &B,
    block_count: usize,
    context: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    scheme: KvQuant,
) -> Result<Kv<B::Buffer>> {
    // Per-role sizing: the prod schemes carry more metadata on K than on V.
    let k_bytes = layout::buffer_bytes(
        scheme,
        KvRole::Key,
        block_count,
        context,
        kv_heads,
        head_dim,
    );
    let v_bytes = layout::buffer_bytes(
        scheme,
        KvRole::Value,
        block_count,
        context,
        kv_heads,
        value_dim,
    );
    let k = backend.alloc_buffer(k_bytes)?;
    let v = match backend.alloc_buffer(v_bytes) {
        Ok(v) => v,
        Err(e) => {
            backend.free_buffer(k);
            return Err(e);
        }
    };
    Ok(Kv {
        k,
        v,
        scheme,
        block_count,
        context,
        kv_heads,
        head_dim,
        value_dim,
    })
}

// Writes the current token's K/V at (layer, pos): validates the position, then
// asks the backend to quantize/copy into the caches at the precomputed byte
// offsets (payload and metadata region origins of the token's kv_heads
// vectors). Offsets are computed HERE from `layout` — backends never re-derive
// the region split (D6).
pub(crate) fn append<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    kv: &Kv<B::Buffer>,
    layer: usize,
    pos: usize,
    k: &B::Buffer,
    v: &B::Buffer,
) -> Result<()> {
    append_batch(backend, enc, kv, layer, pos, k, v, 1)
}

// Writes one contiguous prompt batch. The checked layout is token-major, so
// `rows * kv_heads` vectors occupy one contiguous region in both caches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn append_batch<B: Backend>(
    backend: &B,
    enc: &B::Encoder,
    kv: &Kv<B::Buffer>,
    layer: usize,
    pos: usize,
    k: &B::Buffer,
    v: &B::Buffer,
    rows: usize,
) -> Result<()> {
    layout::check_pos(pos, kv.context)?;
    if rows == 0 || pos.checked_add(rows).is_none_or(|end| end > kv.context) {
        color_eyre::eyre::bail!("kv_cache: batch beyond context length");
    }
    let k_payload_offset =
        layout::payload_offset(kv.scheme, layer, pos, kv.kv_heads, kv.head_dim, kv.context);
    let v_payload_offset =
        layout::payload_offset(kv.scheme, layer, pos, kv.kv_heads, kv.value_dim, kv.context);
    let meta = |role, dim| {
        layout::meta_offset(
            kv.scheme,
            role,
            kv.block_count,
            layer,
            pos,
            kv.kv_heads,
            dim,
            kv.context,
        )
    };
    backend.kv_write(
        enc,
        kv,
        k,
        v,
        k_payload_offset,
        v_payload_offset,
        meta(KvRole::Key, kv.head_dim),
        meta(KvRole::Value, kv.value_dim),
        (rows * kv.kv_heads) as u32,
    )
}

// Releases both buffers. Taking `kv` by value frees the cache exactly once.
pub(crate) fn free<B: Backend>(backend: &B, kv: Kv<B::Buffer>) {
    backend.free_buffer(kv.k);
    backend.free_buffer(kv.v);
}
