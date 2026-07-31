/*
 * gh_zero_engine — KV-cache layout math (backend-agnostic)
 * Pure functions describing the shape and indexing of the KV cache. Each of the
 * K and V buffers holds TWO byte regions, both ordered [layer][token][kv_head]:
 * the payload region (the quantized codes, or the raw f16 values) followed by
 * the metadata region (the per-vector scalars: none for f16, min/scale for
 * int8). The unit is the VECTOR — the head_dim values of one (token, kv_head) —
 * whose per-scheme byte sizes come from `scheme::KvQuant`. Single source of
 * truth for the region bases and offsets: backends receive byte offsets from
 * here (via `kv_cache`) and never re-derive the region split (D6).
 *
 * No GPU handle and no `Backend` trait live here; the only error boundary is
 * the position check.
*/

use color_eyre::eyre::{Result, bail};

use crate::kv_cache::scheme::{KvQuant, KvRole};

// Vectors preceding the (layer, pos) token: the region index shared by the
// payload and metadata offsets ([layer][token][kv_head] order for both).
fn vectors_before(layer: usize, pos: usize, kv_heads: usize, context: usize) -> usize {
    (layer * context + pos) * kv_heads
}

// Vectors of one whole buffer: all layers × context × kv_heads.
fn total_vectors(block_count: usize, context: usize, kv_heads: usize) -> usize {
    block_count * context * kv_heads
}

// Byte origin of the (layer, pos) token's first vector inside the PAYLOAD region.
pub(crate) fn payload_offset(
    scheme: KvQuant,
    layer: usize,
    pos: usize,
    kv_heads: usize,
    head_dim: usize,
    context: usize,
) -> u64 {
    (vectors_before(layer, pos, kv_heads, context) * scheme.payload_bytes_per_vector(head_dim))
        as u64
}

// Byte base of the METADATA region: the payload region of ALL layers precedes it.
pub(crate) fn meta_base(
    scheme: KvQuant,
    block_count: usize,
    context: usize,
    kv_heads: usize,
    head_dim: usize,
) -> u64 {
    (total_vectors(block_count, context, kv_heads) * scheme.payload_bytes_per_vector(head_dim))
        as u64
}

// Byte origin of the (layer, pos) token's first vector metadata (absolute, from
// the buffer start: metadata base + the vector index × per-vector meta bytes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn meta_offset(
    scheme: KvQuant,
    role: KvRole,
    block_count: usize,
    layer: usize,
    pos: usize,
    kv_heads: usize,
    head_dim: usize,
    context: usize,
) -> u64 {
    meta_base(scheme, block_count, context, kv_heads, head_dim)
        + (vectors_before(layer, pos, kv_heads, context) * scheme.meta_bytes_per_vector(role))
            as u64
}

// Bytes of ONE cache buffer (k or v) for the whole context: payload region plus
// metadata region. For F16 this equals total_elems × 2 (no metadata), so the
// F16 sizing is unchanged.
pub(crate) fn buffer_bytes(
    scheme: KvQuant,
    role: KvRole,
    block_count: usize,
    context: usize,
    kv_heads: usize,
    head_dim: usize,
) -> u64 {
    let vectors = total_vectors(block_count, context, kv_heads);
    (vectors * (scheme.payload_bytes_per_vector(head_dim) + scheme.meta_bytes_per_vector(role)))
        as u64
}

// Refuse a position at or past the context window (stop, never out-of-bounds).
pub(crate) fn check_pos(pos: usize, context: usize) -> Result<()> {
    if pos >= context {
        bail!("kv_cache: position {pos} beyond context length {context}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-computed byte regions: 2 layers, context 8, kv_heads 2, head_dim 128.
    // 32 vectors total. f16: 256 B payload, 0 meta; int8: 128 B payload, 4 meta.
    #[test]
    fn byte_regions_match_hand_computed_cases() {
        let (bc, ctx, kvh, hd) = (2usize, 8usize, 2usize, 128usize);

        // f16: payload only, elements × 2.
        assert_eq!(
            buffer_bytes(KvQuant::F16, KvRole::Key, bc, ctx, kvh, hd),
            (32 * 256) as u64
        );
        assert_eq!(payload_offset(KvQuant::F16, 0, 0, kvh, hd, ctx), 0);
        assert_eq!(
            payload_offset(KvQuant::F16, 1, 3, kvh, hd, ctx),
            ((8 + 3) * 2 * 256) as u64
        );
        assert_eq!(meta_base(KvQuant::F16, bc, ctx, kvh, hd), (32 * 256) as u64);

        // int8: payload region then metadata region.
        assert_eq!(
            buffer_bytes(KvQuant::Int8, KvRole::Key, bc, ctx, kvh, hd),
            (32 * (128 + 4)) as u64
        );
        assert_eq!(
            payload_offset(KvQuant::Int8, 1, 3, kvh, hd, ctx),
            ((8 + 3) * 2 * 128) as u64
        );
        assert_eq!(
            meta_base(KvQuant::Int8, bc, ctx, kvh, hd),
            (32 * 128) as u64
        );
        assert_eq!(
            meta_offset(KvQuant::Int8, KvRole::Key, bc, 1, 3, kvh, hd, ctx),
            (32 * 128 + (8 + 3) * 2 * 4) as u64
        );
        // Non-zero layer/pos at the full-context extreme, non-trivial head_dim.
        assert_eq!(
            meta_offset(KvQuant::Int8, KvRole::Value, bc, 0, 7, kvh, hd, ctx),
            (32 * 128 + 7 * 2 * 4) as u64
        );
    }

    #[test]
    fn key_and_value_widths_keep_independent_regions() {
        let (bc, ctx, kvh, key_dim, value_dim) = (2usize, 8usize, 2usize, 8usize, 4usize);
        let scheme = KvQuant::Int8;
        let key_payload = payload_offset(scheme, 1, 3, kvh, key_dim, ctx);
        let value_payload = payload_offset(scheme, 1, 3, kvh, value_dim, ctx);
        let key_meta = meta_offset(scheme, KvRole::Key, bc, 1, 3, kvh, key_dim, ctx);
        let value_meta = meta_offset(scheme, KvRole::Value, bc, 1, 3, kvh, value_dim, ctx);
        assert_eq!(key_payload, 22 * key_dim as u64);
        assert_eq!(value_payload, 22 * value_dim as u64);
        assert_eq!(key_meta, (32 * key_dim + 22 * 4) as u64);
        assert_eq!(value_meta, (32 * value_dim + 22 * 4) as u64);
    }

    // The last vector of the last (layer, pos) touches exactly the end of its
    // region: payload end == metadata base, metadata end == buffer end.
    #[test]
    fn last_position_touches_exactly_the_region_end() {
        let (bc, ctx, kvh, hd) = (2usize, 8usize, 2usize, 128usize);
        for &scheme in KvQuant::ALL {
            for role in [KvRole::Key, KvRole::Value] {
                let pbv = scheme.payload_bytes_per_vector(hd) as u64;
                let mbv = scheme.meta_bytes_per_vector(role) as u64;
                let last_payload =
                    payload_offset(scheme, bc - 1, ctx - 1, kvh, hd, ctx) + kvh as u64 * pbv;
                assert_eq!(
                    last_payload,
                    meta_base(scheme, bc, ctx, kvh, hd),
                    "{}",
                    scheme.name()
                );
                let last_meta =
                    meta_offset(scheme, role, bc, bc - 1, ctx - 1, kvh, hd, ctx) + kvh as u64 * mbv;
                assert_eq!(
                    last_meta,
                    buffer_bytes(scheme, role, bc, ctx, kvh, hd),
                    "{} {:?}",
                    scheme.name(),
                    role
                );
            }
        }
    }

    // Quantized sizing includes metadata and stays under the F16 ceiling the
    // pre-flight uses.
    #[test]
    fn quantized_sizing_includes_metadata_and_fits_the_f16_ceiling() {
        let (bc, ctx, kvh, hd) = (28usize, 4096usize, 8usize, 128usize);
        let f16 = buffer_bytes(KvQuant::F16, KvRole::Key, bc, ctx, kvh, hd);
        let int8 = buffer_bytes(KvQuant::Int8, KvRole::Key, bc, ctx, kvh, hd);
        let payload_only = (bc * ctx * kvh * 128) as u64;
        assert!(int8 > payload_only, "metadata missing from the sizing");
        assert!(int8 <= f16, "pre-flight ceiling violated");
    }

    #[test]
    fn check_pos_rejects_only_at_or_past_context() {
        assert!(check_pos(4095, 4096).is_ok());
        let err = check_pos(4096, 4096).unwrap_err().to_string();
        assert_eq!(err, "kv_cache: position 4096 beyond context length 4096");
    }
}
