/*
 * graph_orizon_engine — CPU quantize-on-write KV kernel family
 * Per-scheme variants of the single `kv_write` op: for each (token, kv_head)
 * vector of the incoming K and V (f16, widened to f32), quantize with the
 * scheme's NORMATIVE scalar reference from `kv_cache` and store payload and
 * metadata at their region offsets (both precomputed by kv_cache from `layout`,
 * D6). Prefill N-token writes iterate vectors through this same code path as
 * single-token decode writes, so `--seq-check` payloads match bit-for-bit.
*/

// AGENTS deroga K: famiglia per-schema della sola operazione kv_write (quantizza e scrive), nessun dispatch cross-operazione né I/O.

use crate::backend::cpu::buffer::CpuBuffer;
use crate::kv_cache::Kv;
use crate::kv_cache::int8;

// INT8 per-token asymmetric write: `vectors` vectors of `kv.head_dim` values
// from `k` and `v` land at byte `payload_offset` (u8 codes) and `meta_offset`
// (min,scale f16 pairs) of the K and V caches (same region layout on both).
#[allow(clippy::too_many_arguments)]
pub(super) fn kv_write_int8(
    kv: &Kv<CpuBuffer>,
    k: &CpuBuffer,
    v: &CpuBuffer,
    k_payload_offset: u64,
    v_payload_offset: u64,
    k_meta_offset: u64,
    v_meta_offset: u64,
    vectors: usize,
) {
    let kf = k.read_f16_as_f32();
    let vf = v.read_f16_as_f32();
    write_vectors(
        &kv.k,
        &kf,
        kv.head_dim,
        k_payload_offset as usize,
        k_meta_offset as usize,
        vectors,
    );
    write_vectors(
        &kv.v,
        &vf,
        kv.value_dim,
        v_payload_offset as usize,
        v_meta_offset as usize,
        vectors,
    );
}

// Quantizes `vectors` consecutive head_dim-sized groups of `x` into the cache
// buffer: payload region (1 byte/value) then metadata region (4 bytes/vector).
fn write_vectors(
    cache: &CpuBuffer,
    x: &[f32],
    head_dim: usize,
    payload_offset: usize,
    meta_offset: usize,
    vectors: usize,
) {
    debug_assert!(x.len() >= vectors * head_dim);
    cache.with_bytes_mut(|bytes| {
        for i in 0..vectors {
            let group = &x[i * head_dim..(i + 1) * head_dim];
            let p = payload_offset + i * head_dim;
            let meta = int8::quantize_group(group, &mut bytes[p..p + head_dim]);
            let m = meta_offset + i * 4;
            bytes[m..m + 4].copy_from_slice(&meta);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::CpuFormat;
    use crate::kv_cache::layout;
    use crate::kv_cache::scheme::KvQuant;

    fn f16_buf(values: &[f32]) -> CpuBuffer {
        let buf = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buf.write_f16_from_f32(values);
        buf
    }

    // The write lands payload and metadata at the layout offsets: quantizing a
    // token at (layer 1, pos 2) must leave every other region byte zero and
    // reproduce the scalar reference bytes exactly.
    #[test]
    fn int8_write_places_payload_and_meta_at_region_offsets() {
        let (bc, ctx, kvh, hd) = (2usize, 4usize, 2usize, 8usize);
        let scheme = KvQuant::Int8;
        let bytes = layout::buffer_bytes(
            scheme,
            crate::kv_cache::scheme::KvRole::Key,
            bc,
            ctx,
            kvh,
            hd,
        ) as usize;
        let kv = Kv {
            k: CpuBuffer::zeroed(bytes, CpuFormat::F16),
            v: CpuBuffer::zeroed(bytes, CpuFormat::F16),
            scheme,
            block_count: bc,
            context: ctx,
            kv_heads: kvh,
            head_dim: hd,
            value_dim: hd,
        };
        let vals: Vec<f32> = (0..kvh * hd).map(|i| i as f32 * 0.25 - 1.0).collect();
        let (layer, pos) = (1usize, 2usize);
        let po = layout::payload_offset(scheme, layer, pos, kvh, hd, ctx);
        let mo = layout::meta_offset(
            scheme,
            crate::kv_cache::scheme::KvRole::Key,
            bc,
            layer,
            pos,
            kvh,
            hd,
            ctx,
        );
        kv_write_int8(&kv, &f16_buf(&vals), &f16_buf(&vals), po, po, mo, mo, kvh);

        let got = kv.k.bytes().clone();
        // Reference bytes straight from the normative scalar quantizer. The
        // incoming buffer stores f16, so reference groups widen the same way.
        let widened = f16_buf(&vals).read_f16_as_f32();
        for i in 0..kvh {
            let mut want = vec![0u8; hd];
            let meta = int8::quantize_group(&widened[i * hd..(i + 1) * hd], &mut want);
            let p = po as usize + i * hd;
            assert_eq!(&got[p..p + hd], &want[..], "payload vector {i}");
            let m = mo as usize + i * 4;
            assert_eq!(&got[m..m + 4], &meta, "metadata vector {i}");
        }
        // Everything outside the written windows stays zero (no overlap).
        for (idx, &b) in got.iter().enumerate() {
            let in_payload = idx >= po as usize && idx < po as usize + kvh * hd;
            let in_meta = idx >= mo as usize && idx < mo as usize + kvh * 4;
            if !in_payload && !in_meta {
                assert_eq!(b, 0, "stray byte at {idx}");
            }
        }
    }
}
