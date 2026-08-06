/*
 * graph_horizon_engine — CPU KV-cache write kernel
 * Copies F16 K/V vectors or delegates INT8 quantize-on-write using byte offsets
 * precomputed by the cache layout; it owns no attention calculation or storage.
 */

// AGENTS deroga K: famiglia per-schema della sola operazione KV-cache write.

use crate::backend::cpu::buffer::CpuBuffer;
use crate::kv_cache::Kv;
use crate::kv_cache::scheme::KvQuant;

use super::write_q;

#[allow(clippy::too_many_arguments)]
pub(crate) fn kv_write(
    kv: &Kv<CpuBuffer>,
    k: &CpuBuffer,
    v: &CpuBuffer,
    k_payload_offset: u64,
    v_payload_offset: u64,
    k_meta_offset: u64,
    v_meta_offset: u64,
    vectors: usize,
) {
    match kv.scheme {
        KvQuant::F16 => kv_write_f16(
            &kv.k,
            &kv.v,
            k,
            v,
            k_payload_offset as usize / 2,
            v_payload_offset as usize / 2,
            vectors * kv.head_dim,
            vectors * kv.value_dim,
        ),
        KvQuant::Int8 => write_q::kv_write_int8(
            kv,
            k,
            v,
            k_payload_offset,
            v_payload_offset,
            k_meta_offset,
            v_meta_offset,
            vectors,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn kv_write_f16(
    k_cache: &CpuBuffer,
    v_cache: &CpuBuffer,
    k: &CpuBuffer,
    v: &CpuBuffer,
    k_offset: usize,
    v_offset: usize,
    k_count: usize,
    v_count: usize,
) {
    k_cache.copy_f16_from(k, k_offset, k_count);
    v_cache.copy_f16_from(v, v_offset, v_count);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::CpuFormat;

    fn buffer(values: &[f32]) -> CpuBuffer {
        let buffer = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buffer.write_f16_from_f32(values);
        buffer
    }

    #[test]
    fn f16_write_copies_selected_ranges() {
        let k_cache = CpuBuffer::zeroed(4 * 2, CpuFormat::F16);
        let v_cache = CpuBuffer::zeroed(4 * 2, CpuFormat::F16);
        let k = buffer(&[1.0, 2.0]);
        let v = buffer(&[3.0, 4.0]);

        kv_write_f16(&k_cache, &v_cache, &k, &v, 2, 2, 2, 2);

        assert_eq!(k_cache.read_f16_as_f32(), vec![0.0, 0.0, 1.0, 2.0]);
        assert_eq!(v_cache.read_f16_as_f32(), vec![0.0, 0.0, 3.0, 4.0]);
    }
}
