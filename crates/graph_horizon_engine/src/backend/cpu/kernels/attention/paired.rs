// Owns temporary paired-query output/scores and their token-major writeback.
// Eligibility is checked by attention_prefill_f16; the numeric kernel owns
// no resources. Each worker receives disjoint (query_pair, kv_head) units.
use crate::backend::cpu::buffer::{CpuBuffer, narrow_f32_to_f16};
use crate::backend::cpu::parallel;

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill(
    out: &CpuBuffer,
    q: &CpuBuffer,
    k_cache: &CpuBuffer,
    v_cache: &CpuBuffer,
    key_dim: usize,
    value_dim: usize,
    kv_heads: usize,
    base: usize,
    n: usize,
    layer: usize,
    context: usize,
) {
    let q_heads = kv_heads * 4;
    let q = q.read_f16_as_f32();
    let kc = k_cache.bytes();
    let vc = v_cache.bytes();
    let kc = &kc[k_cache.window()];
    let vc = &vc[v_cache.window()];
    let unit = 8 * value_dim;
    let mut values = vec![0f32; n * q_heads * value_dim];
    parallel::for_units(&mut values, unit, |u0, chunk| {
        let units = chunk.len() / unit;
        let max_len = base + 2 * ((u0 + units - 1) / kv_heads) + 2;
        let mut scores = vec![0f32; 8 * max_len];
        for j in 0..units {
            let u = u0 + j;
            let pair = u / kv_heads;
            let kvh = u % kv_heads;
            let common = base + 2 * pair + 1;
            let queries = std::array::from_fn(|h| {
                let row = 2 * pair + h / 4;
                let head = kvh * 4 + h % 4;
                let start = (row * q_heads + head) * key_dim;
                &q[start..start + key_dim]
            });
            let kstart = (layer * context * kv_heads + kvh) * key_dim;
            let vstart = (layer * context * kv_heads + kvh) * value_dim;
            // SAFETY: caller checked ISA and positive eight-aligned dimensions.
            // Validated graph/cache windows contain both query positions. The
            // kernel touches common+1 rows at these head-strided cache origins;
            // outputs and eight score segments are complete, disjoint slices.
            unsafe {
                super::simd::attend_positions_avx2(
                    queries,
                    &kc[kstart * 2..],
                    &vc[vstart * 2..],
                    kv_heads * key_dim,
                    kv_heads * value_dim,
                    common,
                    &mut scores[..8 * (common + 1)],
                    &mut chunk[j * unit..(j + 1) * unit],
                );
            }
        }
    });
    // Reorder only independent outputs; narrowing uses the existing FP16 rule.
    out.with_bytes_mut(|dst| {
        for (u, group) in values.chunks_exact(unit).enumerate() {
            let pair = u / kv_heads;
            let kvh = u % kv_heads;
            for row in 0..2 {
                let start = ((2 * pair + row) * q_heads + kvh * 4) * value_dim;
                narrow_f32_to_f16(
                    &group[row * 4 * value_dim..(row + 1) * 4 * value_dim],
                    &mut dst[start * 2..(start + 4 * value_dim) * 2],
                );
            }
        }
    });
}
