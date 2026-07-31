/*
 * gh_zero_engine — CPU dequantize-on-read attention kernel family
 * Per-scheme attention variants: the same three-pass causal GQA structure as
 * the f16 `attend` in the parent module (score → stabilized softmax → V mix,
 * f32 accumulation, fixed pass order), but the K-dot and V-axpy inner loops
 * read the quantized codes and dequantize on the fly with the group's
 * per-(token, kv_head) metadata. Scalar today, so the `no_attn_simd` toggle is
 * trivially honored. Output distribution over the cores mirrors the f16 kernels
 * ((row, head) units via `parallel::for_units`,
 * disjoint head_dim slices), so parallel output is bit-identical to serial.
*/

// AGENTS deroga K: famiglia per-schema della sola operazione attention (decode/prefill dequantize-on-read), nessun dispatch cross-operazione né I/O.

use crate::backend::cpu::buffer::CpuBuffer;
use crate::backend::cpu::parallel;
use crate::kv_cache::Kv;
use crate::kv_cache::int8;

#[allow(clippy::too_many_arguments)]
fn attend_int8_dims(
    q_head: &[f32],
    kc: &[u8],
    vc: &[u8],
    k_meta_base: usize,
    v_meta_base: usize,
    key_dim: usize,
    value_dim: usize,
    kv_heads: usize,
    kvh: usize,
    layer: usize,
    context: usize,
    scale: f32,
    out_head: &mut [f32],
    scores: &mut [f32],
) {
    // Pass 1: per-position scores (f32 dot with on-the-fly dequant) and max.
    let mut m = f32::NEG_INFINITY;
    for (t, score) in scores.iter_mut().enumerate() {
        let vec_idx = (layer * context + t) * kv_heads + kvh;
        let pbase = vec_idx * key_dim;
        let meta = k_meta_base + vec_idx * 4;
        let (mn, sc) = int8::meta_decode(&kc[meta..meta + 4]);
        let mut s = 0f32;
        for (d, &qd) in q_head.iter().enumerate() {
            s += qd * int8::dequant(kc[pbase + d], mn, sc);
        }
        let s = s * scale;
        *score = s;
        m = m.max(s);
    }

    // Pass 2: stabilized exp() and softmax denominator.
    let mut denom = 0f32;
    for score in scores.iter_mut() {
        *score = (*score - m).exp();
        denom += *score;
    }

    // Pass 3: softmax-weighted sum of the dequantized V vectors.
    for (t, &e) in scores.iter().enumerate() {
        let w = e / denom;
        let vec_idx = (layer * context + t) * kv_heads + kvh;
        let pbase = vec_idx * value_dim;
        let meta = v_meta_base + vec_idx * 4;
        let (mn, sc) = int8::meta_decode(&vc[meta..meta + 4]);
        for (d, o) in out_head.iter_mut().enumerate() {
            *o += w * int8::dequant(vc[pbase + d], mn, sc);
        }
    }
}

// out[q_heads*head_dim] = causal GQA int8 attention of the current token over
// cached positions 0..=pos. Parallel over query heads (disjoint output slices).
pub(super) fn attention_decode_int8(
    out: &CpuBuffer,
    q: &CpuBuffer,
    kv: &Kv<CpuBuffer>,
    q_heads: usize,
    pos: usize,
    layer: usize,
) {
    let q = q.read_f16_as_f32();
    let kc = kv.k.bytes();
    let vc = kv.v.bytes();
    let kc: &[u8] = &kc[kv.k.window()];
    let vc: &[u8] = &vc[kv.v.window()];
    let key_dim = kv.head_dim;
    let value_dim = kv.value_dim;
    let k_meta_base = kv.meta_base_for(crate::kv_cache::scheme::KvRole::Key) as usize;
    let v_meta_base = kv.meta_base_for(crate::kv_cache::scheme::KvRole::Value) as usize;
    let scale = 1.0 / (key_dim as f32).sqrt();
    let group = q_heads / kv.kv_heads; // GQA ratio

    let mut o = vec![0f32; q_heads * value_dim];
    parallel::for_units(&mut o, value_dim, |h0, chunk| {
        let mut scores = vec![0f32; pos + 1]; // per-thread scratch
        for j in 0..chunk.len() / value_dim {
            let h = h0 + j; // absolute query head
            attend_int8_dims(
                &q[h * key_dim..(h + 1) * key_dim],
                kc,
                vc,
                k_meta_base,
                v_meta_base,
                key_dim,
                value_dim,
                kv.kv_heads,
                h / group,
                layer,
                kv.context,
                scale,
                &mut chunk[j * value_dim..(j + 1) * value_dim],
                &mut scores,
            );
        }
    });
    out.write_f16_from_f32(&o);
}

// out[n*q_heads*head_dim] = causal GQA int8 attention for N query rows in one
// call: row i (absolute position base+i) attends 0..=base+i. Unit u of the
// parallel split maps to row u/q_heads and head u%q_heads — same mapping as
// the f16 prefill, so n == 1, base == pos degenerates to the decode kernel
// through the SAME `attend_int8` (bit-for-bit).
#[allow(clippy::too_many_arguments)]
pub(super) fn attention_prefill_int8(
    out: &CpuBuffer,
    q: &CpuBuffer,
    kv: &Kv<CpuBuffer>,
    q_heads: usize,
    base: usize,
    n: usize,
    layer: usize,
) {
    let q = q.read_f16_as_f32();
    let kc = kv.k.bytes();
    let vc = kv.v.bytes();
    let kc: &[u8] = &kc[kv.k.window()];
    let vc: &[u8] = &vc[kv.v.window()];
    let key_dim = kv.head_dim;
    let value_dim = kv.value_dim;
    let k_meta_base = kv.meta_base_for(crate::kv_cache::scheme::KvRole::Key) as usize;
    let v_meta_base = kv.meta_base_for(crate::kv_cache::scheme::KvRole::Value) as usize;
    let scale = 1.0 / (key_dim as f32).sqrt();
    let group = q_heads / kv.kv_heads;

    let mut o = vec![0f32; n * q_heads * value_dim];
    parallel::for_units(&mut o, value_dim, |u0, chunk| {
        let units = chunk.len() / value_dim;
        if units == 0 {
            return;
        }
        // Rows grow with the unit index: the chunk's last unit bounds the
        // scratch (Pass 1 overwrites every entry it touches).
        let max_pos = base + (u0 + units - 1) / q_heads;
        let mut scores = vec![0f32; max_pos + 1];
        for j in 0..units {
            let ui = u0 + j;
            let i = ui / q_heads; // query row
            let h = ui % q_heads; // query head
            let pos = base + i;
            attend_int8_dims(
                &q[(i * q_heads + h) * key_dim..(i * q_heads + h + 1) * key_dim],
                kc,
                vc,
                k_meta_base,
                v_meta_base,
                key_dim,
                value_dim,
                kv.kv_heads,
                h / group,
                layer,
                kv.context,
                scale,
                &mut chunk[j * value_dim..(j + 1) * value_dim],
                &mut scores[..pos + 1],
            );
        }
    });
    out.write_f16_from_f32(&o);
}

#[cfg(test)]
mod tests {
    use super::super::write_q::kv_write_int8;
    use super::*;
    use crate::backend::cpu::buffer::CpuFormat;
    use crate::kv_cache::layout;
    use crate::kv_cache::scheme::KvQuant;

    fn f16_buf(values: &[f32]) -> CpuBuffer {
        let buf = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buf.write_f16_from_f32(values);
        buf
    }

    // Builds an int8 KV filled through the write kernel, one token at a time
    // (the decode path), over deterministic values.
    fn int8_kv(
        bc: usize,
        ctx: usize,
        kvh: usize,
        hd: usize,
        tokens: usize,
        layer: usize,
    ) -> Kv<CpuBuffer> {
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
        for pos in 0..tokens {
            let kvals: Vec<f32> = (0..kvh * hd)
                .map(|i| ((i + pos * 3) % 7) as f32 * 0.2 - 0.6)
                .collect();
            let vvals: Vec<f32> = (0..kvh * hd)
                .map(|i| ((i + pos * 5) % 9) as f32 * 0.15 - 0.4)
                .collect();
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
            kv_write_int8(&kv, &f16_buf(&kvals), &f16_buf(&vvals), po, po, mo, mo, kvh);
        }
        kv
    }

    // The int8 attention must match a plain serial reference computed over the
    // DEQUANTIZED cache (same three passes) exactly: same arithmetic, same
    // order, only the storage differs.
    #[test]
    fn decode_matches_dequantized_serial_reference() {
        let (bc, ctx, kvh, hd) = (2usize, 6usize, 2usize, 4usize);
        let (q_heads, pos, layer) = (4usize, 5usize, 1usize);
        let kv = int8_kv(bc, ctx, kvh, hd, pos + 1, layer);
        let q_vals: Vec<f32> = (0..q_heads * hd)
            .map(|i| (i % 11) as f32 * 0.1 - 0.5)
            .collect();
        let q = f16_buf(&q_vals);
        let out = CpuBuffer::zeroed(q_heads * hd * 2, CpuFormat::F16);
        attention_decode_int8(&out, &q, &kv, q_heads, pos, layer);

        // Serial reference over dequantized values.
        let qw = q.read_f16_as_f32();
        let kcg = kv.k.bytes();
        let vcg = kv.v.bytes();
        let meta_base = kv.meta_base() as usize;
        let scale = 1.0 / (hd as f32).sqrt();
        let group = q_heads / kvh;
        let mut expected = vec![0f32; q_heads * hd];
        for h in 0..q_heads {
            let kvhh = h / group;
            let mut scores = vec![0f32; pos + 1];
            let mut m = f32::NEG_INFINITY;
            for (t, sc) in scores.iter_mut().enumerate() {
                let vi = (layer * ctx + t) * kvh + kvhh;
                let (mn, s16) = int8::meta_decode(&kcg[meta_base + vi * 4..meta_base + vi * 4 + 4]);
                let mut s = 0f32;
                for d in 0..hd {
                    s += qw[h * hd + d] * int8::dequant(kcg[vi * hd + d], mn, s16);
                }
                *sc = s * scale;
                m = m.max(*sc);
            }
            let mut denom = 0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            for (t, &e) in scores.iter().enumerate() {
                let vi = (layer * ctx + t) * kvh + kvhh;
                let (mn, s16) = int8::meta_decode(&vcg[meta_base + vi * 4..meta_base + vi * 4 + 4]);
                for d in 0..hd {
                    expected[h * hd + d] += e / denom * int8::dequant(vcg[vi * hd + d], mn, s16);
                }
            }
        }
        let ref_buf = CpuBuffer::zeroed(q_heads * hd * 2, CpuFormat::F16);
        ref_buf.write_f16_from_f32(&expected);
        assert_eq!(out.read_f16_as_f32(), ref_buf.read_f16_as_f32());
    }

    // Bridge invariant: a single-query prefill (n = 1, base = pos) produces
    // exactly the decode output (same `attend_int8`, bit-for-bit).
    #[test]
    fn prefill_n1_matches_decode() {
        let (bc, ctx, kvh, hd) = (2usize, 8usize, 2usize, 4usize);
        let (q_heads, pos, layer) = (4usize, 3usize, 1usize);
        let kv = int8_kv(bc, ctx, kvh, hd, pos + 1, layer);
        let q_vals: Vec<f32> = (0..q_heads * hd)
            .map(|i| (i % 11) as f32 * 0.1 - 0.5)
            .collect();
        let q = f16_buf(&q_vals);
        let dec = CpuBuffer::zeroed(q_heads * hd * 2, CpuFormat::F16);
        attention_decode_int8(&dec, &q, &kv, q_heads, pos, layer);
        let pre = CpuBuffer::zeroed(q_heads * hd * 2, CpuFormat::F16);
        attention_prefill_int8(&pre, &q, &kv, q_heads, pos, 1, layer);
        assert_eq!(dec.read_f16_as_f32(), pre.read_f16_as_f32());
    }

    // Context boundary: attending at pos = context - 1 stays in bounds and
    // yields finite output (the last vector touches exactly the region end).
    #[test]
    fn decode_at_context_boundary_is_finite() {
        let (bc, ctx, kvh, hd) = (1usize, 4usize, 1usize, 4usize);
        let kv = int8_kv(bc, ctx, kvh, hd, ctx, 0);
        let q = f16_buf(&[0.3, -0.2, 0.5, 0.1]);
        let out = CpuBuffer::zeroed(hd * 2, CpuFormat::F16);
        attention_decode_int8(&out, &q, &kv, 1, ctx - 1, 0);
        assert!(out.read_f16_as_f32().iter().all(|v| v.is_finite()));
    }
}
