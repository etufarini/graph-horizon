/*
 * graph_horizon_engine — CPU attention kernels
 * Scalar transcription of attention_decode.comp / attention_prefill.comp. The
 * public entry points branch once per call on the runtime KV scheme; quantized
 * dequantize-on-read variants live in `read_q`.
 * `attention_decode` runs causal GQA attention for the current token;
 * `attention_prefill` is its N-query generalization in one call (row i is the
 * query at absolute position base+i). Both delegate the per-(row,head) work to the
 * shared `attend` helper: query head h maps to kv head h / (q_heads/kv_heads);
 * Q/K/V are FP16 and the dot products / softmax / value mix accumulate in f32; the
 * output is written back in FP16. The shader fuses softmax via the online (flash)
 * algorithm; here a numerically equivalent two-pass form is used (subtract the max
 * before exp), which matches within FP16 tolerance. The position is already
 * validated upstream, so these kernels cannot fail (they return ()).
 * Output is distributed across the cores via `parallel::for_units` over (kv_head)
 * units for decode and (row, kv_head) units for prefill (stride `group*head_dim`):
 * the `group = q_heads/kv_heads` query heads sharing a kv head are processed in
 * groups of four, then pair/single tails, so each cached K/V row is widened once per
 * largest supported group. Each head's three passes stay in fixed order, so
 * the output is bit-identical to the per-head single-thread path. The `scores` scratch
 * is per-thread. KV writes are owned separately by `write`/`write_q`.
*/

// AGENTS deroga K: kernel attention denso decode/prefill, nessun dispatch cross-operazione né I/O.

// AVX2+F16C inner kernels, used internally by the dispatch above.
mod simd;
// Per-scheme quantize-on-write / dequantize-on-read variants of the same op family.
mod read_q;
mod write;
mod write_q;

pub(crate) use write::kv_write;

use crate::backend::cpu::buffer::{CpuBuffer, f16_to_f32};
use crate::backend::cpu::parallel;
use crate::kv_cache::Kv;
use crate::kv_cache::scheme::KvQuant;

// Widens one FP16 cache element (element index `i`) to f32, reading just its two
// bytes. Used to index the K/V caches at the positions actually attended (0..=pos
// for one layer), instead of materializing the whole multi-layer cache.
#[inline]
fn elem(bytes: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]))
}

use std::sync::OnceLock;

// True when the host has the AVX2+FMA+F16C features the SIMD `attend` inner kernels
// need. The result is cached and passed down per `attention_*` call, so unsupported
// hosts use the scalar fallback without repeating feature detection per key.
fn simd_supported() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && is_x86_feature_detected!("f16c")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

// Dot product of an f32 query head with the FP16 key row at element offset `kbase`.
// Dispatches to the AVX2+F16C kernel when `simd`, else the scalar reference.
#[inline]
fn dot_f16(simd: bool, q: &[f32], kc: &[u8], kbase: usize, head_dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only true when avx2+fma+f16c were runtime-detected.
        return unsafe { self::simd::dot_f16_avx2(q, kc, kbase, head_dim) };
    }
    let _ = simd;
    (0..head_dim).map(|d| q[d] * elem(kc, kbase + d)).sum()
}

// Two-query GQA-group dot: dots one key row against two queries sharing the kv head,
// reusing the key's FP16→f32 widening. Bit-identical to two `dot_f16` calls; the SIMD
// path widens the key once. The scalar fallback runs the two folds independently.
#[inline]
fn dot_f16_x2(
    simd: bool,
    q0: &[f32],
    q1: &[f32],
    kc: &[u8],
    kbase: usize,
    head_dim: usize,
) -> (f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only true when avx2+fma+f16c were runtime-detected.
        return unsafe { self::simd::dot_f16_x2_avx2(q0, q1, kc, kbase, head_dim) };
    }
    let _ = simd;
    (
        (0..head_dim).map(|d| q0[d] * elem(kc, kbase + d)).sum(),
        (0..head_dim).map(|d| q1[d] * elem(kc, kbase + d)).sum(),
    )
}

// Four-query GQA dot. AVX2 widens the shared key once; portable execution
// delegates to the established single-head reference to preserve its exact
// arithmetic without adding a second scalar implementation.
#[inline]
#[allow(clippy::too_many_arguments)]
fn dot_f16_x4(
    simd: bool,
    q0: &[f32],
    q1: &[f32],
    q2: &[f32],
    q3: &[f32],
    kc: &[u8],
    kbase: usize,
    head_dim: usize,
) -> (f32, f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is true only after AVX2+FMA+F16C detection.
        return unsafe { self::simd::dot_f16_x4_avx2(q0, q1, q2, q3, kc, kbase, head_dim) };
    }
    (
        dot_f16(false, q0, kc, kbase, head_dim),
        dot_f16(false, q1, kc, kbase, head_dim),
        dot_f16(false, q2, kc, kbase, head_dim),
        dot_f16(false, q3, kc, kbase, head_dim),
    )
}

// Two-output GQA-group axpy: accumulates one value row into two outputs, reusing the
// value's widening. Bit-identical to two `axpy_f16` calls.
#[inline]
#[allow(clippy::too_many_arguments)]
fn axpy_f16_x2(
    simd: bool,
    out0: &mut [f32],
    out1: &mut [f32],
    w0: f32,
    w1: f32,
    vc: &[u8],
    vbase: usize,
    head_dim: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only true when avx2+fma+f16c were runtime-detected.
        return unsafe { self::simd::axpy_f16_x2_avx2(out0, out1, w0, w1, vc, vbase, head_dim) };
    }
    let _ = (simd, head_dim);
    for (d, (o0, o1)) in out0.iter_mut().zip(out1.iter_mut()).enumerate() {
        let vf = elem(vc, vbase + d);
        *o0 += w0 * vf;
        *o1 += w1 * vf;
    }
}

// Four-output GQA value mix, with one shared FP16 widening on AVX2 and the
// scalar operation as the portable fallback.
#[inline]
#[allow(clippy::too_many_arguments)]
fn axpy_f16_x4(
    simd: bool,
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
    vc: &[u8],
    vbase: usize,
    head_dim: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is true only after AVX2+FMA+F16C detection.
        return unsafe {
            self::simd::axpy_f16_x4_avx2(
                out0, out1, out2, out3, w0, w1, w2, w3, vc, vbase, head_dim,
            )
        };
    }
    axpy_f16(false, out0, w0, vc, vbase, head_dim);
    axpy_f16(false, out1, w1, vc, vbase, head_dim);
    axpy_f16(false, out2, w2, vc, vbase, head_dim);
    axpy_f16(false, out3, w3, vc, vbase, head_dim);
}

// Scaled accumulate of the FP16 value row at element offset `vbase` into `out`:
// out[d] += w·f16(v[d]). Dispatches to AVX2+F16C when `simd`, else scalar.
#[inline]
fn axpy_f16(simd: bool, out: &mut [f32], w: f32, vc: &[u8], vbase: usize, head_dim: usize) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only true when avx2+fma+f16c were runtime-detected.
        return unsafe { self::simd::axpy_f16_avx2(out, w, vc, vbase, head_dim) };
    }
    let _ = (simd, head_dim);
    for (d, o) in out.iter_mut().enumerate() {
        *o += w * elem(vc, vbase + d);
    }
}

// Three-pass causal attention for one query head at absolute position `pos`:
// score → stabilized softmax → V mix, accumulating in f32 into `out_head`
// (head_dim outputs, pre-zeroed). `q_head` is the head's head_dim FP16-widened
// query; `kc`/`vc` are the layer-windowed cache bytes; `kvh` is the GQA-mapped kv
// head. `scores` is caller-owned per-thread scratch whose length `pos + 1` encodes
// the attended range 0..=pos. The fixed pass order keeps the result identical to
// the single-thread reference; this is the shared primitive behind both
// `attention_decode` and `attention_prefill`.
#[allow(clippy::too_many_arguments)]
fn attend(
    q_head: &[f32],
    kc: &[u8],
    vc: &[u8],
    key_dim: usize,
    value_dim: usize,
    kv_heads: usize,
    kvh: usize,
    layer: usize,
    context: usize,
    scale: f32,
    out_head: &mut [f32],
    scores: &mut [f32],
    simd: bool,
) {
    // Pass 1: per-position scores (f32 dot product, scaled) and their max, tracked
    // so the softmax can subtract it for numerical stability.
    let mut m = f32::NEG_INFINITY;
    for (t, score) in scores.iter_mut().enumerate() {
        let kbase = ((layer * context + t) * kv_heads + kvh) * key_dim;
        let s = dot_f16(simd, q_head, kc, kbase, key_dim) * scale;
        *score = s;
        m = m.max(s);
    }

    // Pass 2: stabilized exp() and softmax denominator (max already removed).
    let mut denom = 0f32;
    for score in scores.iter_mut() {
        *score = (*score - m).exp();
        denom += *score;
    }

    // Pass 3: softmax-weighted sum of the cached V vectors.
    for (t, &e) in scores.iter().enumerate() {
        let w = e / denom;
        let vbase = ((layer * context + t) * kv_heads + kvh) * value_dim;
        axpy_f16(simd, out_head, w, vc, vbase, value_dim);
    }
}

// Two-head GQA-group variant of `attend`: runs the three passes for two query heads
// that share the same kv head `kvh`, so each cached K/V row is read and widened ONCE
// and reused for both heads (the inner-loop cost is the FP16→f32 widening). Each head
// keeps its own scores/softmax, and the per-head arithmetic order is unchanged, so
// `out0`/`out1` are bit-identical to two separate `attend` calls — only the shared
// K/V read is amortized. `q0`/`q1` are the two heads' queries; `scores0`/`scores1`
// are caller-owned per-thread scratch of length `pos+1`.
#[allow(clippy::too_many_arguments)]
fn attend_pair(
    q0: &[f32],
    q1: &[f32],
    kc: &[u8],
    vc: &[u8],
    key_dim: usize,
    value_dim: usize,
    kv_heads: usize,
    kvh: usize,
    layer: usize,
    context: usize,
    scale: f32,
    out0: &mut [f32],
    out1: &mut [f32],
    scores0: &mut [f32],
    scores1: &mut [f32],
    simd: bool,
) {
    // Pass 1: shared key read per position, one score per head.
    let (mut m0, mut m1) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for t in 0..scores0.len() {
        let kbase = ((layer * context + t) * kv_heads + kvh) * key_dim;
        let (s0, s1) = dot_f16_x2(simd, q0, q1, kc, kbase, key_dim);
        let (s0, s1) = (s0 * scale, s1 * scale);
        scores0[t] = s0;
        scores1[t] = s1;
        m0 = m0.max(s0);
        m1 = m1.max(s1);
    }
    // Pass 2: stabilized exp() and per-head denominators.
    let (mut d0, mut d1) = (0f32, 0f32);
    for (e0, e1) in scores0.iter_mut().zip(scores1.iter_mut()) {
        *e0 = (*e0 - m0).exp();
        *e1 = (*e1 - m1).exp();
        d0 += *e0;
        d1 += *e1;
    }
    // Pass 3: shared value read per position, mixed into both heads with their weights.
    for t in 0..scores0.len() {
        let vbase = ((layer * context + t) * kv_heads + kvh) * value_dim;
        axpy_f16_x2(
            simd,
            out0,
            out1,
            scores0[t] / d0,
            scores1[t] / d1,
            vc,
            vbase,
            value_dim,
        );
    }
}

// Four-head GQA variant for the Ministral group size: each cached K/V row is
// loaded and widened once for all four queries instead of once per pair. Score
// buffers and denominators remain independent, preserving causal softmax state.
#[allow(clippy::too_many_arguments)]
fn attend_quad(
    q0: &[f32],
    q1: &[f32],
    q2: &[f32],
    q3: &[f32],
    kc: &[u8],
    vc: &[u8],
    key_dim: usize,
    value_dim: usize,
    kv_heads: usize,
    kvh: usize,
    layer: usize,
    context: usize,
    scale: f32,
    out0: &mut [f32],
    out1: &mut [f32],
    out2: &mut [f32],
    out3: &mut [f32],
    scores0: &mut [f32],
    scores1: &mut [f32],
    scores2: &mut [f32],
    scores3: &mut [f32],
    simd: bool,
) {
    let (mut m0, mut m1, mut m2, mut m3) = (
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    for t in 0..scores0.len() {
        let kbase = ((layer * context + t) * kv_heads + kvh) * key_dim;
        let (s0, s1, s2, s3) = dot_f16_x4(simd, q0, q1, q2, q3, kc, kbase, key_dim);
        let (s0, s1, s2, s3) = (s0 * scale, s1 * scale, s2 * scale, s3 * scale);
        scores0[t] = s0;
        scores1[t] = s1;
        scores2[t] = s2;
        scores3[t] = s3;
        m0 = m0.max(s0);
        m1 = m1.max(s1);
        m2 = m2.max(s2);
        m3 = m3.max(s3);
    }
    let (mut d0, mut d1, mut d2, mut d3) = (0f32, 0f32, 0f32, 0f32);
    for t in 0..scores0.len() {
        scores0[t] = (scores0[t] - m0).exp();
        scores1[t] = (scores1[t] - m1).exp();
        scores2[t] = (scores2[t] - m2).exp();
        scores3[t] = (scores3[t] - m3).exp();
        d0 += scores0[t];
        d1 += scores1[t];
        d2 += scores2[t];
        d3 += scores3[t];
    }
    for t in 0..scores0.len() {
        let vbase = ((layer * context + t) * kv_heads + kvh) * value_dim;
        axpy_f16_x4(
            simd,
            out0,
            out1,
            out2,
            out3,
            scores0[t] / d0,
            scores1[t] / d1,
            scores2[t] / d2,
            scores3[t] / d3,
            vc,
            vbase,
            value_dim,
        );
    }
}

// Scheme dispatch for decode attention: one match per call. The f16 arm stays
// local while the quantized arms use the `read_q` variants.
pub(crate) fn attention_decode(
    out: &CpuBuffer,
    q: &CpuBuffer,
    kv: &Kv<CpuBuffer>,
    q_heads: usize,
    pos: usize,
    layer: usize,
) {
    match kv.scheme {
        KvQuant::F16 => attention_decode_f16(
            out,
            q,
            &kv.k,
            &kv.v,
            kv.head_dim,
            kv.value_dim,
            kv.kv_heads,
            q_heads,
            pos,
            layer,
            kv.context,
        ),
        KvQuant::Int8 => read_q::attention_decode_int8(out, q, kv, q_heads, pos, layer),
    }
}

// Scheme dispatch for the batched prefill attention (same rule as decode).
pub(crate) fn attention_prefill(
    out: &CpuBuffer,
    q: &CpuBuffer,
    kv: &Kv<CpuBuffer>,
    q_heads: usize,
    base: usize,
    n: usize,
    layer: usize,
) {
    match kv.scheme {
        KvQuant::F16 => attention_prefill_f16(
            out,
            q,
            &kv.k,
            &kv.v,
            kv.head_dim,
            kv.value_dim,
            kv.kv_heads,
            q_heads,
            base,
            n,
            layer,
            kv.context,
        ),
        KvQuant::Int8 => read_q::attention_prefill_int8(out, q, kv, q_heads, base, n, layer),
    }
}

// out[q_heads*head_dim] = causal GQA attention of the current token over cached
// positions 0..=pos. Mirror of attention_decode.comp (two-pass softmax).
#[allow(clippy::too_many_arguments)]
fn attention_decode_f16(
    out: &CpuBuffer,
    q: &CpuBuffer,
    k_cache: &CpuBuffer,
    v_cache: &CpuBuffer,
    key_dim: usize,
    value_dim: usize,
    kv_heads: usize,
    q_heads: usize,
    pos: usize,
    layer: usize,
    context: usize,
) {
    let q = q.read_f16_as_f32();
    // The caches are read by element index (see `elem`): only the 0..=pos slice
    // of the current layer is touched, never the whole multi-layer cache. One
    // read-guard each is held by the parent; workers share the bytes read-only.
    let kc = k_cache.bytes();
    let vc = v_cache.bytes();
    let kc: &[u8] = &kc[k_cache.window()];
    let vc: &[u8] = &vc[v_cache.window()];
    let scale = 1.0 / (key_dim as f32).sqrt(); // same as the Vulkan push-constant
    let group = q_heads / kv_heads; // GQA ratio (1 when q_heads == kv_heads)

    let simd = simd_supported(); // resolved once, not in the per-position inner loop
    // Parallelize over kv heads. Groups of four share one K/V widening; smaller
    // tails retain the pair/single paths, so every valid GQA ratio is covered.
    let unit = group * value_dim;
    let mut o = vec![0f32; q_heads * value_dim];
    parallel::for_units(&mut o, unit, |k0, chunk| {
        let mut scores0 = vec![0f32; pos + 1]; // per-thread scratch, never shared
        let mut scores1 = vec![0f32; pos + 1];
        let mut scores2 = vec![0f32; pos + 1];
        let mut scores3 = vec![0f32; pos + 1];
        for j in 0..chunk.len() / unit {
            let kvh = k0 + j; // absolute kv head
            let group_out = &mut chunk[j * unit..(j + 1) * unit];
            let mut gh = 0;
            while gh + 3 < group {
                let h0 = kvh * group + gh;
                let qbase = h0 * key_dim;
                let outputs = &mut group_out[gh * value_dim..(gh + 4) * value_dim];
                let (out0, tail) = outputs.split_at_mut(value_dim);
                let (out1, tail) = tail.split_at_mut(value_dim);
                let (out2, out3) = tail.split_at_mut(value_dim);
                attend_quad(
                    &q[qbase..qbase + key_dim],
                    &q[qbase + key_dim..qbase + 2 * key_dim],
                    &q[qbase + 2 * key_dim..qbase + 3 * key_dim],
                    &q[qbase + 3 * key_dim..qbase + 4 * key_dim],
                    kc,
                    vc,
                    key_dim,
                    value_dim,
                    kv_heads,
                    kvh,
                    layer,
                    context,
                    scale,
                    out0,
                    out1,
                    out2,
                    out3,
                    &mut scores0,
                    &mut scores1,
                    &mut scores2,
                    &mut scores3,
                    simd,
                );
                gh += 4;
            }
            while gh + 1 < group {
                let h0 = kvh * group + gh;
                let (qb0, qb1) = (h0 * key_dim, (h0 + 1) * key_dim);
                let (lo, hi) = group_out.split_at_mut((gh + 1) * value_dim);
                attend_pair(
                    &q[qb0..qb0 + key_dim],
                    &q[qb1..qb1 + key_dim],
                    kc,
                    vc,
                    key_dim,
                    value_dim,
                    kv_heads,
                    kvh,
                    layer,
                    context,
                    scale,
                    &mut lo[gh * value_dim..(gh + 1) * value_dim],
                    &mut hi[..value_dim],
                    &mut scores0,
                    &mut scores1,
                    simd,
                );
                gh += 2;
            }
            if gh < group {
                let qbase = (kvh * group + gh) * key_dim;
                attend(
                    &q[qbase..qbase + key_dim],
                    kc,
                    vc,
                    key_dim,
                    value_dim,
                    kv_heads,
                    kvh,
                    layer,
                    context,
                    scale,
                    &mut group_out[gh * value_dim..(gh + 1) * value_dim],
                    &mut scores0,
                    simd,
                );
            }
        }
    });
    out.write_f16_from_f32(&o);
}

// out[n*q_heads*head_dim] = causal GQA attention for N query in one call: row i
// (absolute position base+i) attends cached positions 0..=base+i. Naive mirror of
// attention_prefill.comp. Output units are (row, head) pairs distributed over the
// cores: unit u maps to row u/q_heads and head u%q_heads, each writing a disjoint
// head_dim slice — independent, so parallel output is bit-identical to serial.
#[allow(clippy::too_many_arguments)]
fn attention_prefill_f16(
    out: &CpuBuffer,
    q: &CpuBuffer,
    k_cache: &CpuBuffer,
    v_cache: &CpuBuffer,
    key_dim: usize,
    value_dim: usize,
    kv_heads: usize,
    q_heads: usize,
    base: usize,
    n: usize,
    layer: usize,
    context: usize,
) {
    let q = q.read_f16_as_f32();
    let kc = k_cache.bytes();
    let vc = v_cache.bytes();
    let kc: &[u8] = &kc[k_cache.window()];
    let vc: &[u8] = &vc[v_cache.window()];
    let scale = 1.0 / (key_dim as f32).sqrt();
    let group = q_heads / kv_heads; // GQA ratio (1 when q_heads == kv_heads)

    let simd = simd_supported(); // resolved once, not in the O(n²) inner loop
    // Parallelize over (row, kv_head) units. Four-query groups share one K/V
    // widening; pair/single tails preserve every smaller GQA shape.
    let unit = group * value_dim;
    let mut o = vec![0f32; n * q_heads * value_dim];
    parallel::for_units(&mut o, unit, |u0, chunk| {
        let units = chunk.len() / unit;
        if units == 0 {
            return;
        }
        // Rows grow monotonically with the (row, kvh) unit index; the chunk's last
        // unit gives its max pos. One scratch per query head, sliced to `pos+1`
        // (Pass 1 overwrites every entry it touches). Contiguous (row, kvh) order keeps
        // the KV warm — same locality argument as the old (row, head) split.
        let max_pos = base + (u0 + units - 1) / kv_heads;
        let mut scores0 = vec![0f32; max_pos + 1];
        let mut scores1 = vec![0f32; max_pos + 1];
        let mut scores2 = vec![0f32; max_pos + 1];
        let mut scores3 = vec![0f32; max_pos + 1];
        for j in 0..units {
            let ui = u0 + j; // absolute (row, kvh) unit index
            let i = ui / kv_heads; // query row in the batch
            let kvh = ui % kv_heads; // kv head
            let pos = base + i; // absolute position of this row
            let group_out = &mut chunk[j * unit..(j + 1) * unit];
            // Consume the largest shared group before pair/single tails.
            let mut gh = 0;
            while gh + 3 < group {
                let h0 = kvh * group + gh;
                let qbase = (i * q_heads + h0) * key_dim;
                let outputs = &mut group_out[gh * value_dim..(gh + 4) * value_dim];
                let (out0, tail) = outputs.split_at_mut(value_dim);
                let (out1, tail) = tail.split_at_mut(value_dim);
                let (out2, out3) = tail.split_at_mut(value_dim);
                attend_quad(
                    &q[qbase..qbase + key_dim],
                    &q[qbase + key_dim..qbase + 2 * key_dim],
                    &q[qbase + 2 * key_dim..qbase + 3 * key_dim],
                    &q[qbase + 3 * key_dim..qbase + 4 * key_dim],
                    kc,
                    vc,
                    key_dim,
                    value_dim,
                    kv_heads,
                    kvh,
                    layer,
                    context,
                    scale,
                    out0,
                    out1,
                    out2,
                    out3,
                    &mut scores0[..pos + 1],
                    &mut scores1[..pos + 1],
                    &mut scores2[..pos + 1],
                    &mut scores3[..pos + 1],
                    simd,
                );
                gh += 4;
            }
            while gh + 1 < group {
                let h0 = kvh * group + gh;
                let qb0 = (i * q_heads + h0) * key_dim;
                let qb1 = (i * q_heads + h0 + 1) * key_dim;
                let (lo, hi) = group_out.split_at_mut((gh + 1) * value_dim);
                attend_pair(
                    &q[qb0..qb0 + key_dim],
                    &q[qb1..qb1 + key_dim],
                    kc,
                    vc,
                    key_dim,
                    value_dim,
                    kv_heads,
                    kvh,
                    layer,
                    context,
                    scale,
                    &mut lo[gh * value_dim..(gh + 1) * value_dim],
                    &mut hi[..value_dim],
                    &mut scores0[..pos + 1],
                    &mut scores1[..pos + 1],
                    simd,
                );
                gh += 2;
            }
            if gh < group {
                let h = kvh * group + gh;
                let qbase = (i * q_heads + h) * key_dim;
                attend(
                    &q[qbase..qbase + key_dim],
                    kc,
                    vc,
                    key_dim,
                    value_dim,
                    kv_heads,
                    kvh,
                    layer,
                    context,
                    scale,
                    &mut group_out[gh * value_dim..(gh + 1) * value_dim],
                    &mut scores0[..pos + 1],
                    simd,
                );
            }
        }
    });
    out.write_f16_from_f32(&o);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::buffer::{CpuBuffer, CpuFormat};

    fn f16_buf(values: &[f32]) -> CpuBuffer {
        let buf = CpuBuffer::zeroed(values.len() * 2, CpuFormat::F16);
        buf.write_f16_from_f32(values);
        buf
    }

    #[test]
    fn attention_matches_hand_computed_softmax() {
        // kv_heads == q_heads == 1, head_dim = 2, two positions (pos = 1).
        let head_dim = 2;
        let kv_heads = 1;
        let q_heads = 1;
        let context = 2;
        let q = f16_buf(&[1.0, 0.0]);
        // Cache laid out [token][kv_head][head_dim]: t0 then t1.
        let kc = f16_buf(&[1.0, 0.0, 0.0, 1.0]);
        let vc = f16_buf(&[2.0, 0.0, 0.0, 4.0]);
        let out = CpuBuffer::zeroed(q_heads * head_dim * 2, CpuFormat::F16);

        attention_decode_f16(
            &out, &q, &kc, &vc, head_dim, head_dim, kv_heads, q_heads, 1, 0, context,
        );

        // scale = 1/sqrt(2); scores = [scale, 0]; softmax weights w0/w1 below.
        let scale = 1.0f32 / 2f32.sqrt();
        let e0 = 1.0f32; // exp(scale - max) with max = scale
        let e1 = (-scale).exp();
        let denom = e0 + e1;
        let w0 = e0 / denom;
        let w1 = e1 / denom;
        let expected = [w0 * 2.0, w1 * 4.0];

        let got = out.read_f16_as_f32();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-2, "got {g}, expected {e}");
        }
    }

    // Bit-identical parity across heads: with q_heads larger than a typical core
    // count, the per-head chunked output must equal a plain serial loop over the
    // heads, byte for byte. Uses GQA group 4 to exercise the widest reuse path.
    #[test]
    fn attention_parallel_matches_serial_reference() {
        let head_dim = 4;
        let kv_heads = 3;
        let q_heads = 12; // group = 4
        let pos = 5;
        let layer = 0;
        let context = pos + 1;

        let q_vals: Vec<f32> = (0..q_heads * head_dim)
            .map(|i| (i % 11) as f32 * 0.1 - 0.5)
            .collect();
        let cache_len = context * kv_heads * head_dim;
        let kc_vals: Vec<f32> = (0..cache_len).map(|i| (i % 7) as f32 * 0.2 - 0.6).collect();
        let vc_vals: Vec<f32> = (0..cache_len)
            .map(|i| (i % 9) as f32 * 0.15 - 0.4)
            .collect();
        let q = f16_buf(&q_vals);
        let kc = f16_buf(&kc_vals);
        let vc = f16_buf(&vc_vals);
        let out = CpuBuffer::zeroed(q_heads * head_dim * 2, CpuFormat::F16);
        attention_decode_f16(
            &out, &q, &kc, &vc, head_dim, head_dim, kv_heads, q_heads, pos, layer, context,
        );

        // Serial reference mirroring the three passes, head by head.
        let qw = q.read_f16_as_f32();
        let kcb = kc.bytes();
        let vcb = vc.bytes();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group = q_heads / kv_heads;
        let mut expected = vec![0f32; q_heads * head_dim];
        let mut scores = vec![0f32; pos + 1];
        for h in 0..q_heads {
            let kvh = h / group;
            let qbase = h * head_dim;
            let mut m = f32::NEG_INFINITY;
            for (t, score) in scores.iter_mut().enumerate() {
                let kbase = ((layer * context + t) * kv_heads + kvh) * head_dim;
                let mut s = 0f32;
                for d in 0..head_dim {
                    s += qw[qbase + d] * elem(&kcb, kbase + d);
                }
                s *= scale;
                *score = s;
                m = m.max(s);
            }
            let mut denom = 0f32;
            for score in scores.iter_mut() {
                *score = (*score - m).exp();
                denom += *score;
            }
            for (t, &e) in scores.iter().enumerate() {
                let w = e / denom;
                let kbase = ((layer * context + t) * kv_heads + kvh) * head_dim;
                for d in 0..head_dim {
                    expected[qbase + d] += w * elem(&vcb, kbase + d);
                }
            }
        }
        let ref_buf = CpuBuffer::zeroed(q_heads * head_dim * 2, CpuFormat::F16);
        ref_buf.write_f16_from_f32(&expected);
        assert_eq!(out.read_f16_as_f32(), ref_buf.read_f16_as_f32());
    }

    // Bridge between the two kernels: a single-query prefill
    // (n = 1, base = pos) must produce exactly the decode output for that pos.
    // Both go through `attend` with the same f32 ops, so equality is bit-for-bit.
    // Uses GQA group 4 and layer 1 to exercise the widest path and layer offset.
    #[test]
    fn prefill_n1_matches_decode() {
        let head_dim = 4;
        let kv_heads = 2;
        let q_heads = 8; // group = 4
        let pos = 3;
        let layer = 1;
        let context = 8;

        let q_vals: Vec<f32> = (0..q_heads * head_dim)
            .map(|i| (i % 11) as f32 * 0.1 - 0.5)
            .collect();
        // Multi-layer cache so the layer=1 offset is actually distinct.
        let cache_len = (layer + 1) * context * kv_heads * head_dim;
        let kc_vals: Vec<f32> = (0..cache_len).map(|i| (i % 7) as f32 * 0.2 - 0.6).collect();
        let vc_vals: Vec<f32> = (0..cache_len)
            .map(|i| (i % 9) as f32 * 0.15 - 0.4)
            .collect();
        let q = f16_buf(&q_vals);
        let kc = f16_buf(&kc_vals);
        let vc = f16_buf(&vc_vals);

        let dec = CpuBuffer::zeroed(q_heads * head_dim * 2, CpuFormat::F16);
        attention_decode_f16(
            &dec, &q, &kc, &vc, head_dim, head_dim, kv_heads, q_heads, pos, layer, context,
        );
        let pre = CpuBuffer::zeroed(q_heads * head_dim * 2, CpuFormat::F16);
        attention_prefill_f16(
            &pre, &q, &kc, &vc, head_dim, head_dim, kv_heads, q_heads, pos, 1, layer, context,
        );
        assert_eq!(dec.read_f16_as_f32(), pre.read_f16_as_f32());
    }

    // Bit-identical parity across (row, head) units: with more units
    // than a typical core count, the parallel prefill output must equal a plain
    // serial loop over rows and heads, byte for byte. Causal: each row i attends
    // only 0..=i. Uses GQA group 4 to exercise the widest reuse path.
    #[test]
    fn prefill_parallel_matches_serial_reference() {
        let head_dim = 4;
        let kv_heads = 3;
        let q_heads = 12; // group = 4
        let base = 0;
        let n = 5;
        let layer = 0;
        let context = base + n; // max pos = base + n - 1 < context

        let q_vals: Vec<f32> = (0..n * q_heads * head_dim)
            .map(|i| (i % 11) as f32 * 0.1 - 0.5)
            .collect();
        let cache_len = context * kv_heads * head_dim;
        let kc_vals: Vec<f32> = (0..cache_len).map(|i| (i % 7) as f32 * 0.2 - 0.6).collect();
        let vc_vals: Vec<f32> = (0..cache_len)
            .map(|i| (i % 9) as f32 * 0.15 - 0.4)
            .collect();
        let q = f16_buf(&q_vals);
        let kc = f16_buf(&kc_vals);
        let vc = f16_buf(&vc_vals);
        let out = CpuBuffer::zeroed(n * q_heads * head_dim * 2, CpuFormat::F16);
        attention_prefill_f16(
            &out, &q, &kc, &vc, head_dim, head_dim, kv_heads, q_heads, base, n, layer, context,
        );

        // Serial reference mirroring the three passes, row by row then head by head.
        let qw = q.read_f16_as_f32();
        let kcb = kc.bytes();
        let vcb = vc.bytes();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let group = q_heads / kv_heads;
        let mut expected = vec![0f32; n * q_heads * head_dim];
        for i in 0..n {
            let pos = base + i;
            let mut scores = vec![0f32; pos + 1];
            for h in 0..q_heads {
                let kvh = h / group;
                let qbase = (i * q_heads + h) * head_dim;
                let mut m = f32::NEG_INFINITY;
                for (t, score) in scores.iter_mut().enumerate() {
                    let kbase = ((layer * context + t) * kv_heads + kvh) * head_dim;
                    let mut s = 0f32;
                    for d in 0..head_dim {
                        s += qw[qbase + d] * elem(&kcb, kbase + d);
                    }
                    s *= scale;
                    *score = s;
                    m = m.max(s);
                }
                let mut denom = 0f32;
                for score in scores.iter_mut() {
                    *score = (*score - m).exp();
                    denom += *score;
                }
                for (t, &e) in scores.iter().enumerate() {
                    let w = e / denom;
                    let kbase = ((layer * context + t) * kv_heads + kvh) * head_dim;
                    for d in 0..head_dim {
                        expected[qbase + d] += w * elem(&vcb, kbase + d);
                    }
                }
            }
        }
        let ref_buf = CpuBuffer::zeroed(n * q_heads * head_dim * 2, CpuFormat::F16);
        ref_buf.write_f16_from_f32(&expected);
        assert_eq!(out.read_f16_as_f32(), ref_buf.read_f16_as_f32());
    }
}
