/*
 * graph_horizon_engine — INT8 per-token asymmetric KV format
 * One group per (token, kv_head) vector of head_dim values; metadata = min,scale
 * as f16. The exact arithmetic here is normative: CPU and GPU must produce
 * bit-identical codes. Format, for a group x[0..head_dim] (f32, widened
 * from the incoming f16):
 *   1. min = min(x), max = max(x) in f32;
 *   2. scale = (max - min) * (1.0/255.0) in f32 — a MULTIPLY by the f32
 *      constant, never a division: Vulkan guarantees correctly-rounded
 *      fmul/fadd but only 2.5 ULP for fdiv, so a division here would break
 *      CPU<->GPU bit parity;
 *   3. both are rounded to f16 and the F16-ROUNDED values are used for coding
 *      (coding against the stored constants is what makes dequant consistent);
 *   4. q_j = the largest q in 0..=255 with x_j >= min_f16 + (q - 0.5)*scale_f16,
 *      each boundary evaluated as one rounded multiply then one rounded add
 *      (no FMA contraction, no division) — mathematically the same rule as
 *      clamp(floor((x_j - min_f16)/scale_f16 + 0.5), 0, 255); ties resolve to
 *      the upper code because the boundary comparison uses >=, found by
 *      binary search over the monotone boundaries;
 *      IF scale_f16 == 0 THEN q_j = 0 (constant group);
 *   5. payload = head_dim u8 codes; metadata = min_f16, scale_f16 (2+2 bytes LE).
 * Dequant: x~_j = min_f16 + q_j * scale_f16 in f32.
 * NaN/inf inputs are NOT sanitized:
 * if they reach kv_write the upstream forward is already broken; the codes they
 * produce are deterministic garbage, never a crash.
*/

// KV metadata uses the same backend-neutral IEEE conversion as CPU and Vulkan.
pub(crate) use crate::backend::f16::{f16_to_f32, f32_to_f16};

// Quantizes one group (the head_dim f32 values of a (token, kv_head) vector)
// into `payload` (one u8 code per value, same length as `x`) and returns the
// 4-byte metadata `[min_f16, scale_f16]` (little-endian). Normative reference:
// the CPU kernel calls this directly and the Vulkan shader mirrors it.
pub(crate) fn quantize_group(x: &[f32], payload: &mut [u8]) -> [u8; 4] {
    debug_assert_eq!(x.len(), payload.len());
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &v in x {
        min = min.min(v);
        max = max.max(v);
    }
    // Multiply by the constant reciprocal; never divide (see the header).
    let scale = (max - min) * (1.0f32 / 255.0f32);
    // Round min/scale to f16 first, then code against the rounded values:
    // dequant reads exactly these stored constants, so coding with anything
    // more precise would bias the codes.
    let min_bits = f32_to_f16(min);
    let scale_bits = f32_to_f16(scale);
    let min_r = f16_to_f32(min_bits);
    let scale_r = f16_to_f32(scale_bits);
    if scale_r == 0.0 {
        // Constant group (max == min, includes all-zero): codes 0, dequant
        // returns exactly min.
        payload.fill(0);
    } else {
        for (q, &v) in payload.iter_mut().zip(x) {
            *q = code_of(v, min_r, scale_r);
        }
    }
    let mb = min_bits.to_le_bytes();
    let sb = scale_bits.to_le_bytes();
    [mb[0], mb[1], sb[0], sb[1]]
}

// The largest code q in 0..=255 with `v >= min + (q - 0.5) * scale` (scale > 0):
// binary search over the 255 monotone boundaries. Each boundary costs exactly
// one rounded f32 multiply and one rounded f32 add — the two operations Vulkan
// guarantees correctly rounded, so the GPU mirror is bit-identical. The
// boundaries are non-decreasing in q under rounding, so the search is exact.
#[inline]
fn code_of(v: f32, min: f32, scale: f32) -> u8 {
    let above = |q: u32| v >= min + (q as f32 - 0.5) * scale;
    let mut code = 0u32; // invariant: q == 0, or `above(code)` holds
    let mut step = 128u32;
    while step > 0 {
        let cand = code + step;
        if cand <= 255 && above(cand) {
            code = cand;
        }
        step >>= 1;
    }
    code as u8
}

// Widens the 4-byte metadata back to the (min, scale) f32 pair used by dequant.
pub(crate) fn meta_decode(meta: &[u8]) -> (f32, f32) {
    (
        f16_to_f32(u16::from_le_bytes([meta[0], meta[1]])),
        f16_to_f32(u16::from_le_bytes([meta[2], meta[3]])),
    )
}

// Dequantizes one code against the group's (already widened) min/scale.
#[inline]
pub(crate) fn dequant(code: u8, min: f32, scale: f32) -> f32 {
    min + code as f32 * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    // Seeded xorshift64* (same generator family as crate::rng), local to the
    // test so the reference stays dependency-free.
    struct XorShift(u64);
    impl XorShift {
        fn next_f32(&mut self) -> f32 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    fn roundtrip(x: &[f32]) -> (Vec<f32>, f32) {
        let mut payload = vec![0u8; x.len()];
        let meta = quantize_group(x, &mut payload);
        let (min, scale) = meta_decode(&meta);
        (
            payload.iter().map(|&q| dequant(q, min, scale)).collect(),
            scale,
        )
    }

    #[test]
    fn random_groups_round_trip_within_half_scale() {
        let mut rng = XorShift(0x5451_0001);
        for case in 0..50 {
            let x: Vec<f32> = (0..128).map(|_| (rng.next_f32() - 0.5) * 8.0).collect();
            let (back, scale) = roundtrip(&x);
            // f16-rounding of min/scale slightly widens the worst case: allow
            // scale/2 plus the f16 ulp contribution of the reconstruction.
            let tol = scale / 2.0 + (scale * 255.0) * 1e-3;
            for (i, (&a, &b)) in x.iter().zip(&back).enumerate() {
                assert!(
                    (a - b).abs() <= tol,
                    "case {case} elem {i}: {a} vs {b} (tol {tol})"
                );
            }
        }
    }

    #[test]
    fn constant_group_stores_zero_scale_and_exact_min() {
        for &c in &[0.0f32, 1.5, -3.25] {
            let x = [c; 128];
            let mut payload = [1u8; 128];
            let meta = quantize_group(&x, &mut payload);
            let (min, scale) = meta_decode(&meta);
            assert_eq!(scale, 0.0);
            assert!(payload.iter().all(|&q| q == 0));
            let back = dequant(payload[0], min, scale);
            assert_eq!(back, f16_to_f32(f32_to_f16(c))); // exact, no NaN/inf
            assert!(back.is_finite());
        }
    }

    // The formula must code against the F16-ROUNDED min/scale: with a min that
    // is not f16-representable, coding against the unrounded f32 min would
    // shift codes near the boundaries. Verify code 0 dequantizes back to the
    // stored (rounded) min exactly.
    #[test]
    fn coding_uses_the_f16_rounded_constants() {
        let mut x = [0.5f32; 128];
        x[0] = 0.1000123; // not f16-representable
        x[1] = 7.7;
        let mut payload = [0u8; 128];
        let meta = quantize_group(&x, &mut payload);
        let (min, scale) = meta_decode(&meta);
        assert_eq!(min, f16_to_f32(f32_to_f16(0.1000123)));
        assert_eq!(payload[0], 0); // the min element codes to 0 against the rounded min
        assert_eq!(dequant(payload[0], min, scale), min);
    }

    // One extreme outlier still round-trips within the (wide) scale bound.
    #[test]
    fn outlier_group_round_trips_within_bound() {
        let mut rng = XorShift(0x5451_0002);
        let mut x: Vec<f32> = (0..128).map(|_| rng.next_f32() * 0.01).collect();
        x[63] = 1000.0;
        let (back, scale) = roundtrip(&x);
        let tol = scale / 2.0 + (scale * 255.0) * 1e-3;
        for (&a, &b) in x.iter().zip(&back) {
            assert!((a - b).abs() <= tol, "{a} vs {b} (tol {tol})");
        }
    }

    #[test]
    fn f16_helpers_match_known_values() {
        assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(-2.5)), -2.5);
        assert_eq!(f16_to_f32(f32_to_f16(65504.0)), 65504.0);
        assert_eq!(f32_to_f16(0.0), 0);
    }
}
