/*
 * graph_orizon_engine — shared scalar FP16 conversion
 * This file is the backend-wide authority for IEEE binary16/binary32 scalar
 * conversion. Values are represented as raw `u16` bits or native `f32`; narrowing
 * uses round-to-nearest-even, widening is exact, and byte conversion reads and
 * writes little-endian data. CPU SIMD code may accelerate slices, while CPU
 * weights, KV metadata, and Vulkan upload/readback reuse these scalar primitives.
*/

#[cfg(any(
    feature = "cpu",
    feature = "vulkan-hybrid",
    feature = "metal-hybrid",
    test
))]
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // A binary16 subnormal is normalized before applying the f32 bias.
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (127 - 15 + 2 + e) as u32;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        // Preserve infinity and keep a non-zero NaN payload non-zero.
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub(crate) fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }

    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Add the implicit leading bit, then round the shifted subnormal to even.
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let round =
            u16::from(rem > (1 << (shift - 1)) || (rem == (1 << (shift - 1)) && half & 1 == 1));
        return sign | (half + round);
    }

    let half = (e as u16) << 10 | (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    // Ties increment only an odd retained mantissa, producing round-to-even.
    let round = u16::from(rem > 0x1000 || (rem == 0x1000 && half & 1 == 1));
    sign | (half + round)
}

pub(crate) fn f32_to_f16_bytes(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / 2);
    for chunk in src.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.extend_from_slice(&f32_to_f16(value).to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representable_values_round_trip_exactly() {
        for value in [0.0, -0.0, 1.0, -2.0, 0.5, -0.25, 65504.0, -65504.0] {
            assert_eq!(f16_to_f32(f32_to_f16(value)).to_bits(), value.to_bits());
        }
    }

    #[test]
    fn special_values_keep_their_classification() {
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn narrowing_rounds_ties_to_even() {
        assert_eq!(f32_to_f16(1.000_488_3), 0x3c00);
        assert_eq!(f32_to_f16(1.001_464_8), 0x3c02);
    }

    #[test]
    fn byte_conversion_is_little_endian() {
        let src = [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat();
        assert_eq!(f32_to_f16_bytes(&src), [0x00, 0x3c, 0x00, 0xc0]);
    }
}
