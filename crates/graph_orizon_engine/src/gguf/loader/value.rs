/*
 * graph_orizon_engine — GGUF metadata value
 * The typed GGUF metadata value model and its accessors. Mirrors the GGUF
 * value-type tags; arrays are read eagerly into a Vec, so the metadata map is a
 * plain owned structure. Pure data — no I/O, no architecture knowledge.
*/

// A typed GGUF metadata value. Mirrors the GGUF value-type tags; arrays are read
// eagerly into a Vec, so the metadata map is a plain owned structure.
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    // Reads any integer value as u64 (signed values only when non-negative), so
    // hyper-parameters stored as any int width can be consumed uniformly.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            GgufValue::U8(v) => Some(v as u64),
            GgufValue::U16(v) => Some(v as u64),
            GgufValue::U32(v) => Some(v as u64),
            GgufValue::U64(v) => Some(v),
            GgufValue::I8(v) if v >= 0 => Some(v as u64),
            GgufValue::I16(v) if v >= 0 => Some(v as u64),
            GgufValue::I32(v) if v >= 0 => Some(v as u64),
            GgufValue::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            GgufValue::F32(v) => Some(v),
            GgufValue::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            GgufValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_widths_read_as_u64_when_non_negative() {
        assert_eq!(GgufValue::U32(7).as_u64(), Some(7));
        assert_eq!(GgufValue::I32(7).as_u64(), Some(7));
        // A negative signed int is not a valid u64 hyper-parameter.
        assert_eq!(GgufValue::I32(-1).as_u64(), None);
        // A non-integer value has no u64 reading.
        assert_eq!(GgufValue::String("x".into()).as_u64(), None);
    }

    #[test]
    fn typed_accessors_match_variant() {
        assert_eq!(GgufValue::F64(1.5).as_f32(), Some(1.5));
        assert_eq!(GgufValue::String("hi".into()).as_str(), Some("hi"));
        assert!(
            GgufValue::Array(vec![GgufValue::U8(1)])
                .as_array()
                .is_some()
        );
        assert_eq!(GgufValue::Bool(true).as_bool(), Some(true));
        assert_eq!(GgufValue::U8(1).as_str(), None);
    }
}
