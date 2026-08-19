/*
 * graph_horizon_engine — runtime KV-cache scheme selection (the `--kv-quant` contract)
 * One `KvQuant` value chosen at load from the CLI, fixed BEFORE any alloc and
 * never changed for the engine's lifetime (I1). Branching on it is allowed only
 * at op-dispatch level — once per `kv_write`/`attention_*` call — never per
 * token/element inside kernels.
 *
 * The unit of quantization is the VECTOR: the `head_dim` values of one
 * (token, kv_head), for K or for V. Each scheme fixes the payload bytes per
 * vector and the metadata bytes per role; the byte-region math over whole
 * buffers lives in `layout`, and the int8 quantization arithmetic lives in
 * `int8` (bit-identical CPU<->Vulkan, I3).
 *
 * Adding a scheme: a variant here (parse/name/sizing rows), the normative
 * scalar reference under `kv_cache`, one arm in each backend's per-call
 * dispatch (CPU `kernels/attention`, Vulkan `kernels/attention.rs` + shaders),
 * and a row in `support/profiling/validate-kv.sh` / the docs.
*/

// The KV-cache schemes selectable via `--kv-quant`. Parsing is lowercase and
// case-sensitive; the CLI layer owns the invalid-value error message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    #[default]
    F16,
    Int8,
}

// The two cache roles. Current schemes size K and V identically, but keeping the
// role in the layout API makes the buffer contract explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvRole {
    Key,
    Value,
}

impl KvQuant {
    // Every variant, for parse/name round-trip tests and CLI value listings.
    pub const ALL: &'static [KvQuant] = &[KvQuant::F16, KvQuant::Int8];

    // Parses a CLI value. Case-sensitive by design: the flag vocabulary is
    // lowercase like every other flag value (D8). `None` = invalid (the CLI
    // emits the error listing `ALL`).
    pub fn parse(s: &str) -> Option<KvQuant> {
        match s {
            "f16" => Some(KvQuant::F16),
            "int8" => Some(KvQuant::Int8),
            _ => None,
        }
    }

    // The canonical CLI name (inverse of `parse`).
    pub fn name(self) -> &'static str {
        match self {
            KvQuant::F16 => "f16",
            KvQuant::Int8 => "int8",
        }
    }

    // Payload bytes of ONE vector (the head_dim values of a (token, kv_head)).
    // f16: 2 bytes/value; int8: 1 byte/value (u8 code).
    pub fn payload_bytes_per_vector(self, head_dim: usize) -> usize {
        match self {
            KvQuant::F16 => 2 * head_dim,
            KvQuant::Int8 => head_dim,
        }
    }

    // Metadata bytes of ONE vector, per role. f16: none; int8: min + scale as
    // f16 (2+2). K and V share this layout, so the role remains explicit.
    pub fn meta_bytes_per_vector(self, _role: KvRole) -> usize {
        match self {
            KvQuant::F16 => 0,
            KvQuant::Int8 => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_round_trips_through_parse_name() {
        for &s in KvQuant::ALL {
            assert_eq!(KvQuant::parse(s.name()), Some(s));
        }
    }

    #[test]
    fn parse_is_case_sensitive_and_rejects_unknown() {
        assert_eq!(KvQuant::parse("INT8"), None); // D8: lowercase only
        assert_eq!(KvQuant::parse("F16"), None);
        assert_eq!(KvQuant::parse(""), None);
        assert_eq!(KvQuant::parse("int9"), None);
        assert_eq!(KvQuant::parse("q4"), None);
    }

    #[test]
    fn sizing_per_vector_matches_the_formats() {
        // f16: 128 values × 2 bytes, no metadata.
        assert_eq!(KvQuant::F16.payload_bytes_per_vector(128), 256);
        assert_eq!(KvQuant::F16.meta_bytes_per_vector(KvRole::Key), 0);
        // int8: 128 u8 codes + (min, scale) as f16.
        assert_eq!(KvQuant::Int8.payload_bytes_per_vector(128), 128);
        assert_eq!(KvQuant::Int8.meta_bytes_per_vector(KvRole::Key), 4);
    }

    #[test]
    fn default_is_f16() {
        assert_eq!(KvQuant::default(), KvQuant::F16);
    }
}
