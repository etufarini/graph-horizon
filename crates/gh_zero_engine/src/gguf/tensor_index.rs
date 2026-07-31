/*
 * gh_zero_engine — GGUF tensor index
 * Defines the typed tensor descriptor (name, shape, ggml dtype, data offset)
 * and a name→tensor index over the tensor table of a GgufFile. The descriptor
 * also computes a tensor's byte length from its ggml block layout, used by the
 * loader to bounds-check slices.
*/

use std::collections::HashMap;

// GGML tensor element type (dtype / quantization). Only the variants relevant
// to supported GGUFs are named; any other tag is preserved as `Unknown`
// so the loader can still index the tensor table. Block/type sizes follow ggml. The
// variant names mirror ggml's canonical type names (e.g. Q4_K), hence the allow.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2_K,
            11 => GgmlType::Q3_K,
            12 => GgmlType::Q4_K,
            13 => GgmlType::Q5_K,
            14 => GgmlType::Q6_K,
            15 => GgmlType::Q8_K,
            30 => GgmlType::BF16,
            other => GgmlType::Unknown(other),
        }
    }

    pub fn name(&self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::BF16 => "BF16".into(),
            GgmlType::Q4_0 => "Q4_0".into(),
            GgmlType::Q4_1 => "Q4_1".into(),
            GgmlType::Q5_0 => "Q5_0".into(),
            GgmlType::Q5_1 => "Q5_1".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::Q2_K => "Q2_K".into(),
            GgmlType::Q3_K => "Q3_K".into(),
            GgmlType::Q4_K => "Q4_K".into(),
            GgmlType::Q5_K => "Q5_K".into(),
            GgmlType::Q6_K => "Q6_K".into(),
            GgmlType::Q8_K => "Q8_K".into(),
            GgmlType::Unknown(v) => format!("unknown({v})"),
        }
    }

    // Single source of truth for the quantization formats the v0 engine can
    // run: F32/F16 (unquantized — norms are F32, converted to FP16 on upload),
    // the Q4_K_M mix (Q4_K + Q6_K), Q8_0 (embedders ship fully Q8_0) and Q5_K
    // plus Q5_K retained for backend compatibility. Every other dtype is
    // rejected up front by the family detect with the standard unsupported-model
    // error, so the matmul dispatch (vulkan::kernels::matmul) and the weight
    // upload only ever see these.
    pub fn supported_weight(&self) -> bool {
        matches!(
            self,
            GgmlType::F32
                | GgmlType::F16
                | GgmlType::Q4_K
                | GgmlType::Q5_K
                | GgmlType::Q6_K
                | GgmlType::Q8_0
        )
    }

    // (block size in elements, bytes per block). None for unknown types, whose
    // size cannot be derived without knowing their layout.
    fn block_type_size(&self) -> Option<(u64, u64)> {
        Some(match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 => (1, 2),
            GgmlType::BF16 => (1, 2),
            GgmlType::Q4_0 => (32, 18),
            GgmlType::Q4_1 => (32, 20),
            GgmlType::Q5_0 => (32, 22),
            GgmlType::Q5_1 => (32, 24),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q2_K => (256, 84),
            GgmlType::Q3_K => (256, 110),
            GgmlType::Q4_K => (256, 144),
            GgmlType::Q5_K => (256, 176),
            GgmlType::Q6_K => (256, 210),
            GgmlType::Q8_K => (256, 292),
            GgmlType::Unknown(_) => return None,
        })
    }
}

// One tensor's descriptor from the GGUF tensor table. `offset` is relative to
// the start of the file's data section (resolved by the loader).
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    // Number of elements = product of dims (1 for a 0-dim tensor). None on
    // multiplication overflow (malformed dims on untrusted input).
    pub fn element_count(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
    }

    // Total byte length of the tensor data, from its ggml block layout. None
    // when the type is unknown or the element count is not a whole number of
    // blocks (which would make the tensor malformed for this dtype).
    pub fn byte_len(&self) -> Option<u64> {
        let (block, size) = self.ggml_type.block_type_size()?;
        let n = self.element_count()?;
        if block == 0 || n % block != 0 {
            return None;
        }
        (n / block).checked_mul(size)
    }
}

// Name → tensor index over a borrowed tensor table. Built once; lookups are
// O(1). On duplicate tensor names the last occurrence wins.
pub(crate) struct TensorIndex<'a> {
    tensors: &'a [TensorInfo],
    by_name: HashMap<&'a str, usize>,
}

impl<'a> TensorIndex<'a> {
    pub(crate) fn new(tensors: &'a [TensorInfo]) -> Self {
        let mut by_name = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            by_name.insert(t.name.as_str(), i);
        }
        Self { tensors, by_name }
    }

    // Returns borrows tied to the table lifetime `'a`, not to `&self`, so the
    // resolved descriptors can outlive the index (e.g. in a WeightSet).
    pub(crate) fn get(&self, name: &str) -> Option<&'a TensorInfo> {
        let tensors = self.tensors;
        self.by_name.get(name).map(|&i| &tensors[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Q5_K remains a supported backend dtype; Q4_0 stays rejected so
    // the gate keeps excluding the formats the engine cannot run.
    #[test]
    fn q5_k_is_a_supported_weight() {
        assert!(GgmlType::Q5_K.supported_weight());
        assert!(!GgmlType::Q4_0.supported_weight());
    }
}
