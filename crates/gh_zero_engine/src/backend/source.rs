/*
 * gh_zero_engine — weight source boundary
 * The entry contract for weights into the compute layer. It exposes the ordered
 * tensor list plus its dense layout, so loaders never infer optional output or
 * Q/K-norm slots from a family-specific tensor count. Concrete families provide
 * the descriptors; backend code owns only allocation and format handling.
*/

use std::ops::Range;

use crate::gguf::tensor_index::TensorInfo;

#[derive(Clone, Copy)]
pub(crate) struct WeightLayout {
    pub has_output: bool,
    pub layer_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WeightSelection {
    pub layers: Range<usize>,
    pub embedding: bool,
    pub tail: bool,
}

impl WeightSelection {
    pub(crate) fn full(layer_count: usize) -> Self {
        Self {
            layers: 0..layer_count,
            embedding: true,
            tail: true,
        }
    }
}

// A source of model weights, exposed to the backends as a flat list of tensors
// in the canonical order (globals first — `token_embd`, `output_norm`, optional
// `output` — then each layer's tensors in block order). `layout` states which
// optional slots exist; loaders validate the exact count before allocation.
pub(crate) trait WeightSource {
    fn tensors(&self) -> Vec<&TensorInfo>;
    fn layout(&self) -> WeightLayout;
}
