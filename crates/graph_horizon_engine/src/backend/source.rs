/*
 * graph_horizon_engine — weight source boundary
 * The entry contract for weights into the compute layer. It exposes the ordered
 * embedding, tail, and layer groups plus their canonical ordered flat view, so
 * loaders never infer ownership from a family-specific tensor count. Concrete
 * families provide descriptors; backend code owns allocation and formats.
*/

use std::ops::Range;

use crate::gguf::tensor_index::TensorInfo;

pub(crate) struct WeightGroups<'a> {
    pub embedding: &'a TensorInfo,
    pub tail: TailWeights<'a>,
    pub layers: Vec<Vec<&'a TensorInfo>>,
}

pub(crate) struct TailWeights<'a> {
    pub norm: &'a TensorInfo,
    pub output: OutputWeight<'a>,
}

pub(crate) enum OutputWeight<'a> {
    Tied,
    Dedicated(&'a TensorInfo),
}

impl<'a> OutputWeight<'a> {
    #[cfg(any(
        test,
        feature = "cpu",
        feature = "vulkan-hybrid",
        feature = "metal",
        feature = "metal-hybrid"
    ))]
    pub(crate) fn is_tied(&self) -> bool {
        matches!(self, Self::Tied)
    }
}

impl<'a> WeightGroups<'a> {
    pub(crate) fn new(
        embedding: &'a TensorInfo,
        norm: &'a TensorInfo,
        output: Option<&'a TensorInfo>,
        layers: Vec<Vec<&'a TensorInfo>>,
    ) -> Self {
        let output = output
            .map(OutputWeight::Dedicated)
            .unwrap_or(OutputWeight::Tied);
        Self {
            embedding,
            tail: TailWeights { norm, output },
            layers,
        }
    }

    #[cfg(any(
        test,
        feature = "vulkan",
        feature = "vulkan-hybrid",
        feature = "metal",
        feature = "metal-hybrid"
    ))]
    fn tensors(&self) -> Vec<&'a TensorInfo> {
        let mut tensors = Vec::new();
        tensors.extend([self.embedding, self.tail.norm]);
        if let OutputWeight::Dedicated(tensor) = self.tail.output {
            tensors.push(tensor);
        }
        tensors.extend(self.layers.iter().flatten().copied());
        tensors
    }
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
// in the canonical order (globals first — embedding, norm, optional dedicated
// output — then each layer's tensors in block order).
pub(crate) trait WeightSource {
    fn groups(&self) -> WeightGroups<'_>;

    #[cfg(any(
        test,
        feature = "vulkan",
        feature = "vulkan-hybrid",
        feature = "metal",
        feature = "metal-hybrid"
    ))]
    fn tensors(&self) -> Vec<&TensorInfo> {
        self.groups().tensors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tensor_index::GgmlType;

    fn tensor(name: &str, dims: &[u64], ggml_type: GgmlType) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: dims.into(),
            ggml_type,
            offset: 0,
        }
    }

    #[test]
    fn tied_tail_preserves_embedding_identity_without_duplication() {
        let embedding = tensor("embedding", &[256], GgmlType::Q4_K);
        let norm = tensor("norm", &[8], GgmlType::F32);
        let groups = WeightGroups::new(&embedding, &norm, None, Vec::new());
        assert!(groups.tail.output.is_tied());
        let names = groups
            .tensors()
            .into_iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["embedding", "norm"]);
    }

    #[test]
    fn dedicated_tail_and_layer_groups_keep_canonical_order() {
        let embedding = tensor("embedding", &[256], GgmlType::Q4_K);
        let norm = tensor("norm", &[8], GgmlType::F32);
        let output = tensor("output", &[256], GgmlType::Q4_K);
        let layer0 = tensor("layer.0", &[256], GgmlType::Q6_K);
        let layer1 = tensor("layer.1", &[256], GgmlType::Q5_K);
        let groups = WeightGroups::new(
            &embedding,
            &norm,
            Some(&output),
            vec![vec![&layer0], vec![&layer1]],
        );
        assert!(!groups.tail.output.is_tied());
        let names = groups
            .tensors()
            .into_iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["embedding", "norm", "output", "layer.0", "layer.1"]);
        assert_eq!(groups.layers.len(), 2);
    }

    #[test]
    fn malformed_and_overflowing_tensor_sizes_are_rejected() {
        let malformed = tensor("bad-block", &[255], GgmlType::Q4_K);
        let overflow = tensor("overflow", &[u64::MAX, 2], GgmlType::F32);
        assert_eq!(malformed.byte_len(), None);
        assert_eq!(overflow.byte_len(), None);
    }
}
