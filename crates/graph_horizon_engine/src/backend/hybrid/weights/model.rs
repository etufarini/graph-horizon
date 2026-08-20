/*
 * graph_horizon_engine — hybrid model weight accounting
 * Computes aligned unique global and ordered per-layer representation bytes from
 * a neutral WeightSource. It performs no family lookup, I/O, placement, or load.
 */

use color_eyre::eyre::{Result, eyre};

use crate::backend::source::{OutputWeight, WeightSource};
use crate::gguf::tensor_index::{GgmlType, TensorInfo};

const WEIGHT_ALIGNMENT: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlobalBytes {
    pub(crate) embedding: u64,
    pub(crate) output_norm: u64,
    pub(crate) output: Option<u64>,
}

impl GlobalBytes {
    pub(crate) fn all(self) -> Option<u64> {
        checked_sum([self.embedding, self.output_norm, self.output.unwrap_or(0)])
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid", test))]
    pub(crate) fn tail(self) -> Option<u64> {
        checked_sum([self.output_norm, self.output.unwrap_or(self.embedding)])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LayerBytes {
    pub(crate) total: u64,
    pub(crate) peak: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WeightBytes {
    pub(crate) globals: GlobalBytes,
    pub(crate) layers: Vec<LayerBytes>,
}

impl WeightBytes {
    pub(crate) fn from_source(source: &dyn WeightSource) -> Result<Self> {
        let groups = source.groups();
        let output = match groups.tail.output {
            OutputWeight::Tied => None,
            OutputWeight::Dedicated(tensor) => Some(representation_bytes(tensor)?),
        };
        let mut layers = Vec::with_capacity(groups.layers.len());
        for group in groups.layers {
            if group.is_empty() {
                return Err(eyre!("hybrid weight accounting malformed layer group"));
            }
            let mut total = 0u64;
            let mut peak = 0u64;
            for tensor in group {
                let bytes = representation_bytes(tensor)?;
                total = total
                    .checked_add(bytes)
                    .ok_or_else(|| eyre!("hybrid weight accounting overflow"))?;
                peak = peak.max(bytes);
            }
            layers.push(LayerBytes { total, peak });
        }
        Ok(Self {
            globals: GlobalBytes {
                embedding: representation_bytes(groups.embedding)?,
                output_norm: representation_bytes(groups.tail.norm)?,
                output,
            },
            layers,
        })
    }

    pub(crate) fn layer_range(&self, range: std::ops::Range<usize>) -> Option<u64> {
        self.layers
            .get(range)?
            .iter()
            .try_fold(0u64, |sum, item| sum.checked_add(item.total))
    }

    // Full retained representation for one homogeneous owner. Tied output
    // weights are counted once because `globals.all()` preserves their identity.
    #[cfg(any(not(any(feature = "vulkan-hybrid", feature = "metal-hybrid")), test))]
    pub(crate) fn total(&self) -> Option<u64> {
        self.globals
            .all()?
            .checked_add(self.layer_range(0..self.layers.len())?)
    }

    #[cfg(any(feature = "vulkan-hybrid", feature = "metal-hybrid"))]
    pub(crate) fn gpu_peak(&self, split: usize) -> Option<u64> {
        let global = if split == 0 {
            self.globals
                .embedding
                .max(self.globals.output_norm)
                .max(self.globals.output.unwrap_or(0))
        } else if split < self.layers.len() {
            self.globals
                .output_norm
                .max(self.globals.output.unwrap_or(self.globals.embedding))
        } else if split == self.layers.len() {
            0
        } else {
            return None;
        };
        Some(
            self.layers
                .get(split..)?
                .iter()
                .fold(global, |peak, layer| peak.max(layer.peak)),
        )
    }
}

fn representation_bytes(tensor: &TensorInfo) -> Result<u64> {
    let raw = tensor
        .byte_len()
        .ok_or_else(|| eyre!("hybrid weight accounting overflow"))?;
    let retained = if tensor.ggml_type == GgmlType::F32 {
        raw / 2
    } else {
        raw
    };
    align_up(retained).ok_or_else(|| eyre!("hybrid weight accounting overflow"))
}

fn align_up(value: u64) -> Option<u64> {
    value
        .checked_add(WEIGHT_ALIGNMENT - 1)
        .map(|n| n / WEIGHT_ALIGNMENT * WEIGHT_ALIGNMENT)
}

fn checked_sum<const N: usize>(values: [u64; N]) -> Option<u64> {
    values
        .into_iter()
        .try_fold(0u64, |sum, value| sum.checked_add(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::source::WeightGroups;

    struct Source(Vec<TensorInfo>, bool, bool);

    impl WeightSource for Source {
        fn groups(&self) -> WeightGroups<'_> {
            let output = self.1.then_some(&self.0[2]);
            let layers = if self.2 {
                vec![Vec::new()]
            } else {
                vec![vec![&self.0[3]], vec![&self.0[4]]]
            };
            WeightGroups::new(&self.0[0], &self.0[1], output, layers)
        }
    }

    fn tensor(name: &str, ty: GgmlType, dims: &[u64]) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: dims.into(),
            ggml_type: ty,
            offset: 0,
        }
    }

    fn source(dedicated: bool) -> Source {
        Source(
            vec![
                tensor("embedding", GgmlType::Q4_K, &[256]),
                tensor("norm", GgmlType::F32, &[8]),
                tensor("output", GgmlType::Q5_K, &[256]),
                tensor("layer.0", GgmlType::Q6_K, &[256]),
                tensor("layer.1", GgmlType::F16, &[16]),
            ],
            dedicated,
            false,
        )
    }

    #[test]
    fn tied_and_dedicated_tail_bytes_preserve_identity() {
        let tied = WeightBytes::from_source(&source(false)).unwrap();
        assert_eq!(tied.globals.output, None);
        assert_eq!(tied.globals.all(), tied.globals.tail());
        let dedicated = WeightBytes::from_source(&source(true)).unwrap();
        assert!(dedicated.globals.output.is_some());
        assert!(dedicated.globals.all().unwrap() > dedicated.globals.tail().unwrap());
    }

    #[test]
    fn layer_group_alignment_and_ranges_are_checked() {
        let weights = WeightBytes::from_source(&source(false)).unwrap();
        assert_eq!(
            weights.layers[0],
            LayerBytes {
                total: 224,
                peak: 224
            }
        );
        assert_eq!(
            weights.layers[1],
            LayerBytes {
                total: 32,
                peak: 32
            }
        );
        assert_eq!(weights.layer_range(0..2), Some(256));
        assert_eq!(weights.total(), Some(448));
        assert_eq!(weights.layer_range(0..3), None);
        assert_eq!(align_up(u64::MAX), None);
    }

    #[test]
    fn malformed_empty_layer_group_is_rejected() {
        let mut source = source(false);
        source.2 = true;
        assert_eq!(
            WeightBytes::from_source(&source).unwrap_err().to_string(),
            "hybrid weight accounting malformed layer group"
        );
    }
}
