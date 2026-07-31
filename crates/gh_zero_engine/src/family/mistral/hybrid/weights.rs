/*
 * gh_zero_engine — hybrid weight representation accounting
 * Converts validated Ministral descriptors into unique global/layer bytes without placement, I/O, allocation or extrapolation.
 */

use color_eyre::eyre::{Result, eyre};

use crate::family::mistral::MistralConfig;
use crate::gguf::tensor_index::{GgmlType, TensorInfo};
use crate::kv_cache::layout;
use crate::kv_cache::scheme::{KvQuant, KvRole};

use super::super::tensors::{MistralLayer, MistralTensors, OutputTensor};

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

    pub(crate) fn tail(self) -> Option<u64> {
        // A tied lm_head reuses embedding; a dedicated output replaces it.
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

pub(crate) struct RuntimeBytes {
    pub(crate) scratch: u64,
    pub(crate) logits: u64,
    pub(crate) gpu_fixed: u64,
    pub(crate) kv_per_layer: u64,
    pub(crate) crossing: u64,
}

impl RuntimeBytes {
    pub(crate) fn new(cfg: &MistralConfig, context: usize, scheme: KvQuant) -> Result<Self> {
        let row = checked_sum([
            bytes(cfg.embedding_length, 4)?,
            bytes(cfg.embedding_length, 2)?,
            bytes(cfg.q_width, 2)?,
            bytes(cfg.k_width, 2)?,
            bytes(cfg.v_width, 2)?,
            bytes(cfg.attention_width, 2)?,
            bytes(cfg.embedding_length, 2)?,
            bytes(cfg.feed_forward_length, 2)?,
            bytes(cfg.feed_forward_length, 2)?,
            bytes(cfg.feed_forward_length, 2)?,
            bytes(cfg.embedding_length, 2)?,
        ])
        .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?;
        let logits = bytes(cfg.vocab_size, 4)?;
        let reduce = (crate::backend::vulkan::kernels::reduce::TOPK_GROUPS as u64)
            .checked_mul(crate::backend::vulkan::kernels::reduce::MAX_K as u64)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?;
        let mmvq = crate::backend::vulkan::MMVQ_SCRATCH_IN_DIM
            + crate::backend::vulkan::MMVQ_SCRATCH_IN_DIM / 32 * 2 * 4;
        let key = layout::buffer_bytes(
            scheme,
            KvRole::Key,
            1,
            context,
            cfg.kv_head_count,
            cfg.key_length,
        );
        let value = layout::buffer_bytes(
            scheme,
            KvRole::Value,
            1,
            context,
            cfg.kv_head_count,
            cfg.value_length,
        );
        Ok(Self {
            scratch: row
                .checked_mul(1 + super::super::graph::prefill::BATCH_ROWS as u64)
                .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?,
            logits,
            gpu_fixed: checked_sum([logits, reduce, mmvq])
                .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?,
            kv_per_layer: checked_sum([key, value])
                .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?,
            crossing: bytes(cfg.embedding_length, 4)?
                .checked_mul(super::super::graph::prefill::BATCH_ROWS as u64)
                .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))?,
        })
    }
}

impl WeightBytes {
    pub(crate) fn from_tensors(tensors: &MistralTensors<'_>) -> Result<Self> {
        let embedding = representation_bytes(tensors.token_embd)?;
        let output_norm = representation_bytes(tensors.output_norm)?;
        let output = match tensors.output {
            OutputTensor::Tied => None,
            OutputTensor::Dedicated(tensor) => Some(representation_bytes(tensor)?),
        };
        let mut layers = Vec::with_capacity(tensors.layers.len());
        for layer in &tensors.layers {
            layers.push(layer_bytes(layer)?);
        }
        Ok(Self {
            globals: GlobalBytes {
                embedding,
                output_norm,
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

fn layer_bytes(layer: &MistralLayer<'_>) -> Result<LayerBytes> {
    let tensors = [
        layer.attn_norm,
        layer.attn_q,
        layer.attn_k,
        layer.attn_v,
        layer.attn_output,
        layer.ffn_norm,
        layer.ffn_gate,
        layer.ffn_up,
        layer.ffn_down,
    ];
    let mut sum = 0u64;
    let mut peak = 0u64;
    for tensor in tensors {
        let bytes = representation_bytes(tensor)?;
        sum = sum
            .checked_add(bytes)
            .ok_or_else(|| eyre!("hybrid weight accounting overflow"))?;
        peak = peak.max(bytes);
    }
    Ok(LayerBytes { total: sum, peak })
}

fn representation_bytes(tensor: &TensorInfo) -> Result<u64> {
    let raw = tensor
        .byte_len()
        .ok_or_else(|| eyre!("hybrid weight accounting overflow"))?;
    // Backends convert F32 norms to F16; other tensors retain their representation.
    align_up(if tensor.ggml_type == GgmlType::F32 {
        raw / 2
    } else {
        raw
    })
    .ok_or_else(|| eyre!("hybrid weight accounting overflow"))
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

fn bytes(items: usize, item_bytes: usize) -> Result<u64> {
    items
        .checked_mul(item_bytes)
        .and_then(|total| u64::try_from(total).ok())
        .ok_or_else(|| eyre!("hybrid placement arithmetic overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(ggml_type: GgmlType, dims: &[u64]) -> TensorInfo {
        TensorInfo {
            name: "test.weight".into(),
            dims: dims.to_vec(),
            ggml_type,
            offset: 0,
        }
    }

    #[test]
    fn tied_tail_counts_embedding_once() {
        let globals = GlobalBytes {
            embedding: 320,
            output_norm: 32,
            output: None,
        };
        assert_eq!(globals.all(), Some(352));
        assert_eq!(globals.tail(), Some(352));
    }

    #[test]
    fn dedicated_tail_does_not_include_embedding() {
        let globals = GlobalBytes {
            embedding: 320,
            output_norm: 32,
            output: Some(640),
        };
        assert_eq!(globals.all(), Some(992));
        assert_eq!(globals.tail(), Some(672));
    }

    #[test]
    fn alignment_and_overflow_are_checked() {
        assert_eq!(align_up(1), Some(32));
        assert_eq!(align_up(32), Some(32));
        assert_eq!(align_up(33), Some(64));
        assert_eq!(align_up(u64::MAX), None);
        assert_eq!(checked_sum([u64::MAX, 1]), None);
    }

    #[test]
    fn representation_sizes_cover_q8_and_mixed_q4_q6_formats() {
        assert_eq!(
            representation_bytes(&tensor(GgmlType::F32, &[3])).unwrap(),
            32
        );
        assert_eq!(
            representation_bytes(&tensor(GgmlType::Q8_0, &[32])).unwrap(),
            64
        );
        assert_eq!(
            representation_bytes(&tensor(GgmlType::Q4_K, &[256])).unwrap(),
            160
        );
        assert_eq!(
            representation_bytes(&tensor(GgmlType::Q6_K, &[256])).unwrap(),
            224
        );
        assert!(representation_bytes(&tensor(GgmlType::Q4_K, &[255])).is_err());
    }

    #[test]
    fn heterogeneous_layers_are_summed_independently() {
        let weights = WeightBytes {
            globals: GlobalBytes {
                embedding: 32,
                output_norm: 32,
                output: None,
            },
            layers: vec![
                LayerBytes {
                    total: 64,
                    peak: 32,
                },
                LayerBytes {
                    total: 96,
                    peak: 64,
                },
                LayerBytes {
                    total: 160,
                    peak: 128,
                },
            ],
        };
        assert_eq!(weights.layer_range(0..3), Some(320));
        assert_eq!(weights.layer_range(1..3), Some(256));
        assert_eq!(weights.gpu_peak(0), Some(128));
        assert_eq!(weights.gpu_peak(2), Some(128));
        assert_eq!(weights.gpu_peak(3), Some(0));
        assert_eq!(weights.gpu_peak(4), None);
    }
}
