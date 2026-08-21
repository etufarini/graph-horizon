/*
 * Metal profiler classification
 * Assigns sampled dispatches to request phases and dense graph operations using
 * the fixed seven-matmul layer order. It owns no Metal resources or reporting.
 */

use crate::backend::metal::pipeline::Kernel;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Phase {
    Prefill,
    Decode,
    Sampling,
}

impl Phase {
    pub(super) const COUNT: usize = 3;
    pub(super) const ALL: [Self; Self::COUNT] = [Self::Prefill, Self::Decode, Self::Sampling];

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Sampling => "sampling",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum Category {
    Attention,
    ProjectionQ,
    ProjectionK,
    ProjectionV,
    ProjectionOutput,
    MlpGate,
    MlpUp,
    MlpDown,
    Normalization,
    Rope,
    KvCache,
    Elementwise,
    Embedding,
    Logits,
    Reduction,
}

impl Category {
    pub(super) const COUNT: usize = 15;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::Attention,
        Self::ProjectionQ,
        Self::ProjectionK,
        Self::ProjectionV,
        Self::ProjectionOutput,
        Self::MlpGate,
        Self::MlpUp,
        Self::MlpDown,
        Self::Normalization,
        Self::Rope,
        Self::KvCache,
        Self::Elementwise,
        Self::Embedding,
        Self::Logits,
        Self::Reduction,
    ];

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Attention => "attention",
            Self::ProjectionQ => "projection_q",
            Self::ProjectionK => "projection_k",
            Self::ProjectionV => "projection_v",
            Self::ProjectionOutput => "projection_output",
            Self::MlpGate => "mlp_gate",
            Self::MlpUp => "mlp_up",
            Self::MlpDown => "mlp_down",
            Self::Normalization => "normalization",
            Self::Rope => "rope",
            Self::KvCache => "kv_cache",
            Self::Elementwise => "elementwise",
            Self::Embedding => "embedding",
            Self::Logits => "logits",
            Self::Reduction => "reduction",
        }
    }
}

pub(super) fn classify(
    kernel: Kernel,
    constants: &[u8],
    matmul_slot: &mut usize,
) -> (Category, Option<Phase>) {
    match kernel {
        Kernel::Matmul if fp32_output(constants) => (Category::Logits, None),
        Kernel::Matmul
        | Kernel::MatmulBatched
        | Kernel::MatmulBatchedWide
        | Kernel::MatmulBatchedTensor => {
            let category = match *matmul_slot % 7 {
                0 => Category::ProjectionQ,
                1 => Category::ProjectionK,
                2 => Category::ProjectionV,
                3 => Category::ProjectionOutput,
                4 => Category::MlpGate,
                5 => Category::MlpUp,
                _ => Category::MlpDown,
            };
            *matmul_slot += 1;
            let phase = if matches!(
                kernel,
                Kernel::MatmulBatched | Kernel::MatmulBatchedWide | Kernel::MatmulBatchedTensor
            ) {
                Phase::Prefill
            } else {
                Phase::Decode
            };
            (category, Some(phase))
        }
        Kernel::Attention => (Category::Attention, None),
        #[cfg(feature = "metal")]
        Kernel::AttentionGqaDecode | Kernel::AttentionGqaSplit | Kernel::AttentionGqaReduce => {
            (Category::Attention, Some(Phase::Decode))
        }
        #[cfg(feature = "metal")]
        Kernel::AttentionPrefillMatrix => (Category::Attention, Some(Phase::Prefill)),
        Kernel::Embedding => (Category::Embedding, None),
        Kernel::Rmsnorm => (Category::Normalization, None),
        Kernel::Rope => (Category::Rope, None),
        Kernel::KvWrite => (Category::KvCache, None),
        Kernel::SiluMul | Kernel::ResidualAdd => (Category::Elementwise, None),
        Kernel::Argmax | Kernel::Topk => (Category::Reduction, Some(Phase::Sampling)),
    }
}

fn fp32_output(constants: &[u8]) -> bool {
    constants
        .get(12..16)
        .and_then(|bytes| bytes.try_into().ok())
        .is_some_and(|bytes| u32::from_ne_bytes(bytes) == 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_order_and_logits_are_distinct() {
        let mut slot = 0;
        let names = (0..7)
            .map(|_| {
                classify(Kernel::MatmulBatchedWide, &[0; 16], &mut slot)
                    .0
                    .name()
            })
            .collect::<Vec<_>>();
        assert_eq!(names[0], "projection_q");
        assert_eq!(names[3], "projection_output");
        assert_eq!(names[6], "mlp_down");
        let mut logits = [0; 16];
        logits[12..].copy_from_slice(&1u32.to_ne_bytes());
        assert_eq!(
            classify(Kernel::Matmul, &logits, &mut slot).0.name(),
            "logits"
        );
        assert_eq!(slot, 7);
    }
}
