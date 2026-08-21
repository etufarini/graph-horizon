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
    AttentionScan,
    AttentionReduce,
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
    Activation,
    Residual,
    Embedding,
    Logits,
    Reduction,
}

impl Category {
    pub(super) const COUNT: usize = 17;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::AttentionScan,
        Self::AttentionReduce,
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
        Self::Activation,
        Self::Residual,
        Self::Embedding,
        Self::Logits,
        Self::Reduction,
    ];

    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::AttentionScan => "attention_scan",
            Self::AttentionReduce => "attention_reduce",
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
            Self::Activation => "activation",
            Self::Residual => "residual",
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
        Kernel::Attention => (Category::AttentionScan, None),
        #[cfg(feature = "metal")]
        Kernel::AttentionGqaDecode | Kernel::AttentionGqaSplit => {
            (Category::AttentionScan, Some(Phase::Decode))
        }
        #[cfg(feature = "metal")]
        Kernel::AttentionGqaReduce => (Category::AttentionReduce, Some(Phase::Decode)),
        #[cfg(feature = "metal")]
        Kernel::AttentionPrefillMatrix => (Category::AttentionScan, Some(Phase::Prefill)),
        Kernel::Embedding => (Category::Embedding, None),
        Kernel::Rmsnorm => (Category::Normalization, None),
        Kernel::Rope => (Category::Rope, None),
        Kernel::KvWrite => (Category::KvCache, None),
        Kernel::SiluMul => (Category::Activation, None),
        Kernel::ResidualAdd => (Category::Residual, None),
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

    #[test]
    fn fused_categories_keep_distinct_measured_boundaries() {
        let mut slot = 0;
        assert_eq!(
            classify(Kernel::AttentionGqaSplit, &[], &mut slot).0.name(),
            "attention_scan"
        );
        assert_eq!(
            classify(Kernel::AttentionGqaReduce, &[], &mut slot)
                .0
                .name(),
            "attention_reduce"
        );
        assert_eq!(
            classify(Kernel::SiluMul, &[], &mut slot).0.name(),
            "activation"
        );
        assert_eq!(
            classify(Kernel::ResidualAdd, &[], &mut slot).0.name(),
            "residual"
        );
    }
}
