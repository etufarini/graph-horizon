/*
 * Vulkan profiler classification: assigns commands to inference phases and
 * preserves the fixed seven-matmul order used to separate projections from
 * MLP work. It owns no timestamps, resources, or output formatting.
 */

use crate::backend::vulkan::pipeline::Kernel;

#[derive(Clone, Copy, PartialEq, Eq)]
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
pub(super) enum Category {
    Attention,
    ProjectionQ,
    ProjectionK,
    ProjectionV,
    ProjectionOutput,
    MlpGate,
    MlpUp,
    MlpActivation,
    MlpDown,
    Matmul,
    Normalization,
    Rope,
    KvCache,
    Elementwise,
    Embedding,
    Logits,
    Reduction,
}

impl Category {
    pub(super) const COUNT: usize = 17;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::Attention,
        Self::ProjectionQ,
        Self::ProjectionK,
        Self::ProjectionV,
        Self::ProjectionOutput,
        Self::MlpGate,
        Self::MlpUp,
        Self::MlpActivation,
        Self::MlpDown,
        Self::Matmul,
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
            Self::MlpActivation => "mlp_activation",
            Self::MlpDown => "mlp_down",
            Self::Matmul => "matmul_other",
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

const fn direct(kernel: Kernel) -> Category {
    match kernel {
        Kernel::AttentionDecode
        | Kernel::AttentionDecodeWide
        | Kernel::AttentionDecode1024
        | Kernel::AttentionDecodeGqaSplit
        | Kernel::AttentionDecodeGqaReduce
        | Kernel::AttentionPrefill
        | Kernel::AttentionPrefillWide
        | Kernel::AttentionPrefillTiled
        | Kernel::AttentionPrefillTiledCoopQk
        | Kernel::AttentionPrefillMatrix2
        | Kernel::AttentionDecodeInt8
        | Kernel::AttentionPrefillInt8 => Category::Attention,
        Kernel::MatmulF16
        | Kernel::MatmulQ4KTiled
        | Kernel::MatmulQ5K
        | Kernel::MatmulQ6K
        | Kernel::MatmulQ4KBatchF16Out
        | Kernel::MatmulQ6KBatchF16Out
        | Kernel::MatmulQ4KCoopmatF16Out
        | Kernel::MatmulQ4KCoopmatMetadataF16Out
        | Kernel::MatmulQ4KMatrix2F16Out
        | Kernel::MatmulQ6KCoopmatF16Out
        | Kernel::MatmulQ6KMatrix2F16Out
        | Kernel::QuantAQ8F16
        | Kernel::MatmulQ4KMmvqF16Out => Category::Matmul,
        Kernel::Logits | Kernel::LogitsQ4K | Kernel::LogitsQ5K | Kernel::LogitsQ6K => {
            Category::Logits
        }
        Kernel::EmbedF16 | Kernel::EmbedQ4K | Kernel::EmbedQ5K | Kernel::EmbedQ6K => {
            Category::Embedding
        }
        Kernel::RmsNormX => Category::Normalization,
        Kernel::Rope => Category::Rope,
        Kernel::SiluMul => Category::MlpActivation,
        Kernel::Residual => Category::Elementwise,
        Kernel::KvWrite | Kernel::KvWriteInt8 => Category::KvCache,
        Kernel::Argmax | Kernel::TopkPartial => Category::Reduction,
    }
}

const fn is_projection_matmul(kernel: Kernel) -> bool {
    matches!(
        kernel,
        Kernel::MatmulF16
            | Kernel::MatmulQ4KTiled
            | Kernel::MatmulQ5K
            | Kernel::MatmulQ6K
            | Kernel::MatmulQ4KBatchF16Out
            | Kernel::MatmulQ6KBatchF16Out
            | Kernel::MatmulQ4KCoopmatF16Out
            | Kernel::MatmulQ4KCoopmatMetadataF16Out
            | Kernel::MatmulQ4KMatrix2F16Out
            | Kernel::MatmulQ6KCoopmatF16Out
            | Kernel::MatmulQ6KMatrix2F16Out
            | Kernel::MatmulQ4KMmvqF16Out
    )
}

pub(super) const fn phase(kernel: Kernel) -> Option<Phase> {
    match kernel {
        Kernel::AttentionPrefill
        | Kernel::AttentionPrefillWide
        | Kernel::AttentionPrefillTiled
        | Kernel::AttentionPrefillTiledCoopQk
        | Kernel::AttentionPrefillMatrix2
        | Kernel::AttentionPrefillInt8 => Some(Phase::Prefill),
        Kernel::AttentionDecode
        | Kernel::AttentionDecodeWide
        | Kernel::AttentionDecode1024
        | Kernel::AttentionDecodeGqaSplit
        | Kernel::AttentionDecodeGqaReduce
        | Kernel::AttentionDecodeInt8 => Some(Phase::Decode),
        Kernel::Argmax | Kernel::TopkPartial => Some(Phase::Sampling),
        _ => None,
    }
}

pub(super) fn profiled(kernel: Kernel, matmul_slot: &mut usize) -> Category {
    if !is_projection_matmul(kernel) {
        return direct(kernel);
    }
    // Every dense layer records Q, K, V, output, gate, up, then down. This
    // invariant is shared by the batched prefill and single-row decode graphs.
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
    category
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_sequence_splits_attention_projections_and_mlp() {
        let mut slot = 0;
        let names = (0..7)
            .map(|_| profiled(Kernel::MatmulQ4KBatchF16Out, &mut slot).name())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "projection_q",
                "projection_k",
                "projection_v",
                "projection_output",
                "mlp_gate",
                "mlp_up",
                "mlp_down"
            ]
        );

        let mut slot = 0;
        let names = (0..7)
            .map(|_| profiled(Kernel::MatmulQ4KTiled, &mut slot).name())
            .collect::<Vec<_>>();
        assert_eq!(names[0], "projection_q");
        assert_eq!(names[3], "projection_output");
        assert_eq!(names[6], "mlp_down");
    }

    #[test]
    fn attention_and_reduction_select_command_phase() {
        assert!(matches!(
            phase(Kernel::AttentionPrefillWide),
            Some(Phase::Prefill)
        ));
        assert!(matches!(
            phase(Kernel::AttentionPrefillTiledCoopQk),
            Some(Phase::Prefill)
        ));
        assert!(matches!(
            phase(Kernel::AttentionPrefillMatrix2),
            Some(Phase::Prefill)
        ));
        assert!(matches!(
            phase(Kernel::AttentionDecode1024),
            Some(Phase::Decode)
        ));
        assert!(matches!(
            phase(Kernel::AttentionDecodeGqaSplit),
            Some(Phase::Decode)
        ));
        assert!(matches!(
            phase(Kernel::AttentionDecodeGqaReduce),
            Some(Phase::Decode)
        ));
        assert!(matches!(phase(Kernel::Argmax), Some(Phase::Sampling)));
        assert!(phase(Kernel::Rope).is_none());
    }

    #[test]
    fn fused_silu_multiply_is_mlp_activation() {
        let mut slot = 0;
        assert_eq!(
            profiled(Kernel::SiluMul, &mut slot).name(),
            "mlp_activation"
        );
        assert_eq!(profiled(Kernel::Residual, &mut slot).name(), "elementwise");
        assert_eq!(slot, 0);
    }
}
