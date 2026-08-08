/*
 * Vulkan pipeline specification: maps every reachable kernel to its SPIR-V module and ABI without owning device state or resources.
 */
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    MatmulF16,
    MatmulQ4KTiled,
    MatmulQ5K,
    MatmulQ6K,
    Logits,
    LogitsQ4K,
    LogitsQ5K,
    LogitsQ6K,
    EmbedF16,
    EmbedQ4K,
    EmbedQ5K,
    EmbedQ6K,
    RmsNormX,
    Rope,
    Residual,
    KvWrite,
    AttentionDecode,
    AttentionDecodeWide,
    AttentionDecode1024,
    AttentionPrefill,
    AttentionPrefillWide,
    KvWriteInt8,
    AttentionDecodeInt8,
    AttentionPrefillInt8,
    Argmax,
    TopkPartial,
    SiluMul,
    MatmulQ4KBatchF16Out,
    MatmulQ6KBatchF16Out,
    MatmulQ4KCoopmatF16Out,
    QuantAQ8F16,
    MatmulQ4KMmvqF16Out,
}

#[cfg(feature = "vulkan-profile")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileCategory {
    Attention,
    ProjectionQkv,
    ProjectionOutput,
    Mlp,
    Matmul,
    Normalization,
    Rope,
    KvCache,
    Elementwise,
    Embedding,
    Logits,
    Reduction,
}

#[cfg(feature = "vulkan-profile")]
impl ProfileCategory {
    pub(crate) const COUNT: usize = 12;
    pub(crate) const ALL: [Self; Self::COUNT] = [
        Self::Attention,
        Self::ProjectionQkv,
        Self::ProjectionOutput,
        Self::Mlp,
        Self::Matmul,
        Self::Normalization,
        Self::Rope,
        Self::KvCache,
        Self::Elementwise,
        Self::Embedding,
        Self::Logits,
        Self::Reduction,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Attention => "attention",
            Self::ProjectionQkv => "projection_qkv",
            Self::ProjectionOutput => "projection_output",
            Self::Mlp => "mlp",
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

#[cfg(feature = "vulkan-profile")]
impl Kernel {
    const fn profile_category(self) -> ProfileCategory {
        match self {
            Self::AttentionDecode
            | Self::AttentionDecodeWide
            | Self::AttentionDecode1024
            | Self::AttentionPrefill
            | Self::AttentionPrefillWide
            | Self::AttentionDecodeInt8
            | Self::AttentionPrefillInt8 => ProfileCategory::Attention,
            Self::MatmulF16
            | Self::MatmulQ4KTiled
            | Self::MatmulQ5K
            | Self::MatmulQ6K
            | Self::MatmulQ4KBatchF16Out
            | Self::MatmulQ6KBatchF16Out
            | Self::MatmulQ4KCoopmatF16Out
            | Self::QuantAQ8F16
            | Self::MatmulQ4KMmvqF16Out => ProfileCategory::Matmul,
            Self::Logits | Self::LogitsQ4K | Self::LogitsQ5K | Self::LogitsQ6K => {
                ProfileCategory::Logits
            }
            Self::EmbedF16 | Self::EmbedQ4K | Self::EmbedQ5K | Self::EmbedQ6K => {
                ProfileCategory::Embedding
            }
            Self::RmsNormX => ProfileCategory::Normalization,
            Self::Rope => ProfileCategory::Rope,
            Self::Residual | Self::SiluMul => ProfileCategory::Elementwise,
            Self::KvWrite | Self::KvWriteInt8 => ProfileCategory::KvCache,
            Self::Argmax | Self::TopkPartial => ProfileCategory::Reduction,
        }
    }

    pub(crate) const fn is_batched_matmul(self) -> bool {
        matches!(
            self,
            Self::MatmulQ4KBatchF16Out | Self::MatmulQ6KBatchF16Out | Self::MatmulQ4KCoopmatF16Out
        )
    }

    pub(crate) const fn is_prefill_attention(self) -> bool {
        matches!(
            self,
            Self::AttentionPrefill | Self::AttentionPrefillWide | Self::AttentionPrefillInt8
        )
    }

    pub(crate) fn profiled_category(self, batched: &mut usize) -> ProfileCategory {
        if !self.is_batched_matmul() {
            return self.profile_category();
        }
        // Every prefill layer records Q, K, V, output, gate, up, then down;
        // preserving that seven-dispatch order is the category invariant.
        let category = match *batched % 7 {
            0..=2 => ProfileCategory::ProjectionQkv,
            3 => ProfileCategory::ProjectionOutput,
            _ => ProfileCategory::Mlp,
        };
        *batched += 1;
        category
    }
}

// SPIR-V bytes, storage-buffer binding count, and push-constant bytes.
pub(super) fn spec(kernel: Kernel) -> (&'static [u8], u32, u32) {
    macro_rules! spv {
        ($name:literal) => {
            include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
        };
    }
    match kernel {
        Kernel::MatmulF16 => (spv!("matmul_fp16"), 3, 8),
        Kernel::MatmulQ4KTiled => (spv!("matmul_tiled"), 3, 8),
        Kernel::MatmulQ5K => (spv!("matmul_q5_k"), 3, 8),
        Kernel::MatmulQ6K => (spv!("matmul_q6_k"), 3, 8),
        Kernel::Logits => (spv!("logits"), 3, 8),
        Kernel::LogitsQ4K => (spv!("logits_q4_k"), 3, 8),
        Kernel::LogitsQ5K => (spv!("logits_q5_k"), 3, 8),
        Kernel::LogitsQ6K => (spv!("logits_q6_k"), 3, 8),
        Kernel::EmbedF16 => (spv!("embed_f16"), 2, 8),
        Kernel::EmbedQ4K => (spv!("embed_q4_k"), 2, 8),
        Kernel::EmbedQ5K => (spv!("embed_q5_k"), 2, 8),
        Kernel::EmbedQ6K => (spv!("embed_q6_k"), 2, 8),
        Kernel::RmsNormX => (spv!("rmsnorm_x"), 3, 8),
        Kernel::Rope => (spv!("rope"), 1, 40),
        Kernel::Residual => (spv!("residual"), 2, 4),
        Kernel::KvWrite => (spv!("kv_write"), 4, 8),
        Kernel::AttentionDecode => (spv!("attention_decode"), 4, 32),
        Kernel::AttentionDecodeWide => (spv!("attention_decode_wide"), 4, 32),
        Kernel::AttentionDecode1024 => (spv!("attention_decode_1024"), 4, 32),
        Kernel::AttentionPrefill => (spv!("attention_prefill"), 4, 32),
        Kernel::AttentionPrefillWide => (spv!("attention_prefill_wide"), 4, 32),
        Kernel::KvWriteInt8 => (spv!("kv_write_int8"), 4, 16),
        Kernel::AttentionDecodeInt8 => (spv!("attention_decode_int8"), 4, 32),
        Kernel::AttentionPrefillInt8 => (spv!("attention_prefill_int8"), 4, 36),
        Kernel::Argmax => (spv!("argmax"), 2, 4),
        Kernel::TopkPartial => (spv!("topk_partial"), 2, 8),
        Kernel::SiluMul => (spv!("silu_mul"), 3, 4),
        Kernel::MatmulQ4KBatchF16Out => (spv!("matmul_q4_k_batch_f16"), 3, 12),
        Kernel::MatmulQ6KBatchF16Out => (spv!("matmul_q6_k_batch_f16"), 3, 12),
        Kernel::MatmulQ4KCoopmatF16Out => (spv!("matmul_q4_k_coopmat_f16out"), 3, 12),
        Kernel::QuantAQ8F16 => (spv!("quant_a_q8_f16"), 3, 4),
        Kernel::MatmulQ4KMmvqF16Out => (spv!("matmul_q4_k_mmvq_f16out"), 4, 8),
    }
}
