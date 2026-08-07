/*
 * graph_horizon_engine — Vulkan pipeline specification table
 * Defines the exhaustive reachable kernel set and maps each variant directly
 * to one compiled SPIR-V module, storage-binding count, and push-constant size.
 * It owns no device state, dispatch, or resource lifecycle.
 */

// One variant maps to one retained shader and one pipeline.
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
