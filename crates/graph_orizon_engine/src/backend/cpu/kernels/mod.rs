/*
 * graph_orizon_engine — CPU compute kernels
 * No logic here: just the collection of the kernel submodules. The common-path
 * kernels (matmul/logits and the per-element ops) live in `matmul` and
 * `elementwise`; the attention path (attention_decode, kv_write) lives in
 * `attention`, with its AVX2+F16C inner kernels in `attention::simd`.
*/

pub(super) mod attention;
pub(super) mod elementwise;
pub(super) mod matmul;
