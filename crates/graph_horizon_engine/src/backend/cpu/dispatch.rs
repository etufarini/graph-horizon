/*
 * graph_horizon_engine — CPU backend GEMM-family compute dispatch
 * The seam between the `Backend` trait and the matmul kernel module for the three
 * GEMM-family ops (`matmul`, `matmul_batched`, `logits`): the trait methods in `mod.rs`
 * delegate here, and these forward to `kernels::matmul`, which selects the per-CpuFormat
 * kernel (Q4_K/Q5_K/Q6_K fused, else the generic F16 path). Every CpuFormat
 * is covered there, so no weight ever reaches a kernel for the wrong format. Only the
 * u32→usize narrowing lives here; no numeric work. Moved 1:1 from the former `mod.rs`.
*/

use super::buffer::CpuBuffer;
use super::kernels;

// y = W·a, FP16 out. Routed by weight format inside `kernels::matmul::matmul`.
pub(super) fn matmul(out: &CpuBuffer, a: &CpuBuffer, w: &CpuBuffer, in_dim: u32, out_dim: u32) {
    kernels::matmul::matmul(out, a, w, in_dim as usize, out_dim as usize);
}

// Batched prefill matmul: dequant each weight row once, reuse across the N prompt
// tokens (the per-token `matmul` re-reads the weights N times).
pub(super) fn matmul_batched(
    out: &CpuBuffer,
    a: &CpuBuffer,
    w: &CpuBuffer,
    in_dim: u32,
    out_dim: u32,
    n: u32,
) {
    kernels::matmul::matmul_batched(out, a, w, in_dim as usize, out_dim as usize, n as usize);
}

// Same as `matmul` but the output is the FP32 vocab logits (no FP16 narrowing).
pub(super) fn logits(out: &CpuBuffer, x: &CpuBuffer, w: &CpuBuffer, in_dim: u32, out_dim: u32) {
    kernels::matmul::logits(out, x, w, in_dim as usize, out_dim as usize);
}
