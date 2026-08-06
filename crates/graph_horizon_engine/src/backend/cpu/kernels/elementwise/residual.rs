/*
 * graph_horizon_engine — CPU residual-add kernel
 * Adds an FP16 branch into the FP32 residual stream in place, preserving the
 * caller-owned buffer and range boundaries.
 */

// AGENTS deroga K: singolo kernel numerico di somma residuale.

use crate::backend::cpu::buffer::CpuBuffer;
use crate::backend::cpu::parallel;

pub(crate) fn residual_add(x: &CpuBuffer, y: &CpuBuffer, n: usize) {
    let mut residual = x.read_f32();
    let branch = y.read_f16_as_f32();
    // Only the caller-selected prefix is mutated; `start` is an absolute index.
    parallel::for_units(&mut residual[..n], 1, |start, chunk| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            *value += branch[start + offset];
        }
    });
    x.write_f32(&residual);
}
