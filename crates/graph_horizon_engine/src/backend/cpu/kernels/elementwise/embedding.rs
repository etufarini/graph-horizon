/*
 * graph_horizon_engine — CPU embedding kernel
 * Decodes one quantized embedding row into the FP32 residual stream without
 * owning dispatch or storage lifecycle.
 */

// AGENTS deroga K: singolo kernel numerico di lookup embedding.

use crate::backend::cpu::buffer::CpuBuffer;
use crate::backend::cpu::dequant;

pub(crate) fn embed(x: &CpuBuffer, token_embd: &CpuBuffer, token: usize, embd: usize) {
    let mut row = vec![0f32; embd];
    // Slice the guard to the buffer's window so a sub-view reads only its bytes.
    let bytes = token_embd.bytes();
    dequant::dequant_row(
        token_embd.format,
        &bytes[token_embd.window()],
        token,
        embd,
        &mut row,
    );
    x.write_f32(&row);
}
