/*
 * graph_horizon_engine — CUDA single-token causal-attention dispatch.
 */

use color_eyre::eyre::Result;

use super::super::super::exec::dispatch;
use super::super::super::module::{Kernel, Module};
use super::super::super::{CudaBuffer, CudaEncoder};
use crate::kv_cache::Kv;
use crate::kv_cache::scheme::KvQuant;

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    encoder: &CudaEncoder,
    module: &Module,
    out: &CudaBuffer,
    query: &CudaBuffer,
    kv: &Kv<CudaBuffer>,
    q_heads: u32,
    position: u32,
    layer: u32,
) -> Result<()> {
    let args = super::validate(out, query, kv, q_heads, position, 1, layer)?;
    let kernel = match kv.scheme {
        KvQuant::F16 => Kernel::AttentionDecodeF16,
        KvQuant::Int8 => Kernel::AttentionDecodeInt8,
    };
    // Four warps score a context tile; each thread owns up to two dimensions.
    dispatch::launch(encoder, module, kernel, &args, (q_heads, 1, 1), (128, 1, 1))
}
