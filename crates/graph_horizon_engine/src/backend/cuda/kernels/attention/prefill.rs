/*
 * graph_horizon_engine — CUDA batched causal-attention dispatch.
 */

use color_eyre::eyre::Result;

use super::super::super::exec::dispatch::{self, Arg};
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
    base: u32,
    rows: u32,
    layer: u32,
) -> Result<()> {
    let mut args = super::validate(out, query, kv, q_heads, base, rows, layer)?;
    // The prefill entry always owns a rows scalar, including a one-row tail.
    args.insert(8, Arg::U32(rows));
    // Isolate fixed128 F16 matrix staging from short histories and every fallback.
    let tensor = kv.scheme == KvQuant::F16 && kv.head_dim == 128 && rows >= 4 && base >= 512;
    let kernel = match kv.scheme {
        KvQuant::F16 if tensor => Kernel::AttentionPrefillTensorF16,
        KvQuant::F16 => Kernel::AttentionPrefillF16,
        KvQuant::Int8 => Kernel::AttentionPrefillInt8,
    };
    let total = u64::from(rows)
        .checked_mul(u64::from(q_heads))
        .ok_or_else(super::super::arithmetic)?;
    let total = u32::try_from(total).map_err(|_| super::super::arithmetic())?;
    let groups = if tensor {
        rows.div_ceil(8)
            .checked_mul(q_heads)
            .ok_or_else(super::super::arithmetic)?
    } else {
        total.div_ceil(4)
    };
    // Tensor blocks own eight queries of one head; fallback warps own independent heads.
    dispatch::launch(encoder, module, kernel, &args, (groups, 1, 1), (128, 1, 1))
}
