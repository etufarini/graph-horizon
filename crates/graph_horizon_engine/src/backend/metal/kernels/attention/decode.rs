/*
 * graph_horizon_engine — Metal causal-attention decode dispatch
 * Selects the qualified grouped split/reduce route and preserves the generic
 * attention fallback without owning pipelines or buffers.
 */
use super::super::super::{MetalBuffer, MetalEncoder, pipeline::PipelineRegistry};
#[cfg(feature = "metal")]
use super::super::super::{exec::dispatch, pipeline::Kernel};
#[cfg(feature = "metal")]
use super::constants;
use super::prefill;
use crate::kv_cache::Kv;
#[cfg(feature = "metal")]
use crate::kv_cache::scheme::KvQuant;

#[cfg(feature = "metal")]
const PARALLEL_CONTEXT: u32 = 1024;
#[cfg(feature = "metal")]
const GQA_PARTS: u64 = 8;

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_decode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    out: &MetalBuffer,
    q: &MetalBuffer,
    kv: &Kv<MetalBuffer>,
    qh: u32,
    base: u32,
    layer: u32,
    mixed_placement: bool,
    scratch: &MetalBuffer,
) -> color_eyre::eyre::Result<()> {
    #[cfg(feature = "metal")]
    {
        let split = p.get(Kernel::AttentionGqaSplit);
        let reduce = p.get(Kernel::AttentionGqaReduce);
        let qualified = !mixed_placement
            && base >= PARALLEL_CONTEXT - 1
            && kv.scheme == KvQuant::F16
            && kv.head_dim == 128
            && kv.kv_heads.checked_mul(4) == Some(qh as usize)
            && split.width == 32
            && split.max_threads >= 256
            && split.threadgroup_memory == 16 * 1024
            && reduce.width == 32
            && reduce.max_threads >= 32
            && reduce.threadgroup_memory == 0;
        let partial_bytes = u64::from(qh)
            .checked_mul(GQA_PARTS * 128 * 4)
            .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
        let state_bytes = u64::from(qh)
            .checked_mul(GQA_PARTS * 2 * 4)
            .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
        let total_bytes = partial_bytes
            .checked_add(state_bytes)
            .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
        if qualified && total_bytes <= scratch.len() as u64 {
            let partial = scratch.view(0, partial_bytes)?;
            let state = scratch.view(partial_bytes, state_bytes)?;
            let mut c = constants(kv, qh, base, 1, layer);
            c.extend(0_u32.to_ne_bytes());
            dispatch::encode_threadgroups(
                e,
                p,
                Kernel::AttentionGqaSplit,
                &[q, &kv.k, &kv.v, &partial, &state],
                &c,
                [kv.kv_heads, 4, 1],
                256,
            )?;
            return dispatch::encode_threadgroups(
                e,
                p,
                Kernel::AttentionGqaReduce,
                &[&partial, &state, out],
                &c,
                [qh as usize, 1, 1],
                32,
            );
        }
    }
    #[cfg(not(feature = "metal"))]
    let _ = scratch;
    prefill::encode(e, p, out, q, kv, qh, base, 1, layer, mixed_placement)
}
