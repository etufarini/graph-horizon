/*
 * graph_horizon_engine — Metal causal attention dispatch
 * Binds decode/prefill geometry and selects F16 or int8 KV once per operation.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder,
    pipeline::{Kernel, PipelineRegistry},
};
use crate::kv_cache::{
    Kv,
    scheme::{KvQuant, KvRole},
};
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    out: &MetalBuffer,
    q: &MetalBuffer,
    kv: &Kv<MetalBuffer>,
    qh: u32,
    base: u32,
    rows: u32,
    layer: u32,
) -> color_eyre::eyre::Result<()> {
    let mut c = super::u32s(&[
        kv.head_dim as u32,
        kv.kv_heads as u32,
        qh,
        base,
        rows,
        layer,
        kv.context as u32,
        u32::from(kv.scheme == KvQuant::Int8),
    ]);
    for n in [
        kv.meta_base_for(KvRole::Key),
        kv.meta_base_for(KvRole::Value),
    ] {
        c.extend(n.to_ne_bytes());
    }
    c.extend((1.0f32 / (kv.head_dim as f32).sqrt()).to_ne_bytes());
    c.extend(u32::from(!cfg!(feature = "metal-hybrid")).to_ne_bytes());
    let width = p.get(Kernel::Attention).width;
    let heads = (rows as usize)
        .checked_mul(qh as usize)
        .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
    let threads = if cfg!(feature = "metal-hybrid") {
        heads
    } else {
        heads
            .checked_mul(width)
            .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?
    };
    dispatch::encode(
        e,
        p,
        Kernel::Attention,
        &[q, &kv.k, &kv.v, out],
        &c,
        [threads, 1, 1],
    )
}
