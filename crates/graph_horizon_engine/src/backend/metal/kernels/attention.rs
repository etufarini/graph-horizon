/*
 * graph_horizon_engine — Metal causal attention dispatch
 * Binds decode/prefill geometry and selects F16 or int8 KV once per operation.
 * Features control availability; shape, context and effective placement select
 * serial, per-head SIMD or long-context segmented execution.
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

const SEGMENTED_CONTEXT: u32 = 512;

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
    mixed_placement: bool,
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
    let width = p.get(Kernel::Attention).width;
    let (mode, threads) = geometry(mixed_placement, rows, qh, kv.head_dim as u32, base, width)?;
    c.extend(mode.to_ne_bytes());
    dispatch::encode(
        e,
        p,
        Kernel::Attention,
        &[q, &kv.k, &kv.v, out],
        &c,
        [threads, 1, 1],
    )
}

fn geometry(
    mixed_placement: bool,
    rows: u32,
    qh: u32,
    dim: u32,
    base: u32,
    width: usize,
) -> color_eyre::eyre::Result<(u32, usize)> {
    let heads = (rows as usize)
        .checked_mul(qh as usize)
        .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
    if mixed_placement {
        Ok((0, heads))
    } else {
        let threads = heads
            .checked_mul(width)
            .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
        // Mode 2 is intentionally narrow: four eight-lane segments retain one
        // dispatch only for the qualified decode shape and long KV scans.
        let segmented = rows == 1 && dim == 128 && width == 32 && base >= SEGMENTED_CONTEXT - 1;
        Ok((if segmented { 2 } else { 1 }, threads))
    }
}

#[cfg(test)]
mod tests {
    use super::geometry;

    #[test]
    fn metal_attention_route_depends_on_effective_placement_not_feature() {
        assert_eq!(geometry(false, 2, 4, 128, 511, 32).unwrap(), (1, 256));
        assert_eq!(geometry(false, 1, 4, 128, 510, 32).unwrap(), (1, 128));
        assert_eq!(geometry(false, 1, 4, 128, 511, 32).unwrap(), (2, 128));
        assert_eq!(geometry(false, 1, 4, 64, 511, 32).unwrap(), (1, 128));
        assert_eq!(geometry(true, 1, 4, 128, 511, 32).unwrap(), (0, 4));
        assert!(geometry(false, u32::MAX, u32::MAX, 128, 511, 32).is_err());
        assert_eq!(
            geometry(true, u32::MAX, u32::MAX, 128, 511, 32).unwrap(),
            (0, u32::MAX as usize * u32::MAX as usize)
        );
    }
}
