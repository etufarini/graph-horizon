/*
 * graph_horizon_engine — Metal causal-attention prefill dispatch
 * Selects matrix, tiled, segmented, or serial geometry from shape, placement,
 * format, and compiled pipeline capabilities.
 */
use super::super::super::{
    MetalBuffer, MetalEncoder,
    exec::dispatch,
    pipeline::{Kernel, PipelineRegistry},
};
use super::constants;
use crate::kv_cache::{Kv, scheme::KvQuant};

const SEGMENTED_CONTEXT: u32 = 512;
const PARALLEL_CONTEXT: u32 = 1024;

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
    let mut c = constants(kv, qh, base, rows, layer);
    #[cfg(feature = "metal")]
    if !mixed_placement
        && rows == 64
        && kv.scheme == KvQuant::F16
        && kv.head_dim == 128
        && kv.kv_heads.checked_mul(4) == Some(qh as usize)
    {
        let matrix = p.get(Kernel::AttentionPrefillMatrix);
        if matrix.width == 32 && matrix.max_threads >= 128 && matrix.threadgroup_memory <= 28 * 1024
        {
            let groups = (rows as usize / 8)
                .checked_mul(qh as usize)
                .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
            c.extend(0_u32.to_ne_bytes());
            return dispatch::encode_threadgroups(
                e,
                p,
                Kernel::AttentionPrefillMatrix,
                &[q, &kv.k, &kv.v, out],
                &c,
                [groups, 1, 1],
                128,
            );
        }
    }
    #[cfg(feature = "metal")]
    if !mixed_placement
        && rows == 1
        && base >= PARALLEL_CONTEXT - 1
        && kv.scheme == KvQuant::F16
        && kv.head_dim == 128
        && kv.kv_heads.checked_mul(4) == Some(qh as usize)
    {
        c.extend(0_u32.to_ne_bytes());
        return dispatch::encode_threadgroups(
            e,
            p,
            Kernel::AttentionGqaDecode,
            &[q, &kv.k, &kv.v, out],
            &c,
            [kv.kv_heads, 1, 1],
            256,
        );
    }
    let width = p.get(Kernel::Attention).width;
    let (mode, grid, group_threads) = geometry(
        mixed_placement,
        rows,
        qh,
        kv.kv_heads as u32,
        kv.head_dim as u32,
        base,
        width,
        kv.scheme == KvQuant::F16,
    )?;
    c.extend(mode.to_ne_bytes());
    let buffers = &[q, &kv.k, &kv.v, out];
    if group_threads == 0 {
        dispatch::encode(e, p, Kernel::Attention, buffers, &c, [grid, 1, 1])
    } else {
        dispatch::encode_threadgroups(
            e,
            p,
            Kernel::Attention,
            buffers,
            &c,
            [grid, 1, 1],
            group_threads,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn geometry(
    mixed_placement: bool,
    rows: u32,
    qh: u32,
    kvh: u32,
    dim: u32,
    base: u32,
    width: usize,
    f16: bool,
) -> color_eyre::eyre::Result<(u32, usize, usize)> {
    let heads = (rows as usize)
        .checked_mul(qh as usize)
        .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
    if mixed_placement {
        Ok((0, heads, 0))
    } else {
        let qualified = dim == 128 && width == 32;
        if f16 && qualified && matches!(rows, 32 | 64) && kvh.checked_mul(4) == Some(qh) {
            let grid = (rows as usize / 4)
                .checked_mul(kvh as usize)
                .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
            return Ok((4, grid, width * 4));
        }
        let parallel = if rows == 1 {
            base >= PARALLEL_CONTEXT - 1
        } else {
            base >= SEGMENTED_CONTEXT
        };
        if qualified && parallel {
            let wide_decode = rows == 1 && f16;
            return Ok((
                if wide_decode { 5 } else { 3 },
                heads,
                width * if wide_decode { 4 } else { 2 },
            ));
        }
        let threads = heads
            .checked_mul(width)
            .ok_or_else(|| color_eyre::eyre::eyre!("metal: buffer arithmetic overflow"))?;
        // Mode 2 retains one SIMD-group per head for the established medium
        // context route; mode 3 widens only qualified long decode threadgroups.
        let segmented = rows == 1 && qualified && base >= SEGMENTED_CONTEXT - 1;
        Ok((if segmented { 2 } else { 1 }, threads, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::geometry;

    #[test]
    fn metal_attention_route_depends_on_effective_placement_not_feature() {
        assert_eq!(
            geometry(false, 2, 4, 1, 128, 511, 32, true).unwrap(),
            (1, 256, 0)
        );
        assert_eq!(
            geometry(false, 2, 4, 1, 128, 512, 32, true).unwrap(),
            (3, 8, 64)
        );
        assert_eq!(
            geometry(false, 32, 4, 1, 128, 0, 32, true).unwrap(),
            (4, 8, 128)
        );
        assert_eq!(
            geometry(false, 64, 4, 1, 128, 0, 32, true).unwrap(),
            (4, 16, 128)
        );
        assert_eq!(
            geometry(false, 32, 4, 1, 128, 512, 32, false).unwrap(),
            (3, 128, 64)
        );
        assert_eq!(
            geometry(false, 1, 4, 1, 128, 510, 32, true).unwrap(),
            (1, 128, 0)
        );
        assert_eq!(
            geometry(false, 1, 4, 1, 128, 511, 32, true).unwrap(),
            (2, 128, 0)
        );
        assert_eq!(
            geometry(false, 1, 4, 1, 128, 1022, 32, true).unwrap(),
            (2, 128, 0)
        );
        assert_eq!(
            geometry(false, 1, 4, 1, 128, 1023, 32, true).unwrap(),
            (5, 4, 128)
        );
        assert_eq!(
            geometry(false, 1, 4, 1, 128, 1023, 32, false).unwrap(),
            (3, 4, 64)
        );
        assert_eq!(
            geometry(false, 1, 4, 1, 64, 1023, 32, true).unwrap(),
            (1, 128, 0)
        );
        assert_eq!(
            geometry(true, 1, 4, 1, 128, 1023, 32, true).unwrap(),
            (0, 4, 0)
        );
        assert!(geometry(false, u32::MAX, u32::MAX, 1, 128, 511, 32, true).is_err());
        assert_eq!(
            geometry(true, u32::MAX, u32::MAX, 1, 128, 1023, 32, true).unwrap(),
            (0, u32::MAX as usize * u32::MAX as usize, 0)
        );
    }
}
