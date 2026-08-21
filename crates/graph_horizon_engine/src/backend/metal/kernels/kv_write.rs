/*
 * graph_horizon_engine — Metal KV-write dispatch
 * Binds checked payload/metadata windows and selects F16 or int8 encoding once.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder,
    pipeline::{Kernel, PipelineRegistry},
};
use crate::kv_cache::{Kv, scheme::KvQuant};
use color_eyre::eyre::{Result, eyre};
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    kv: &Kv<MetalBuffer>,
    k: &MetalBuffer,
    v: &MetalBuffer,
    kp: u64,
    vp: u64,
    km: u64,
    vm: u64,
    vectors: u32,
) -> Result<()> {
    if kv.head_dim != kv.value_dim {
        return Err(eyre!("metal: buffer arithmetic overflow"));
    }
    let mut c = Vec::new();
    for n in [kp, vp, km, vm] {
        c.extend(n.to_ne_bytes());
    }
    c.extend(super::u32s(&[
        vectors,
        kv.head_dim as u32,
        u32::from(kv.scheme == KvQuant::Int8),
        u32::from(
            k.len()
                == (vectors as usize)
                    .checked_mul(kv.head_dim)
                    .and_then(|items| items.checked_mul(4))
                    .ok_or_else(|| eyre!("metal: buffer arithmetic overflow"))?,
        ),
    ]));
    dispatch::encode(
        e,
        p,
        Kernel::KvWrite,
        &[k, v, &kv.k, &kv.v],
        &c,
        [vectors as usize, 1, 1],
    )
}
