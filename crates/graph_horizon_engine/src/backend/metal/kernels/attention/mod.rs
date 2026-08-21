/*
 * graph_horizon_engine — Metal causal-attention module boundary
 * Exposes separate decode and prefill dispatch policies and owns only their
 * shared immutable parameter encoding.
 */
mod decode;
mod prefill;

pub(crate) use decode::encode_decode;
pub(crate) use prefill::encode;

use super::super::MetalBuffer;
use crate::kv_cache::{
    Kv,
    scheme::{KvQuant, KvRole},
};

fn constants(
    kv: &Kv<MetalBuffer>,
    qh: u32,
    base: u32,
    rows: u32,
    layer: u32,
    q32: bool,
) -> Vec<u8> {
    let mut bytes = super::u32s(&[
        kv.head_dim as u32,
        kv.kv_heads as u32,
        qh,
        base,
        rows,
        layer,
        kv.context as u32,
        u32::from(kv.scheme == KvQuant::Int8),
    ]);
    bytes.extend(kv.meta_base_for(KvRole::Key).to_ne_bytes());
    bytes.extend(kv.meta_base_for(KvRole::Value).to_ne_bytes());
    bytes.extend((1.0f32 / (kv.head_dim as f32).sqrt()).to_ne_bytes());
    bytes.extend(u32::from(q32).to_ne_bytes());
    bytes
}
