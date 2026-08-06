/*
 * graph_horizon_engine — CPU elementwise kernel surface
 * Exposes the independent CPU elementwise operations to the backend while each
 * numeric kernel remains owned by its single-purpose module.
 */

mod activation;
mod embedding;
mod normalization;
mod residual;
mod rope;

pub(crate) use activation::silu_mul;
pub(crate) use embedding::embed;
pub(crate) use normalization::rmsnorm_x;
pub(crate) use residual::residual_add;
pub(crate) use rope::rope_yarn;
