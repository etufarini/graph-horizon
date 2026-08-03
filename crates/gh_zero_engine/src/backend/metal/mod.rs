/*
 * gh_zero_engine — native Metal backend namespace
 * Wires the statically selected Metal implementation on qualified Apple hosts.
 * Resource ownership, execution, memory, and kernels live in focused children.
 */

#![deny(clippy::undocumented_unsafe_blocks)]

mod backend;
mod device;
mod exec;
mod loader;
mod mem;
mod pipeline;

pub(crate) use device::Device;
pub(crate) use exec::encoder::MetalEncoder;
pub(crate) use loader::load;
pub(crate) use mem::buffer::{MetalBuffer, MetalFormat};

use crate::backend::buffers::Buffers;

pub(crate) struct MetalBackend {
    pub(crate) device: Device,
    pub(crate) pipelines: pipeline::PipelineRegistry,
    pub(crate) buffers: Buffers<MetalBuffer>,
    pub(crate) reduce: MetalBuffer,
    pub(crate) staging: MetalBuffer,
}
