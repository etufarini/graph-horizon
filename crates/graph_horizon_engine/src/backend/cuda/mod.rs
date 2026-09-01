/*
 * graph_horizon_engine — standalone CUDA backend namespace and owned state.
 * The backend owns one context and stream; child modules own loading, memory,
 * execution, and operation dispatch without exposing CUDA outside this tree.
 */

#![deny(clippy::undocumented_unsafe_blocks)]

mod backend;
mod device;
mod exec;
mod kernels;
mod loader;
mod mem;
mod module;

pub(crate) use device::Device;
pub(crate) use exec::encoder::CudaEncoder;
pub(crate) use loader::load;
pub(crate) use mem::buffer::{CudaBuffer, CudaFormat};

pub(crate) const PREFILL_ROWS: usize = 32;

pub(crate) struct CudaBackend {
    pub(crate) device: Device,
    pub(crate) module: module::Module,
    pub(crate) buffers: crate::backend::buffers::Buffers<CudaBuffer>,
    pub(crate) reduce: CudaBuffer,
}
