/*
 * graph_horizon_engine — hybrid device contract
 * Defines the single extension trait a GPU backend implements for generic
 * placement, selected loading, and one host-residual upload. It owns no family
 * graph, tokenizer, sampling, placement arithmetic, or request lifecycle.
 */

// AGENTS deroga I: definizione del solo trait HybridDevice.

use color_eyre::eyre::Result;

use crate::backend::Backend;
use crate::backend::hybrid::placement::{BudgetInput, MemoryTopology};
use crate::backend::hybrid::weights::runtime::{DeviceFixedBytes, RuntimeShape};
use crate::backend::source::{WeightSelection, WeightSource};
use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::ModelMetadata;

pub(crate) trait HybridDevice: Backend {
    type Device;

    fn host_available() -> Result<u64>;
    fn acquire() -> Result<Option<Self::Device>>;
    fn budget(device: &Self::Device) -> Result<BudgetInput>;
    fn topology() -> MemoryTopology;
    fn all_mode_name() -> &'static str;
    fn invalid_percentage_error() -> &'static str;
    fn fixed_bytes(shape: &RuntimeShape) -> Result<DeviceFixedBytes>;
    fn load_selected(
        device: Self::Device,
        meta: &ModelMetadata,
        source: &dyn WeightSource,
        gguf: &GgufFile,
        selection: &WeightSelection,
    ) -> Result<Self>;
    fn buffer_bytes(buffer: &Self::Buffer) -> u64;
    fn upload_residual(&self, target: &Self::Buffer, bytes: &[u8]) -> Result<()>;
}
