/*
 * gh_zero_engine — bounded Metal dispatch encoding
 * Validates grids and buffer windows, binds one immutable pipeline plus scalar
 * bytes, and derives legal threadgroup geometry. It owns no submission or math.
 */

use std::ffi::c_void;
use std::ptr::NonNull;

use color_eyre::eyre::{Result, eyre};
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLSize};

use super::encoder::MetalEncoder;
use crate::backend::metal::mem::buffer::MetalBuffer;
use crate::backend::metal::pipeline::{Kernel, PipelineRegistry};

pub(crate) fn encode(
    encoder: &MetalEncoder,
    registry: &PipelineRegistry,
    kernel: Kernel,
    buffers: &[&MetalBuffer],
    constants: &[u8],
    grid: [usize; 3],
) -> Result<()> {
    if grid.contains(&0) {
        return Err(arithmetic());
    }
    let compute = encoder.compute()?;
    let pipeline = registry.get(kernel);
    if pipeline.width == 0 || pipeline.max_threads == 0 {
        return Err(arithmetic());
    }
    compute.setComputePipelineState(&pipeline.raw);
    for (index, buffer) in buffers.iter().enumerate() {
        let end = buffer
            .offset()
            .checked_add(buffer.len())
            .ok_or_else(arithmetic)?;
        if end > buffer.raw().length() {
            return Err(arithmetic());
        }
        // SAFETY: the checked logical window fits the retained MTLBuffer and
        // index derives from the bounded caller-owned binding slice.
        unsafe {
            compute.setBuffer_offset_atIndex(Some(buffer.raw()), buffer.offset(), index);
        }
    }
    if !constants.is_empty() {
        let pointer =
            NonNull::new(constants.as_ptr().cast_mut().cast::<c_void>()).ok_or_else(arithmetic)?;
        // SAFETY: `constants` remains alive for this synchronous encoding call;
        // Metal copies setBytes data and the buffer length is exact.
        unsafe {
            compute.setBytes_length_atIndex(pointer, constants.len(), buffers.len());
        }
    }
    let threads = pipeline.width.min(pipeline.max_threads);
    compute.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: grid[0],
            height: grid[1],
            depth: grid[2],
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn arithmetic() -> color_eyre::Report {
    eyre!("metal: buffer arithmetic overflow")
}
