/*
 * graph_horizon_engine — CUDA operation wiring and shared span arithmetic.
 */

pub(crate) mod argmax;
pub(crate) mod attention;
pub(crate) mod embedding;
pub(crate) mod kv_write;
pub(crate) mod matmul;
pub(crate) mod normalization;
pub(crate) mod residual_add;
pub(crate) mod rope;
pub(crate) mod silu_mul;
pub(crate) mod topk;

#[cfg(test)]
mod tests;

use color_eyre::eyre::{Result, eyre};

use super::{CudaBuffer, CudaFormat};

fn bytes(items: u64, width: u64) -> Result<usize> {
    items
        .checked_mul(width)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(arithmetic)
}

fn format_code(format: CudaFormat) -> Result<u32> {
    match format {
        CudaFormat::F16 => Ok(0),
        CudaFormat::Q4K => Ok(1),
        CudaFormat::Q5K => Ok(2),
        CudaFormat::Q6K => Ok(3),
        CudaFormat::Q6KCached => Ok(4),
        _ => Err(arithmetic()),
    }
}

fn weight_bytes(format: CudaFormat, width: u32, rows: u32) -> Result<usize> {
    let row = match format {
        CudaFormat::F16 => u64::from(width).checked_mul(2),
        CudaFormat::Q4K if width.is_multiple_of(256) => u64::from(width / 256).checked_mul(144),
        CudaFormat::Q5K if width.is_multiple_of(256) => u64::from(width / 256).checked_mul(176),
        CudaFormat::Q6K if width.is_multiple_of(256) => u64::from(width / 256).checked_mul(210),
        CudaFormat::Q6KCached if width.is_multiple_of(256) => {
            u64::from(width / 256).checked_mul(320)
        }
        _ => None,
    }
    .ok_or_else(arithmetic)?;
    bytes(row, u64::from(rows))
}

fn span(buffer: &CudaBuffer, format: CudaFormat, bytes: usize) -> Result<()> {
    // Request-local graph arenas are byte-only `Raw` allocations; each operation
    // supplies their concrete interpretation and proves the complete byte span.
    let compatible = buffer.format() == format
        || (buffer.format() == CudaFormat::Raw && format != CudaFormat::Raw);
    if compatible && bytes <= buffer.len() {
        Ok(())
    } else {
        Err(arithmetic())
    }
}

fn arithmetic() -> color_eyre::Report {
    eyre!("cuda: buffer arithmetic overflow")
}
