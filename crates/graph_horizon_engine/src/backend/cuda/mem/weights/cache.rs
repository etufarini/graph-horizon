/*
 * Optional Q6 prefill companion. Raw weights stay owned and authoritative for
 * decode; conversion synchronizes before the two representations are attached.
 */

use color_eyre::eyre::{Result, eyre};

use crate::backend::cuda::exec::dispatch::{self, Arg};
use crate::backend::cuda::module::{Kernel, Module};
use crate::backend::cuda::{CudaBuffer, CudaEncoder, CudaFormat, Device};

pub(in crate::backend::cuda) fn upload(
    device: &Device,
    module: Option<&Module>,
    bytes: &[u8],
    format: CudaFormat,
) -> Result<CudaBuffer> {
    let Some(module) = module.filter(|_| format == CudaFormat::Q6K) else {
        return CudaBuffer::upload(device, bytes, format);
    };
    if bytes.is_empty() || !bytes.len().is_multiple_of(210) {
        return Err(eyre!("cuda: buffer arithmetic overflow"));
    }
    for block in bytes.chunks_exact(210) {
        let bits = u16::from_le_bytes([block[208], block[209]]);
        if bits & 0x7c00 == 0x7c00 {
            return CudaBuffer::upload(device, bytes, format);
        }
    }
    let blocks =
        u32::try_from(bytes.len() / 210).map_err(|_| eyre!("cuda: buffer arithmetic overflow"))?;
    let size = u64::from(blocks)
        .checked_mul(320)
        .ok_or_else(|| eyre!("cuda: buffer arithmetic overflow"))?;
    let output_bytes =
        usize::try_from(size).map_err(|_| eyre!("cuda: buffer arithmetic overflow"))?;
    let raw = CudaBuffer::upload(device, bytes, format)?;
    let output = CudaBuffer::allocate(device, size, CudaFormat::Q6KCached)?;
    let encoder = CudaEncoder::begin(device);
    encoder.latch(dispatch::launch(
        &encoder,
        module,
        Kernel::WeightCache,
        &[
            Arg::Buffer(&raw, bytes.len()),
            Arg::Buffer(&output, output_bytes),
        ],
        (blocks, 1, 1),
        (256, 1, 1),
    ));
    // Even an enqueue failure synchronizes before either allocation drops.
    encoder.submit()?;
    raw.with_prefill(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::mem::buffer::{reset_test_counts, test_counts};

    #[test]
    fn companion_formats_sizes_aliases_and_lifetimes_are_checked() -> Result<()> {
        let device = Device::acquire()?;
        let module = Module::load(&device.context)?;
        for len in [0, 209, 211] {
            assert!(upload(&device, Some(&module), &vec![0; len], CudaFormat::Q6K).is_err());
        }
        for bits in [0x7c00u16, 0xfc00, 0x7e01, 0xfe01] {
            let mut raw = vec![0; 210];
            raw[208..210].copy_from_slice(&bits.to_le_bytes());
            let weight = upload(&device, Some(&module), &raw, CudaFormat::Q6K)?;
            assert_eq!(weight.prefill().format(), CudaFormat::Q6K);
            assert_eq!(weight.read(&device, 210)?, raw);
        }
        reset_test_counts();
        let raw = upload(&device, Some(&module), &[0; 420], CudaFormat::Q6K)?;
        assert_eq!(raw.format(), CudaFormat::Q6K);
        assert_eq!(raw.prefill().format(), CudaFormat::Q6KCached);
        assert_eq!(raw.retained_bytes(), Some(448 + 640));
        assert!(raw.prefill().view(2, 4).is_err());
        let full = raw.view(0, 420)?;
        let partial = raw.view(210, 210)?;
        assert_eq!(full.prefill().format(), CudaFormat::Q6KCached);
        assert_eq!(partial.prefill().format(), CudaFormat::Q6K);
        drop(raw);
        assert_eq!(full.prefill().read(&device, 4)?, [0; 4]);
        drop(full);
        assert_eq!(
            test_counts(),
            (2, 1),
            "partial raw alias does not retain the cache"
        );
        drop(partial);
        assert_eq!(test_counts(), (2, 2));
        for (raw_format, raw_bytes, cache_format, cache_bytes) in [
            (CudaFormat::Q4K, 210, CudaFormat::Q6KCached, 320),
            (CudaFormat::Q6K, 210, CudaFormat::Q6K, 320),
            (CudaFormat::Q6K, 210, CudaFormat::Q6KCached, 319),
            (CudaFormat::Q6K, 209, CudaFormat::Q6KCached, 320),
        ] {
            let raw = CudaBuffer::allocate(&device, raw_bytes, raw_format)?;
            let cache = CudaBuffer::allocate(&device, cache_bytes, cache_format)?;
            assert!(raw.with_prefill(cache).is_err());
        }
        let attached = upload(&device, Some(&module), &[0; 210], CudaFormat::Q6K)?;
        let another = CudaBuffer::allocate(&device, 320, CudaFormat::Q6KCached)?;
        assert!(attached.with_prefill(another).is_err());
        Ok(())
    }
}
