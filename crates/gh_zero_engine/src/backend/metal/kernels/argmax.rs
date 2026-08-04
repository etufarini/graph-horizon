/*
 * gh_zero_engine — Metal deterministic argmax
 * Records the existing reduction on a caller-owned encoder, decodes the result
 * only after accepted completion, and composes both for standalone reads. The
 * backend retains command lifecycle and persistent reduction-buffer ownership.
 */
use super::super::{
    Device, MetalBuffer,
    exec::{dispatch, encoder::MetalEncoder},
    pipeline::{Kernel, PipelineRegistry},
};
use color_eyre::eyre::{Result, bail, eyre};

pub(crate) fn encode(
    encoder: &MetalEncoder,
    pipelines: &PipelineRegistry,
    logits: &MetalBuffer,
    out: &MetalBuffer,
    vocab: usize,
) -> Result<()> {
    let vocab = checked_vocab(vocab)?;
    dispatch::encode(
        encoder,
        pipelines,
        Kernel::Argmax,
        &[logits, out],
        &vocab.to_ne_bytes(),
        [1, 1, 1],
    )
}

pub(crate) fn completed(out: &MetalBuffer, vocab: usize) -> Result<u32> {
    let vocab = checked_vocab(vocab)?;
    let bytes = out.read(4)?;
    let token = u32::from_ne_bytes(
        bytes
            .try_into()
            .map_err(|_| eyre!("metal: incomplete argmax readback"))?,
    );
    if token >= vocab {
        bail!("metal: invalid argmax result");
    }
    Ok(token)
}

pub(crate) fn read(
    d: &Device,
    p: &PipelineRegistry,
    logits: &MetalBuffer,
    out: &MetalBuffer,
    vocab: usize,
) -> Result<u32> {
    checked_vocab(vocab)?;
    let encoder = MetalEncoder::begin(d)?;
    encode(&encoder, p, logits, out, vocab)?;
    encoder.submit()?;
    completed(out, vocab)
}

fn checked_vocab(vocab: usize) -> Result<u32> {
    let vocab = u32::try_from(vocab).map_err(|_| eyre!("metal: buffer arithmetic overflow"))?;
    if vocab == 0 {
        bail!("metal: invalid readback size");
    }
    Ok(vocab)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::MetalFormat;

    #[test]
    fn fused_record_completes_once_and_preserves_lowest_tie() -> Result<()> {
        let device = Device::acquire()?;
        let pipelines = PipelineRegistry::load(&device)?;
        let logits = MetalBuffer::allocate(&device, 16, MetalFormat::F32)?;
        let bytes: Vec<u8> = [1.0_f32, 3.0, 3.0, 2.0]
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        logits.write(&bytes)?;
        let out = MetalBuffer::allocate(&device, 4, MetalFormat::Raw)?;

        let encoder = MetalEncoder::begin(&device)?;
        encode(&encoder, &pipelines, &logits, &out, 4)?;
        encoder.submit()?;
        assert_eq!(completed(&out, 4)?, 1);
        Ok(())
    }

    #[test]
    fn invalid_vocabulary_is_rejected_before_dispatch() -> Result<()> {
        let device = Device::acquire()?;
        let pipelines = PipelineRegistry::load(&device)?;
        let buffer = MetalBuffer::allocate(&device, 4, MetalFormat::Raw)?;
        let encoder = MetalEncoder::begin(&device)?;
        assert_eq!(
            encode(&encoder, &pipelines, &buffer, &buffer, 0)
                .unwrap_err()
                .to_string(),
            "metal: invalid readback size"
        );
        Ok(())
    }

    #[test]
    fn completed_read_failure_is_terminal() -> Result<()> {
        let device = Device::acquire()?;
        let out = MetalBuffer::allocate(&device, 1, MetalFormat::Raw)?;
        assert_eq!(
            completed(&out, 4).unwrap_err().to_string(),
            "metal: readback past buffer window"
        );
        let out = MetalBuffer::allocate(&device, 4, MetalFormat::Raw)?;
        out.write(&9_u32.to_ne_bytes())?;
        assert_eq!(
            completed(&out, 4).unwrap_err().to_string(),
            "metal: invalid argmax result"
        );
        Ok(())
    }
}
