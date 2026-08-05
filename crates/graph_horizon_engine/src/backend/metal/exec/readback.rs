/*
 * graph_horizon_engine — bounded Metal shared-buffer readback
 * Copies completed FP32 logits from a validated window, rejects non-finite data,
 * and provides deterministic argmax/top-k ordering. It submits no GPU work.
 */

use color_eyre::eyre::{Result, bail, eyre};

use crate::backend::metal::MetalBuffer;

pub(crate) fn logits(buffer: &MetalBuffer, vocab: usize) -> Result<Vec<f32>> {
    if vocab == 0 {
        bail!("metal: invalid readback size");
    }
    let bytes = vocab.checked_mul(4).ok_or_else(arithmetic)?;
    let raw = buffer.read(bytes)?;
    if raw.len() != bytes {
        bail!("metal: invalid readback size");
    }
    let values = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        bail!("metal: non-finite logits");
    }
    Ok(values)
}

#[cfg(test)]
pub(crate) fn argmax(buffer: &MetalBuffer, vocab: usize) -> Result<u32> {
    let values = logits(buffer, vocab)?;
    let index = values
        .iter()
        .enumerate()
        .max_by(|(left_i, left), (right_i, right)| {
            left.total_cmp(right).then_with(|| right_i.cmp(left_i))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| eyre!("metal: invalid readback size"))?;
    u32::try_from(index).map_err(|_| arithmetic())
}

#[cfg(test)]
pub(crate) fn topk(buffer: &MetalBuffer, vocab: usize, k: usize) -> Result<Vec<(u32, f32)>> {
    if k == 0 {
        bail!("metal: invalid readback size");
    }
    let values = logits(buffer, vocab)?;
    let mut indexed = values.into_iter().enumerate().collect::<Vec<_>>();
    indexed.sort_unstable_by(|(left_i, left), (right_i, right)| {
        right.total_cmp(left).then_with(|| left_i.cmp(right_i))
    });
    indexed
        .into_iter()
        .take(k.min(vocab))
        .map(|(index, value)| {
            u32::try_from(index)
                .map(|index| (index, value))
                .map_err(|_| arithmetic())
        })
        .collect()
}

fn arithmetic() -> color_eyre::Report {
    eyre!("metal: buffer arithmetic overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::{Device, MetalFormat};

    fn buffer(values: &[f32]) -> Result<MetalBuffer> {
        let device = Device::acquire()?;
        let output = MetalBuffer::allocate(&device, (values.len() * 4) as u64, MetalFormat::F32)?;
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        output.write(&bytes)?;
        Ok(output)
    }

    #[test]
    fn reductions_are_bounded_finite_and_deterministic() -> Result<()> {
        let values = buffer(&[1.0, 3.0, 3.0, -2.0])?;
        assert_eq!(logits(&values, 4)?, vec![1.0, 3.0, 3.0, -2.0]);
        assert_eq!(argmax(&values, 4)?, 1);
        assert_eq!(topk(&values, 4, 3)?, vec![(1, 3.0), (2, 3.0), (0, 1.0)]);
        assert_eq!(
            logits(&values, 5).unwrap_err().to_string(),
            "metal: readback past buffer window"
        );
        assert_eq!(
            logits(&values, 0).unwrap_err().to_string(),
            "metal: invalid readback size"
        );
        assert_eq!(
            topk(&values, 4, 0).unwrap_err().to_string(),
            "metal: invalid readback size"
        );
        assert_eq!(
            logits(&buffer(&[f32::NAN])?, 1).unwrap_err().to_string(),
            "metal: non-finite logits"
        );
        Ok(())
    }
}
