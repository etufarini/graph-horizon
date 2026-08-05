/*
 * graph_horizon_engine — Metal projection dispatch
 * Selects scalar or bounded cooperative projection and binds its checked grid.
 * Features control availability; effective Mixed placement and unsupported
 * shapes retain the per-row path.
 */
use super::super::exec::dispatch;
use super::super::{
    MetalBuffer, MetalEncoder, MetalFormat,
    pipeline::{Kernel, PipelineRegistry},
};
use color_eyre::eyre::{Result, eyre};
pub(crate) fn encode(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    out: &MetalBuffer,
    a: &MetalBuffer,
    w: &MetalBuffer,
    input: u32,
    output: u32,
    fp32: bool,
) -> Result<()> {
    let format = match w.format() {
        MetalFormat::F16 => 0,
        MetalFormat::Q4K => 1,
        MetalFormat::Q5K => 2,
        MetalFormat::Q6K => 3,
        other => return Err(eyre!("metal: unsupported weight format '{other:?}'")),
    };
    dispatch::encode(
        e,
        p,
        Kernel::Matmul,
        &[a, w, out],
        &super::u32s(&[input, output, format, u32::from(fp32)]),
        [output as usize, 1, 1],
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_batched(
    e: &MetalEncoder,
    p: &PipelineRegistry,
    out: &MetalBuffer,
    a: &MetalBuffer,
    w: &MetalBuffer,
    input: u32,
    output: u32,
    rows: u32,
    mixed_placement: bool,
) -> Result<()> {
    let format = match w.format() {
        MetalFormat::Q4K => 1,
        MetalFormat::Q6K => 3,
        _ => 0,
    };
    // The cooperative path is retained only where its A/B control passed;
    // an effective Mixed suffix keeps the stable per-row execution order.
    if cooperative(mixed_placement, rows, input, output, format) {
        // One 32-lane SIMD group computes each 8-column output tile.
        if p.get(Kernel::MatmulBatched).width != 32 {
            return Err(eyre!("metal: batched projection requires 32 threads"));
        }
        return dispatch::encode(
            e,
            p,
            Kernel::MatmulBatched,
            &[a, w, out],
            &super::u32s(&[input, output, format, rows]),
            [(output as usize).div_ceil(8) * 32, 1, 1],
        );
    }

    let input_stride = u64::from(input) * 2;
    let output_stride = u64::from(output) * 2;
    for row in 0..u64::from(rows) {
        let input_row = a.view(row * input_stride, input_stride)?;
        let output_row = out.view(row * output_stride, output_stride)?;
        encode(e, p, &output_row, &input_row, w, input, output, false)?;
    }
    Ok(())
}

fn cooperative(mixed_placement: bool, rows: u32, input: u32, output: u32, format: u32) -> bool {
    !mixed_placement
        && rows > 1
        && rows <= 32
        && input.is_multiple_of(256)
        && output.is_multiple_of(32)
        && format != 0
}

#[cfg(test)]
mod tests {
    use super::cooperative;

    #[test]
    fn metal_matmul_route_depends_on_effective_placement_not_feature() {
        assert!(cooperative(false, 2, 256, 32, 1));
        assert!(!cooperative(true, 2, 256, 32, 1));
        for args in [
            (1, 256, 32, 1),
            (2, 255, 32, 1),
            (2, 256, 31, 1),
            (2, 256, 32, 0),
        ] {
            assert!(!cooperative(false, args.0, args.1, args.2, args.3));
            assert!(!cooperative(true, args.0, args.1, args.2, args.3));
        }
    }
}
