/*
 * graph_horizon_engine — device-side logit reduction (argmax / top-k)
 * The single file for the Vulkan-side reduction of the FP32 logits in the decode
 * hot-path: it dispatches `argmax.comp` / `topk_partial.comp`, reads back the
 * minimal result through the backend's host mirror, and (for top-k) performs the
 * final host merge of the per-workgroup partials with the EXACT `sampling.rs`
 * comparator. It owns no pipeline and no buffer — the device, registry, the
 * device-local `reduce` scratch and the host mirror are all passed in by the
 * backend. The parity invariant (tiebreak identical to `sampling.rs`) lives at
 * the merge step here and in the shaders.
*/

use ash::vk;
use color_eyre::eyre::{Result, bail, eyre};

use crate::backend::vulkan::buffers::GpuBuffer;
use crate::backend::vulkan::device::Device;
use crate::backend::vulkan::pipeline::{Kernel, PipelineRegistry, dispatch};

// Number of single-thread workgroups the top-k partial scan splits the vocab
// across. Must match the dispatch below; the host then merges GROUPS*k partials.
pub(crate) const TOPK_GROUPS: u32 = 64;

// Upper bound on k for the device path. MUST equal `sampling::MAX_DEVICE_K` and
// the `MAXK` constant in `topk_partial.comp` (the shader sizes its in-register
// working set with it). The `sampling::plan` classifier never emits a larger k
// (it falls back instead), so this is only a defensive clamp.
pub(crate) const MAX_K: usize = 128;

// Bytes of one (uint index, float value) pair as laid out by `topk_partial.comp`
// (std430 `struct Pair { uint; float; }`, tightly packed).
const PAIR_BYTES: usize = 8;

// Argmax of `vocab` FP32 logits on the device: dispatch the single-workgroup
// reduction, copy the 4-byte index into the host mirror, read it back. Same
// tiebreak as `sampling.rs` (lowest index at the max).
pub(crate) fn argmax(
    dev: &Device,
    reg: &PipelineRegistry,
    logits: &GpuBuffer,
    reduce: &GpuBuffer,
    host: &GpuBuffer,
    vocab: usize,
) -> Result<u32> {
    if vocab == 0 {
        bail!("vulkan: read_argmax on empty logits");
    }
    require_aligned(dev, reduce)?;

    let cmd = dev.begin_commands()?;
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::Argmax,
        &[
            (logits.buffer, logits.offset, logits.size),
            (reduce.buffer, reduce.offset, reduce.size),
        ],
        &(vocab as u32).to_le_bytes(),
        1,
    );
    copy_to_host(dev, cmd, reduce, host, 4);
    dev.submit_wait(cmd)?;

    let bytes = read_host(dev, host, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

// Top-k of `vocab` FP32 logits: dispatch the per-workgroup partial scan, read the
// GROUPS*k partial pairs back, then merge them on the host with the exact
// comparator and keep the leading `min(k, vocab)`. Returns `(index, raw logit)`
// pairs under `(logit desc, index asc)`.
pub(crate) fn topk(
    dev: &Device,
    reg: &PipelineRegistry,
    logits: &GpuBuffer,
    reduce: &GpuBuffer,
    host: &GpuBuffer,
    vocab: usize,
    k: usize,
) -> Result<Vec<(u32, f32)>> {
    if vocab == 0 {
        bail!("vulkan: read_topk on empty logits");
    }
    require_aligned(dev, reduce)?;

    // The shader keeps at most MAX_K per stripe; `plan` guarantees k <= MAX_K, so
    // this clamp only guards the buffer. The final length is min(k, vocab).
    let shader_k = k.min(MAX_K).min(vocab).max(1);
    let mut push = Vec::with_capacity(8);
    push.extend_from_slice(&(vocab as u32).to_le_bytes());
    push.extend_from_slice(&(shader_k as u32).to_le_bytes());

    let pairs_bytes = (TOPK_GROUPS as usize * shader_k * PAIR_BYTES) as u64;
    let cmd = dev.begin_commands()?;
    dispatch(
        dev,
        reg,
        cmd,
        Kernel::TopkPartial,
        &[
            (logits.buffer, logits.offset, logits.size),
            (reduce.buffer, reduce.offset, reduce.size),
        ],
        &push,
        TOPK_GROUPS,
    );
    copy_to_host(dev, cmd, reduce, host, pairs_bytes);
    dev.submit_wait(cmd)?;

    let raw = read_host(dev, host, pairs_bytes as usize)?;
    let mut cand: Vec<(u32, f32)> = raw
        .chunks_exact(PAIR_BYTES)
        .map(|c| {
            let idx = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let val = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            (idx, val)
        })
        // Skip the sentinel slots written by stripes shorter than k.
        .filter(|&(idx, _)| (idx as usize) < vocab)
        .collect();

    // Final merge with the exact `sampling.rs` total order, then keep the top-k.
    let order = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
    let keep = k.min(vocab);
    if keep < cand.len() {
        cand.select_nth_unstable_by(keep - 1, order);
        cand.truncate(keep);
    }
    cand.sort_unstable_by(order);
    Ok(cand)
}

// The reduce scratch is a dedicated full allocation (offset 0), so this always
// holds; the check documents the binding-offset invariant explicitly.
fn require_aligned(dev: &Device, reduce: &GpuBuffer) -> Result<()> {
    let align = dev.min_storage_buffer_offset_alignment;
    if !reduce.offset.is_multiple_of(align) {
        bail!("vulkan: reduce buffer offset not aligned to {align} bytes");
    }
    Ok(())
}

// compute-shader-write → transfer-read barrier, then copy `bytes` from the
// device-local reduce scratch into the host-visible mirror.
fn copy_to_host(
    dev: &Device,
    cmd: vk::CommandBuffer,
    reduce: &GpuBuffer,
    host: &GpuBuffer,
    bytes: u64,
) {
    #[cfg(feature = "vulkan-profile")]
    dev.profile.barrier();
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
    let region = vk::BufferCopy::default()
        .src_offset(reduce.offset)
        .dst_offset(0)
        .size(bytes);
    // SAFETY: `cmd` is recording; the barrier orders the reduce shader's write before the
    // transfer read, and `region` copies exactly `bytes` from `reduce` (a live buffer).
    unsafe {
        dev.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        dev.device
            .cmd_copy_buffer(cmd, reduce.buffer, host.buffer, &[region]);
    }
}

// Maps the host mirror and copies `bytes` out (the mirror is already filled by a
// submitted+waited copy). The mirror is host-coherent (allocated host-visible).
fn read_host(dev: &Device, host: &GpuBuffer, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; bytes];
    // SAFETY: host-coherent mapping of the full mirror; we copy exactly `bytes`,
    // which the callers sized to fit (mirror holds the FP32 vocab logits).
    unsafe {
        let ptr = dev
            .device
            .map_memory(host.memory, 0, host.size, vk::MemoryMapFlags::empty())
            .map_err(|_| eyre!("vulkan: reduce map failed"))?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, out.as_mut_ptr(), bytes);
        dev.device.unmap_memory(host.memory);
    }
    Ok(out)
}
