/*
 * graph_horizon_engine — checked CUDA kernel launch boundary.
 * It binds the owning context, proves buffer windows, retains cudarc guards
 * through enqueue, and is the sole source launch site in the backend.
 */

use std::ffi::c_void;

use color_eyre::eyre::{Result, eyre};
use cudarc::driver::{SyncOnDrop, result};

use super::encoder::CudaEncoder;
use crate::backend::cuda::mem::buffer::CudaBuffer;
use crate::backend::cuda::module::{Kernel, Module};

type GridAndBlock = ((u32, u32, u32), (u32, u32, u32));

pub(crate) enum Arg<'a> {
    Buffer(&'a CudaBuffer, usize),
    U32(u32),
    U64(u64),
    F32(f32),
}

enum Param {
    U32(u32),
    U64(u64),
    F32(f32),
}

impl Param {
    fn pointer(&mut self) -> *mut c_void {
        match self {
            Self::U32(value) => (value as *mut u32).cast(),
            Self::U64(value) => (value as *mut u64).cast(),
            Self::F32(value) => (value as *mut f32).cast(),
        }
    }
}

pub(crate) fn launch(
    encoder: &CudaEncoder,
    module: &Module,
    kernel: Kernel,
    args: &[Arg<'_>],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
) -> Result<()> {
    // Catch prefill ABI regressions in tests before the driver reads scalar pointers.
    #[cfg(test)]
    if matches!(
        kernel,
        Kernel::AttentionPrefillF16 | Kernel::AttentionPrefillInt8
    ) {
        let expected = if matches!(kernel, Kernel::AttentionPrefillInt8) {
            13
        } else {
            11
        };
        assert_eq!(args.len(), expected, "CUDA prefill argument count");
    }
    if !encoder.ready() {
        return Ok(());
    }
    let block_threads = block
        .0
        .checked_mul(block.1)
        .and_then(|value| value.checked_mul(block.2))
        .ok_or_else(arithmetic)?;
    if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 || block_threads == 0 || block_threads > 256 {
        return Err(arithmetic());
    }
    encoder
        .stream
        .context()
        .bind_to_thread()
        .map_err(|_| failed())?;
    let mut params = Vec::with_capacity(args.len());
    let mut guards: Vec<SyncOnDrop<'_>> = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            Arg::Buffer(buffer, bytes) => {
                let (pointer, guard) = buffer.device_ptr(&encoder.stream, *bytes)?;
                params.push(Param::U64(pointer));
                guards.push(guard);
            }
            Arg::U32(value) => params.push(Param::U32(*value)),
            Arg::U64(value) => params.push(Param::U64(*value)),
            Arg::F32(value) => params.push(Param::F32(*value)),
        }
    }
    let mut pointers = params.iter_mut().map(Param::pointer).collect::<Vec<_>>();
    // SAFETY: the function is a fixed ABI handle owned by the live Module; the
    // current context owns the sole non-blocking stream; Rust validated every
    // complete byte range before deriving device pointers; scalar Param storage
    // and pointer guards remain live through enqueue; stream order prevents
    // concurrent mutation and submit synchronizes before resources can release.
    unsafe {
        result::launch_kernel(
            module.functions.get(kernel),
            grid,
            block,
            0,
            encoder.stream.cu_stream(),
            &mut pointers,
        )
    }
    .map_err(|_| failed())?;
    drop(guards);
    encoder.submitted_launch();
    Ok(())
}

pub(crate) fn one_dim(total: u64) -> Result<GridAndBlock> {
    const THREADS: u64 = 256;
    if total == 0 {
        return Err(arithmetic());
    }
    let groups = total
        .checked_add(THREADS - 1)
        .and_then(|value| value.checked_div(THREADS))
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(arithmetic)?;
    Ok(((groups, 1, 1), (THREADS as u32, 1, 1)))
}

fn arithmetic() -> color_eyre::Report {
    eyre!("cuda: buffer arithmetic overflow")
}

fn failed() -> color_eyre::Report {
    eyre!("cuda: command submission failed")
}
