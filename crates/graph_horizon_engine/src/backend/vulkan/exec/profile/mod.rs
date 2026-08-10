/*
 * Vulkan inference profiler: owns one reusable timestamp pool and aggregates
 * feature-gated prefill, decode, sampling, synchronization, and allocation
 * diagnostics. Classification and stderr presentation remain in siblings.
 */

mod category;
mod report;
mod summary;

use std::sync::Mutex;
use std::time::Instant;

use ash::vk;
use color_eyre::eyre::{Result, eyre};

use self::category::Phase;
use self::summary::{Mark, PhaseTotals};
use crate::backend::vulkan::pipeline::Kernel;

const QUERY_COUNT: u32 = 8192;
type Stamp = Option<(vk::QueryPool, u32)>;

pub(crate) struct Profile {
    period_ns: f64,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    pool: Option<vk::QueryPool>,
    next: u32,
    started: Option<Instant>,
    marks: Vec<Mark>,
    barriers: u64,
    matmul_slot: usize,
    layer: Option<u8>,
    residuals: u8,
    phase: Option<Phase>,
    cpu_ms: [f64; 4],
    totals: PhaseTotals,
    allocations: [u64; 3],
    allocation_ms: f64,
}

impl Profile {
    pub(crate) fn new(period_ns: f32) -> Self {
        Self {
            period_ns: period_ns as f64,
            state: Mutex::new(State::default()),
        }
    }

    pub(crate) fn begin(&self, device: &ash::Device, cmd: vk::CommandBuffer) -> Result<()> {
        let mut state = self.state.lock().expect("vulkan profile lock");
        let pool = match state.pool {
            Some(pool) => pool,
            None => {
                let info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(QUERY_COUNT);
                // SAFETY: the live logical device owns the returned query pool.
                let pool = unsafe { device.create_query_pool(&info, None) }
                    .map_err(|_| eyre!("vulkan profile: cannot create query pool"))?;
                state.pool = Some(pool);
                pool
            }
        };
        // SAFETY: the synchronous queue guarantees the prior pool use completed.
        unsafe { device.reset_query_pool(pool, 0, QUERY_COUNT) };
        state.next = 1;
        state.started = Some(Instant::now());
        state.marks.clear();
        state.barriers = 0;
        state.matmul_slot = 0;
        state.layer = None;
        state.residuals = 0;
        state.phase = None;
        state.cpu_ms = [0.0; 4];
        // SAFETY: query zero is reset and `cmd` is recording on this device.
        unsafe { device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 0) };
        Ok(())
    }

    pub(crate) fn begin_kernel(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        kernel: Kernel,
        groups_x: u32,
        groups_y: u32,
    ) -> Stamp {
        let mut state = self.state.lock().expect("vulkan profile lock");
        if state.next + 2 >= QUERY_COUNT {
            state.next = QUERY_COUNT;
            return None;
        }
        let (start, end) = (state.next, state.next + 1);
        state.next += 2;
        let category = category::profiled(kernel, &mut state.matmul_slot);
        if kernel == Kernel::RmsNormX && state.layer.is_none() {
            state.layer = Some(0);
        }
        if let Some(phase) = category::phase(kernel) {
            // One submitted command belongs to at most one inference phase.
            debug_assert!(state.phase.is_none_or(|current| current == phase));
            state.phase = Some(phase);
        }
        let layer = state.layer;
        state.marks.push(Mark {
            category,
            kernel,
            start,
            end,
            groups: [groups_x, groups_y],
            layer,
        });
        if kernel == Kernel::Residual {
            state.residuals += 1;
            if state.residuals == 2 {
                state.layer = state.layer.and_then(|layer| layer.checked_add(1));
                state.residuals = 0;
            }
        }
        let pool = state.pool.expect("profile pool while recording");
        // SAFETY: the reserved query is reset and `cmd` is recording.
        unsafe {
            device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, pool, start)
        };
        Some((pool, end))
    }

    pub(crate) fn end_kernel(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        stamp: Stamp,
        descriptor_ms: f64,
    ) {
        if let Some((pool, end)) = stamp {
            // SAFETY: `end` was reserved with this live pool for this recording.
            unsafe {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::COMPUTE_SHADER, pool, end)
            };
        }
        self.state.lock().expect("vulkan profile lock").cpu_ms[3] += descriptor_ms;
    }

    pub(crate) fn end(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        let mut state = self.state.lock().expect("vulkan profile lock");
        if state.next < QUERY_COUNT {
            let pool = state.pool.expect("profile pool while recording");
            let end = state.next;
            state.next += 1;
            // SAFETY: the reserved query is reset and `cmd` remains recording.
            unsafe {
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pool, end)
            };
        }
        state.cpu_ms[0] = state
            .started
            .take()
            .map_or(0.0, |start| start.elapsed().as_secs_f64() * 1000.0);
    }

    pub(crate) fn barrier(&self) {
        self.state.lock().expect("vulkan profile lock").barriers += 1;
    }

    pub(crate) fn cpu(&self, index: usize, elapsed_ms: f64) {
        self.state.lock().expect("vulkan profile lock").cpu_ms[index] = elapsed_ms;
    }

    pub(crate) fn allocation(&self, bytes: u64, host: bool, elapsed_ms: f64) {
        let mut state = self.state.lock().expect("vulkan profile lock");
        // The first completed prefill command is the phase latch: model loading
        // allocations are excluded, while later request-local churn is retained.
        if state.totals[Phase::Prefill as usize].counts[0] > 0 {
            state.allocations[0] += 1;
            state.allocations[1] += bytes;
            state.allocations[2] += u64::from(host);
            state.allocation_ms += elapsed_ms;
        }
    }

    pub(crate) fn resolve(&self, device: &ash::Device) -> Result<()> {
        let mut state = self.state.lock().expect("vulkan profile lock");
        if state.next >= QUERY_COUNT {
            return Err(eyre!("vulkan profile: query capacity exceeded"));
        }
        let end = state
            .next
            .checked_sub(1)
            .ok_or_else(|| eyre!("vulkan profile: missing command end"))?;
        let mut values = vec![0u64; state.next as usize];
        // SAFETY: the submission fence completed, so every query is available.
        unsafe {
            device.get_query_pool_results(
                state.pool.expect("profile pool after submission"),
                0,
                &mut values,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )
        }
        .map_err(|_| eyre!("vulkan profile: cannot read timestamps"))?;
        let Some(phase) = state.phase else {
            return Ok(());
        };
        let tick_ms = self.period_ns / 1_000_000.0;
        let gpu_ms = values[end as usize].wrapping_sub(values[0]) as f64 * tick_ms;
        let (dispatches, barriers, cpu_ms) =
            (state.marks.len() as u64, state.barriers, state.cpu_ms);
        let State { marks, totals, .. } = &mut *state;
        totals[phase as usize].add_command(
            gpu_ms, dispatches, barriers, cpu_ms, marks, &values, tick_ms,
        );
        Ok(())
    }

    pub(crate) fn finish(&self, device: &ash::Device) {
        let mut state = self.state.lock().expect("vulkan profile lock");
        report::write(&state.totals, state.allocations, state.allocation_ms);
        if let Some(pool) = state.pool.take() {
            // SAFETY: Device::drop waited idle; no command can use this pool.
            unsafe { device.destroy_query_pool(pool, None) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QUERY_COUNT;

    #[test]
    fn selected_prefill_batch_fits_query_pool() {
        let dispatches = 32 + 26 * 78 + 2;
        assert!(1 + dispatches * 2 + 1 < QUERY_COUNT);
    }

    #[test]
    fn selected_decode_step_fits_query_pool() {
        let dispatches = 1 + 26 * 16 + 2;
        assert!(1 + dispatches * 2 + 1 < QUERY_COUNT);
    }
}
