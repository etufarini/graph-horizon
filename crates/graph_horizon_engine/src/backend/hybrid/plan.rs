/*
 * graph_horizon_engine — immutable hybrid plan
 * Validates one contiguous CPU-prefix/GPU-suffix split and stores its exact byte
 * report. It owns no allocation, device probing, graph traversal, or retry.
 */

use color_eyre::eyre::{Result, bail};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HybridMode {
    AllGpu,
    Mixed,
    CpuOnly,
}

impl HybridMode {
    pub(crate) fn name_for(self, all_gpu: &'static str) -> &'static str {
        match self {
            Self::AllGpu => all_gpu,
            Self::Mixed => "mixed",
            Self::CpuOnly => "cpu-only",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HybridPlan {
    pub(crate) mode: HybridMode,
    pub(crate) split: usize,
    pub(crate) block_count: usize,
    pub(crate) cpu_layers: usize,
    pub(crate) gpu_layers: usize,
    pub(crate) cpu: BackendBytes,
    pub(crate) gpu: BackendBytes,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BackendBytes {
    pub(crate) weights: u64,
    pub(crate) kv: u64,
    pub(crate) scratch: u64,
    pub(crate) fixed: u64,
    pub(crate) staging: u64,
    pub(crate) crossing: u64,
    pub(crate) reserve: u64,
    pub(crate) total: u64,
}

impl HybridPlan {
    pub(crate) fn new(
        split: usize,
        block_count: usize,
        cpu: BackendBytes,
        gpu: BackendBytes,
    ) -> Result<Self> {
        if split > block_count {
            bail!("invalid hybrid split");
        }
        let mode = if split == 0 {
            HybridMode::AllGpu
        } else if split == block_count {
            HybridMode::CpuOnly
        } else {
            HybridMode::Mixed
        };
        Ok(Self {
            mode,
            split,
            block_count,
            cpu_layers: split,
            gpu_layers: block_count - split,
            cpu,
            gpu,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_split_before_resource_work() {
        let error =
            HybridPlan::new(3, 2, BackendBytes::default(), BackendBytes::default()).unwrap_err();
        assert_eq!(error.to_string(), "invalid hybrid split");
    }
}
