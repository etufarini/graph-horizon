/*
 * gh_zero_engine — Vulkan memory budget
 * Owns VRAM discovery, reserve/weight caps, hybrid fallback and pure preflight.
*/

// OnceLock backs the dial cache only in the single-backend builds; in hybrid
// `weight_vram_percent()` is a const 100 and never caches.
#[cfg(not(feature = "hybrid"))]
use std::sync::OnceLock;

use ash::vk;

use super::memory::Budget;
use crate::backend::vulkan::device::Device;

// VRAM bytes held back before filling weights. Override wins; otherwise use
// `max(256 MiB, 5% of total VRAM)`.
#[cfg(any(test, not(feature = "hybrid")))]
pub(crate) fn reserve_bytes(total_vram: u64, override_mib: Option<u64>) -> u64 {
    override_mib
        .map(|m| m.saturating_mul(1024 * 1024))
        .unwrap_or_else(|| (256 * 1024 * 1024).max(total_vram / 20))
}

// Weight-percent dial, seeded once by `Engine::new` in single-backend builds.
#[cfg(not(feature = "hybrid"))]
static WEIGHTS_PERCENT: OnceLock<Option<u8>> = OnceLock::new();
#[cfg(not(feature = "hybrid"))]
static RESERVE_MIB: OnceLock<Option<u64>> = OnceLock::new();

// Seeds the dial from the CLI value. See `WEIGHTS_PERCENT`.
#[cfg(not(feature = "hybrid"))]
pub(crate) fn set_weights_percent(percent: Option<u8>) {
    let _ = WEIGHTS_PERCENT.set(percent);
}

#[cfg(not(feature = "hybrid"))]
pub(crate) fn set_reserve_mib(reserve_mib: Option<u64>) {
    let _ = RESERVE_MIB.set(reserve_mib);
}

// Resolves the dial: the seeded value (CLI-validated), else the default 100.
#[cfg(not(feature = "hybrid"))]
pub(super) fn weight_vram_percent() -> u8 {
    WEIGHTS_PERCENT
        .get()
        .copied()
        .flatten()
        .unwrap_or(100)
        .min(100)
}

#[cfg(not(feature = "hybrid"))]
pub(super) fn configured_reserve_mib() -> Option<u64> {
    RESERVE_MIB.get().copied().flatten()
}

#[cfg(all(feature = "hybrid", test))]
pub(super) fn configured_reserve_mib() -> Option<u64> {
    None
}

// In the hybrid build the dial is governed by the split runtime (which calls CPU
// for offloaded layers), not by host-visible spill: the GPU side keeps every
// weight it is given device-local, so here the cap is always 100% (no spill).
#[cfg(all(feature = "hybrid", test))]
pub(super) fn weight_vram_percent() -> u8 {
    100
}

// Byte-exact cap on weights kept resident in VRAM. The product is computed in
// u128 so no plausible model overflows.
#[cfg(any(test, not(feature = "hybrid")))]
pub(super) fn weight_vram_budget(total_weights: u64, vram_for_weights: u64, pct: u8) -> u64 {
    let scaled = (pct as u128 * total_weights as u128 / 100) as u64;
    scaled.min(vram_for_weights)
}

// VRAM is the largest device-local heap.
pub(crate) fn device_budget(dev: &Device) -> Budget {
    let mut vram = 0u64;
    let heaps = &dev.mem_props.memory_heaps[..dev.mem_props.memory_heap_count as usize];
    for heap in heaps {
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            vram = vram.max(heap.size);
        }
    }
    Budget { vram }
}

// Prefer the live budget extension for hybrid auto placement; fall back to the
// physical heap size with one sanitized diagnostic.
#[cfg(feature = "hybrid")]
pub(crate) fn vram_for_auto(dev: &Device) -> u64 {
    dev.free_vram().unwrap_or_else(|| {
        let total = device_budget(dev).vram;
        eprintln!(
            "vulkan: VK_EXT_memory_budget unavailable, using total VRAM ({total} bytes) for the auto split"
        );
        total
    })
}

// Checked preflight for the pure Vulkan backend. It deliberately separates
// non-context storage (weights + fixed buffers) from context storage (KV + scratch):
// if the former does not fit, the model itself is too large (E15); if only adding
// context storage overflows, the requested context is rejected without reduction
// (E17). Staging is the peak transient upload buffer. `percent` is a hard ceiling,
// so pure Vulkan with 0% weights cannot load.
#[cfg(any(test, not(feature = "hybrid")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pure_preflight(
    device_vram: u64,
    reserve_bytes: u64,
    percent: u8,
    weights: u64,
    fixed: u64,
    staging: u64,
    kv: u64,
    scratch: u64,
    context_len: usize,
) -> color_eyre::eyre::Result<()> {
    use color_eyre::eyre::{bail, eyre};

    let after_reserve = device_vram.checked_sub(reserve_bytes).ok_or_else(|| {
        eyre!("Vulkan memory is insufficient: required {reserve_bytes} bytes, available {device_vram} bytes")
    })?;
    let available = after_reserve;
    let weight_cap = weight_vram_budget(weights, after_reserve, percent.min(100));
    if weights > weight_cap {
        bail!(
            "Vulkan memory is insufficient: required {weights} bytes, available {weight_cap} bytes"
        );
    }
    let persistent = weights.checked_add(fixed).ok_or_else(|| {
        eyre!("Vulkan memory is insufficient: required overflow bytes, available {available} bytes")
    })?;
    let non_context = persistent.checked_add(staging).ok_or_else(|| {
        eyre!("Vulkan memory is insufficient: required overflow bytes, available {available} bytes")
    })?;
    if non_context > available {
        bail!(
            "Vulkan memory is insufficient: required {non_context} bytes, available {available} bytes"
        );
    }
    let context = kv.checked_add(scratch).ok_or_else(|| {
        eyre!("context {context_len} does not fit the selected backend; context was not reduced")
    })?;
    let required = non_context.checked_add(context).ok_or_else(|| {
        eyre!("context {context_len} does not fit the selected backend; context was not reduced")
    })?;
    if required > available {
        bail!("context {context_len} does not fit the selected backend; context was not reduced");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_at_full_percent_is_historical_min() {
        // Inv-2: pct = 100 reproduces today's weight budget, min(total, vram).
        assert_eq!(weight_vram_budget(800, 1000, 100), 800); // model fits in VRAM
        assert_eq!(weight_vram_budget(1500, 1000, 100), 1000); // capped by VRAM
    }

    #[test]
    fn budget_at_zero_percent_is_zero_weight_cap() {
        assert_eq!(weight_vram_budget(1000, 10_000, 0), 0);
    }

    #[test]
    fn budget_intermediate_floors_and_clamps() {
        assert_eq!(weight_vram_budget(1000, 10_000, 50), 500); // floor of fraction
        assert_eq!(weight_vram_budget(1001, 10_000, 50), 500); // floor (500.5 -> 500)
        assert_eq!(weight_vram_budget(10_000, 400, 90), 400); // cap > vram -> clamped
    }

    #[test]
    fn budget_large_input_does_not_overflow() {
        let total = 200_000_000_000u64; // ~200 GB of weights
        assert_eq!(weight_vram_budget(total, u64::MAX, 100), total);
        assert_eq!(weight_vram_budget(total, u64::MAX, 50), total / 2);
    }

    #[test]
    fn pure_preflight_accepts_exact_fit() {
        pure_preflight(100, 10, 100, 50, 10, 0, 20, 10, 32).expect("exact fit");
    }

    #[test]
    fn error_matrix_e15_rejects_reserve_underflow() {
        let msg = pure_preflight(10, 11, 100, 1, 0, 0, 0, 0, 1)
            .unwrap_err()
            .to_string();
        assert_eq!(
            msg,
            "Vulkan memory is insufficient: required 11 bytes, available 10 bytes"
        );
    }

    #[test]
    fn pure_preflight_rejects_weight_percent_cap_as_e15() {
        let msg = pure_preflight(100, 0, 50, 80, 0, 0, 0, 0, 1)
            .unwrap_err()
            .to_string();
        assert_eq!(
            msg,
            "Vulkan memory is insufficient: required 80 bytes, available 40 bytes"
        );
    }

    #[test]
    fn pure_preflight_rejects_fixed_over_budget_as_e15() {
        let msg = pure_preflight(100, 0, 100, 70, 31, 0, 0, 0, 1)
            .unwrap_err()
            .to_string();
        assert_eq!(
            msg,
            "Vulkan memory is insufficient: required 101 bytes, available 100 bytes"
        );
    }

    #[test]
    fn pure_preflight_rejects_peak_staging_over_budget_as_e15() {
        let msg = pure_preflight(100, 0, 100, 70, 10, 21, 0, 0, 1)
            .unwrap_err()
            .to_string();
        assert_eq!(
            msg,
            "Vulkan memory is insufficient: required 101 bytes, available 100 bytes"
        );
    }

    #[test]
    fn error_matrix_e17_rejects_context_over_budget() {
        let msg = pure_preflight(100, 0, 100, 70, 10, 0, 15, 6, 4096)
            .unwrap_err()
            .to_string();
        assert_eq!(
            msg,
            "context 4096 does not fit the selected backend; context was not reduced"
        );
    }

    #[test]
    fn pure_preflight_arithmetic_overflow_is_reported() {
        let msg = pure_preflight(u64::MAX, 0, 100, u64::MAX, 1, 0, 0, 0, 1)
            .unwrap_err()
            .to_string();
        assert_eq!(
            msg,
            "Vulkan memory is insufficient: required overflow bytes, available 18446744073709551615 bytes"
        );
    }

    #[test]
    fn reserve_bytes_override_and_default() {
        // Override wins (MiB → bytes), even when larger than VRAM.
        assert_eq!(reserve_bytes(8 << 30, Some(512)), 512 * 1024 * 1024);
        assert_eq!(reserve_bytes(1 << 20, Some(4096)), 4096 * 1024 * 1024);
        // Default = max(256 MiB, 5%). Large VRAM ⇒ 5% wins.
        assert_eq!(reserve_bytes(16u64 << 30, None), (16u64 << 30) / 20);
        // Small VRAM ⇒ the 256 MiB floor wins.
        assert_eq!(reserve_bytes(1u64 << 30, None), 256 * 1024 * 1024);
        assert_eq!(reserve_bytes(u64::MAX, Some(u64::MAX)), u64::MAX);
    }
}
