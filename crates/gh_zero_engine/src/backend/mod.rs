/*
 * gh_zero_engine — backend module wiring
 * This file owns only the backend namespace: common model-agnostic support,
 * the feature-selected concrete modules, and the compile-time backend choice.
 * The `Backend` contract itself lives in `contract.rs`; no resource algorithm
 * or kernel dispatch belongs here.
*/

pub(crate) mod buffers;
pub(crate) mod contract;
pub(crate) mod f16;
pub(crate) mod hybrid;
pub(crate) mod rope;
pub(crate) mod selection;
pub(crate) mod source;

pub(crate) use contract::Backend;

#[cfg(any(feature = "vulcan", feature = "vulcan-hybrid"))]
pub(crate) mod vulkan;

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod metal;

#[cfg(all(
    feature = "metal",
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!("metal profiles require macOS on Apple Silicon");

#[cfg(any(feature = "cpu", feature = "vulcan-hybrid"))]
pub(crate) mod cpu;

#[cfg(not(any(
    feature = "cpu",
    feature = "vulcan",
    feature = "vulcan-hybrid",
    feature = "metal"
)))]
compile_error!(
    "no backend selected: choose exactly one of cpu, vulcan, vulcan-hybrid, metal, or metal-hybrid"
);

#[cfg(any(
    all(feature = "cpu", feature = "vulcan"),
    all(feature = "cpu", feature = "vulcan-hybrid"),
    all(feature = "cpu", feature = "metal"),
    all(feature = "vulcan", feature = "vulcan-hybrid"),
    all(feature = "vulcan", feature = "metal"),
    all(feature = "vulcan-hybrid", feature = "metal")
))]
compile_error!(
    "multiple backend profiles selected: choose exactly one of cpu, vulcan, vulcan-hybrid, metal, or metal-hybrid"
);
