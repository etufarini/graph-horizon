/*
 * graph_horizon_engine — backend module wiring
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

#[cfg(all(feature = "cuda", target_os = "linux", target_arch = "x86_64"))]
pub(crate) mod cuda;

#[cfg(all(
    feature = "cuda",
    not(all(target_os = "linux", target_arch = "x86_64"))
))]
compile_error!("cuda profile requires Linux x86_64");

#[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
pub(crate) mod vulkan;

#[cfg(all(
    any(feature = "metal", feature = "metal-hybrid"),
    target_os = "macos",
    target_arch = "aarch64"
))]
pub(crate) mod metal;

#[cfg(all(
    any(feature = "metal", feature = "metal-hybrid"),
    not(all(target_os = "macos", target_arch = "aarch64"))
))]
compile_error!("metal profiles require macOS on Apple Silicon");

#[cfg(any(feature = "cpu", feature = "vulkan-hybrid", feature = "metal-hybrid"))]
pub(crate) mod cpu;

#[cfg(not(any(
    feature = "cpu",
    feature = "vulkan",
    feature = "vulkan-hybrid",
    feature = "metal",
    feature = "metal-hybrid",
    feature = "cuda"
)))]
compile_error!(
    "no backend selected: choose exactly one of cpu, vulkan, vulkan-hybrid, metal, metal-hybrid, or cuda"
);

#[cfg(any(
    all(feature = "cpu", feature = "vulkan"),
    all(feature = "cpu", feature = "vulkan-hybrid"),
    all(feature = "cpu", feature = "metal"),
    all(feature = "vulkan", feature = "vulkan-hybrid"),
    all(feature = "vulkan", feature = "metal"),
    all(feature = "vulkan-hybrid", feature = "metal"),
    all(feature = "cpu", feature = "metal-hybrid"),
    all(feature = "vulkan", feature = "metal-hybrid"),
    all(feature = "vulkan-hybrid", feature = "metal-hybrid"),
    all(feature = "metal", feature = "metal-hybrid"),
    all(feature = "cuda", feature = "cpu"),
    all(feature = "cuda", feature = "vulkan"),
    all(feature = "cuda", feature = "vulkan-hybrid"),
    all(feature = "cuda", feature = "metal"),
    all(feature = "cuda", feature = "metal-hybrid")
))]
compile_error!(
    "multiple backend profiles selected: choose exactly one of cpu, vulkan, vulkan-hybrid, metal, metal-hybrid, or cuda"
);
