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
pub(crate) mod rope;
pub(crate) mod source;

pub(crate) use contract::Backend;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;

#[cfg(feature = "cpu")]
pub(crate) mod cpu;

#[cfg(not(any(feature = "vulkan", feature = "cpu")))]
compile_error!(
    "no backend selected: build with exactly one of --features vulkan or --features cpu"
);

#[cfg(all(feature = "vulkan", feature = "cpu", not(feature = "hybrid")))]
compile_error!(
    "vulkan and cpu backends are mutually exclusive: enable exactly one, or use --features hybrid to compile both for the split runtime"
);
