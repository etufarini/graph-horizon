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

#[cfg(any(feature = "vulcan", feature = "vulcan-hybrid"))]
pub(crate) mod vulkan;

#[cfg(any(feature = "cpu", feature = "vulcan-hybrid"))]
pub(crate) mod cpu;

#[cfg(not(any(feature = "cpu", feature = "vulcan", feature = "vulcan-hybrid")))]
compile_error!(
    "no backend selected: choose exactly one of cpu, vulcan, vulcan-hybrid, metal, or metal-hybrid"
);

#[cfg(any(
    all(feature = "cpu", feature = "vulcan"),
    all(feature = "cpu", feature = "vulcan-hybrid"),
    all(feature = "vulcan", feature = "vulcan-hybrid")
))]
compile_error!(
    "multiple backend profiles selected: choose exactly one of cpu, vulcan, vulcan-hybrid, metal, or metal-hybrid"
);
