/*
 * graph_horizon_engine — feature-gated offline GPU source compilation.
 */

#[cfg(any(feature = "cuda", feature = "cuda-hybrid"))]
#[path = "build/cuda.rs"]
mod cuda;
#[cfg(any(feature = "metal", feature = "metal-hybrid"))]
#[path = "build/metal.rs"]
mod metal;
#[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
#[path = "build/vulkan.rs"]
mod vulkan;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(hybrid_backend)");
    println!("cargo:rustc-check-cfg=cfg(runtime_bytes)");
    if cfg!(any(
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda-hybrid"
    )) {
        println!("cargo:rustc-cfg=hybrid_backend");
    }
    if cfg!(any(
        feature = "metal",
        feature = "vulkan-hybrid",
        feature = "metal-hybrid",
        feature = "cuda",
        feature = "cuda-hybrid"
    )) {
        println!("cargo:rustc-cfg=runtime_bytes");
    }
    let selected = [
        cfg!(feature = "cpu"),
        cfg!(feature = "vulkan"),
        cfg!(feature = "vulkan-hybrid"),
        cfg!(feature = "metal"),
        cfg!(feature = "metal-hybrid"),
        cfg!(feature = "cuda"),
        cfg!(feature = "cuda-hybrid"),
    ];
    if selected.into_iter().filter(|enabled| *enabled).count() != 1 {
        return;
    }

    #[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
    vulkan::build();
    #[cfg(any(feature = "metal", feature = "metal-hybrid"))]
    metal::build();
    #[cfg(any(feature = "cuda", feature = "cuda-hybrid"))]
    cuda::build();
    println!("cargo:rerun-if-changed=build.rs");
}
