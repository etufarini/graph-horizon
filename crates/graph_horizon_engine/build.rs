/*
 * graph_horizon_engine — feature-gated offline GPU source compilation.
 */

#[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
#[path = "build/vulkan.rs"]
mod vulkan;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(hybrid_backend)");
    println!("cargo:rustc-check-cfg=cfg(runtime_bytes)");
    if cfg!(feature = "vulkan-hybrid") {
        println!("cargo:rustc-cfg=hybrid_backend");
    }
    if cfg!(feature = "vulkan-hybrid") {
        println!("cargo:rustc-cfg=runtime_bytes");
    }
    let selected = [
        cfg!(feature = "cpu"),
        cfg!(feature = "vulkan"),
        cfg!(feature = "vulkan-hybrid"),
    ];
    if selected.into_iter().filter(|enabled| *enabled).count() != 1 {
        return;
    }

    #[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
    vulkan::build();

    println!("cargo:rerun-if-changed=build.rs");
}
