/*
 * graph_horizon_engine — feature-gated offline GPU source compilation.
 */

#[cfg(feature = "cuda")]
#[path = "build/cuda.rs"]
mod cuda;
#[cfg(any(feature = "metal", feature = "metal-hybrid"))]
#[path = "build/metal.rs"]
mod metal;
#[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
#[path = "build/vulkan.rs"]
mod vulkan;

fn main() {
    let selected = [
        cfg!(feature = "cpu"),
        cfg!(feature = "vulkan"),
        cfg!(feature = "vulkan-hybrid"),
        cfg!(feature = "metal"),
        cfg!(feature = "metal-hybrid"),
        cfg!(feature = "cuda"),
    ];
    if selected.into_iter().filter(|enabled| *enabled).count() != 1 {
        return;
    }

    #[cfg(any(feature = "vulkan", feature = "vulkan-hybrid"))]
    vulkan::build();
    #[cfg(any(feature = "metal", feature = "metal-hybrid"))]
    metal::build();
    #[cfg(feature = "cuda")]
    cuda::build();
}
