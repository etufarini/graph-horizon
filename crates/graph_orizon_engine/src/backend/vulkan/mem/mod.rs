/*
 * graph_orizon_engine — Vulkan memory domain
 * Groups VRAM budgeting, allocation, placement planning, and weight upload.
 * The module owns no execution or hybrid model routing.
 */
pub(crate) mod budget;
pub(crate) mod buffers;
pub(crate) mod memory;
pub(crate) mod weights;
