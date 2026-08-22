// Vulkan execution namespace: dispatch routing and host readback are visible to
// the backend delegators, while command-buffer lifecycle helpers remain private.
mod commands;
pub(super) mod dispatch;
pub(super) mod readback;
