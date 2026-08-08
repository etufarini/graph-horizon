// Esecuzione del backend Vulkan: registrazione dispatch, comandi, readback host.
// `dispatch` e `readback` sono `pub(super)` perché i delegatori di `impl Backend`
// in `vulkan/mod.rs` li invocano; `commands` (impl su `Device`) resta privato.
mod commands;
pub(super) mod dispatch;
#[cfg(feature = "vulkan-profile")]
pub(crate) mod profile;
pub(super) mod readback;
