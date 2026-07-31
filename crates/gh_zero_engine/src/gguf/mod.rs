/*
 * gh_zero_engine — GGUF module
 * Low-level, architecture-agnostic GGUF access: an mmap-based loader, a typed
 * metadata view and a name→tensor index. It contains no family specialization;
 * model-specific validation and tensor naming live under `family`.
*/

pub(crate) mod loader;
pub(crate) mod metadata;
pub(crate) mod tensor_index;
