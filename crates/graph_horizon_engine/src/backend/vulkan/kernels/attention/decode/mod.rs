/*
 * Vulkan decode-attention boundary: exposes the datatype-specific F16 and INT8
 * routers while keeping their independent capability and fallback policies in
 * focused sibling modules.
 */

mod f16;
mod int8;

pub(crate) use f16::run as f16;
pub(crate) use int8::run as int8;
