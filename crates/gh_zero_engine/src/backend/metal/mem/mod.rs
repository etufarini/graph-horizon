/*
 * gh_zero_engine — Metal memory namespace
 * Exports shared allocation primitives and persistent runtime buffer assembly.
 * Budget and weight representation remain in their focused sibling modules.
 */

pub(crate) mod budget;
pub(crate) mod buffer;
pub(crate) mod buffers;
pub(crate) mod weights;
