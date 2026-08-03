/*
 * gh_zero_engine — Metal residual addition kernel family
 * Numeric implementation is isolated to this single operation and owns no I/O.
 */

// AGENTS deroga K: varianti coese della sola operazione residual addition.

#include <metal_stdlib>
using namespace metal;

kernel void metal_residual_add_stub(uint id [[thread_position_in_grid]]) {
    (void)id;
}
