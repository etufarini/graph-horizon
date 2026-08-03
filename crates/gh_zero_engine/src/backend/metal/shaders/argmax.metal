/*
 * gh_zero_engine — Metal argmax reduction kernel family
 * Numeric implementation is isolated to this single operation and owns no I/O.
 */

// AGENTS deroga K: varianti coese della sola operazione argmax reduction.

#include <metal_stdlib>
using namespace metal;

kernel void metal_argmax_stub(uint id [[thread_position_in_grid]]) {
    (void)id;
}
