/*
 * gh_zero_engine — Metal projection kernel family
 * Numeric implementation is isolated to this single operation and owns no I/O.
 */

// AGENTS deroga K: varianti coese della sola operazione projection.

#include <metal_stdlib>
using namespace metal;

kernel void metal_matmul_stub(uint id [[thread_position_in_grid]]) {
    (void)id;
}
