/*
 * gh_zero_engine — Metal RMS normalization kernel family
 * Numeric implementation is isolated to this single operation and owns no I/O.
 */

// AGENTS deroga K: varianti coese della sola operazione RMS normalization.

#include <metal_stdlib>
using namespace metal;

kernel void metal_rmsnorm_stub(uint id [[thread_position_in_grid]]) {
    (void)id;
}
