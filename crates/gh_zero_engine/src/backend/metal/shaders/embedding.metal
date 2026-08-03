/*
 * gh_zero_engine — Metal embedding kernel family
 * Numeric implementation is isolated to this single operation and owns no I/O.
 */

// AGENTS deroga K: varianti coese della sola operazione embedding.

#include <metal_stdlib>
using namespace metal;

kernel void metal_embedding_stub(uint id [[thread_position_in_grid]]) {
    (void)id;
}
