/*
 * gh_zero_engine — Metal YaRN rotary embedding kernel family
 * Numeric implementation is isolated to this single operation and owns no I/O.
 */

// AGENTS deroga K: varianti coese della sola operazione YaRN rotary embedding.

#include <metal_stdlib>
using namespace metal;

kernel void metal_rope_stub(uint id [[thread_position_in_grid]]) {
    (void)id;
}
