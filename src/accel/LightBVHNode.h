#pragma once
#include "accel/AABB.h"
#include <cstdint>

// Flat node for the Light BVH. Each node stores the AABB of the emissive
// geometry in its subtree and the summed sampling weight (matches the
// per-light `weight` used by the flat CDF path: area * luminance of emission).
// Leaves hold a contiguous range of light indices in the reordered array.
struct LightBVHNode {
    AABB     bounds;
    float    weight      = 0.0f;  // sum of light weights in this subtree
    uint32_t primCount   = 0;     // 0 => internal node
    union {
        uint32_t leftChild;
        uint32_t primOffset;      // offset into reorderedLightIndices
    };
    uint32_t rightChild  = 0;

    HD bool isLeaf() const { return primCount > 0; }
};
