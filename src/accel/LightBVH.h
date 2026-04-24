#pragma once
#include "accel/LightBVHNode.h"
#include <vector>
#include <cstdint>

struct LightBVHData {
    std::vector<LightBVHNode> nodes;
    // Reordered light indices (contiguous per leaf). Indexed by
    // [node.primOffset, node.primOffset + node.primCount).
    std::vector<uint32_t>     orderedLightIndices;
    uint32_t                  rootIndex = 0;
};

// Input: one AABB + sampling weight per light (indexed by light ID).
// Output: flat BVH tree + reordered light index list.
class LightBVH {
public:
    LightBVHData build(const AABB* bounds, const float* weights, uint32_t lightCount);
};
