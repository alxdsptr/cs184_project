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

    // Bottom-up refit support. `nodesByLevel[level]` lists every node at that
    // depth (level 0 = leaves, level N = root). The light-BVH refit kernel
    // launches one wave per level from 0 upward; each thread merges its
    // node's children into a fresh AABB + summed weight. Internal-only on
    // levels >= 1 (level 0 is leaves whose bounds come from the per-light
    // update kernel).
    std::vector<std::vector<uint32_t>> nodesByLevel;
};

// Input: one AABB + sampling weight per light (indexed by light ID).
// Output: flat BVH tree + reordered light index list.
class LightBVH {
public:
    LightBVHData build(const AABB* bounds, const float* weights, uint32_t lightCount);
};
