#pragma once
#include "accel/BVH.h"
#include <vector>

class SAH_BVH {
public:
    // CPU build: returns flat node array + reordered primitive indices
    BVHData build(const float3* positions, const uint32_t* indices, uint32_t triCount);
};
