#pragma once
#include "accel/AABB.h"
#include <cstdint>

struct BVHNode {
    AABB bounds;
    uint32_t primCount = 0;
    union {
        uint32_t leftChild;
        uint32_t primOffset;
    };
    uint32_t rightChild = 0;

    HD bool isLeaf() const { return primCount > 0; }
};
