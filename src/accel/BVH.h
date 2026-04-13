#pragma once
#include "accel/BVHNode.h"
#include "gpu/RayTypes.h"
#include <vector>
#include <cstdint>

struct BVHData {
    std::vector<BVHNode> nodes;
    std::vector<uint32_t> orderedPrimIndices;
    uint32_t rootIndex = 0;
};

// ── BVH traversal (device-side, inlined) ─────────────────────
inline D bool bvh_closestHit(
    const Ray& ray,
    const BVHNode* nodes,
    uint32_t rootIndex,
    const float3* positions,
    const uint32_t* indices,
    const int* materialIndices,
    HitRecord& hit)
{
    float3 invDir = safeInvDir(ray.direction);
    float closestT = ray.tmax;
    bool anyHit = false;

    // Iterative traversal with fixed-size stack
    uint32_t stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = rootIndex;

    while (stackPtr > 0) {
        uint32_t nodeIdx = stack[--stackPtr];
        const BVHNode& node = nodes[nodeIdx];

        if (!node.bounds.intersect(ray.origin, invDir, ray.tmin, closestT))
            continue;

        if (node.isLeaf()) {
            // Test all triangles in leaf
            for (uint32_t i = 0; i < node.primCount; i++) {
                uint32_t triIdx = node.primOffset + i;
                uint32_t i0 = indices[triIdx * 3 + 0];
                uint32_t i1 = indices[triIdx * 3 + 1];
                uint32_t i2 = indices[triIdx * 3 + 2];

                float3 v0 = positions[i0];
                float3 v1 = positions[i1];
                float3 v2 = positions[i2];

                // Moller-Trumbore intersection
                float3 e1 = v1 - v0;
                float3 e2 = v2 - v0;
                float3 h  = cross(ray.direction, e2);
                float  a  = dot(e1, h);
                if (fabsf(a) < 1e-8f) continue;

                float  f = 1.0f / a;
                float3 s = ray.origin - v0;
                float  u = f * dot(s, h);
                if (u < 0.0f || u > 1.0f) continue;

                float3 q = cross(s, e1);
                float  v = f * dot(ray.direction, q);
                if (v < 0.0f || u + v > 1.0f) continue;

                float t = f * dot(e2, q);
                if (t > ray.tmin && t < closestT) {
                    closestT = t;
                    anyHit = true;

                    hit.t = t;
                    hit.position = ray.origin + ray.direction * t;
                    float3 geoNormal = normalize(cross(e1, e2));
                    hit.normal = geoNormal;
                    hit.shadingNormal = geoNormal; // TODO: interpolate vertex normals
                    hit.uv = make_float2(u, v);
                    hit.materialIndex = materialIndices[triIdx];
                    hit.primitiveIndex = (int)triIdx;
                    hit.frontFace = dot(ray.direction, geoNormal) < 0.0f;
                    if (!hit.frontFace) hit.shadingNormal = -hit.shadingNormal;
                }
            }
        } else {
            stack[stackPtr++] = node.leftChild;
            stack[stackPtr++] = node.rightChild;
        }
    }

    return anyHit;
}

// ── Occlusion test (BDPT-ready) ─────────────────────────────
inline D bool bvh_anyHit(
    float3 origin, float3 target,
    const BVHNode* nodes, uint32_t rootIndex,
    const float3* positions, const uint32_t* indices)
{
    float3 dir = target - origin;
    float tmax = length(dir) - 1e-4f;
    dir = normalize(dir);
    float3 invDir = safeInvDir(dir);

    uint32_t stack[64];
    int stackPtr = 0;
    stack[stackPtr++] = rootIndex;

    while (stackPtr > 0) {
        uint32_t nodeIdx = stack[--stackPtr];
        const BVHNode& node = nodes[nodeIdx];

        if (!node.bounds.intersect(origin, invDir, 1e-4f, tmax))
            continue;

        if (node.isLeaf()) {
            for (uint32_t i = 0; i < node.primCount; i++) {
                uint32_t triIdx = node.primOffset + i;
                uint32_t i0 = indices[triIdx * 3 + 0];
                uint32_t i1 = indices[triIdx * 3 + 1];
                uint32_t i2 = indices[triIdx * 3 + 2];

                float3 v0 = positions[i0];
                float3 e1 = positions[i1] - v0;
                float3 e2 = positions[i2] - v0;
                float3 h  = cross(dir, e2);
                float  a  = dot(e1, h);
                if (fabsf(a) < 1e-8f) continue;

                float  f = 1.0f / a;
                float3 s = origin - v0;
                float  u = f * dot(s, h);
                if (u < 0.0f || u > 1.0f) continue;

                float3 q = cross(s, e1);
                float  v = f * dot(dir, q);
                if (v < 0.0f || u + v > 1.0f) continue;

                float t = f * dot(e2, q);
                if (t > 1e-4f && t < tmax)
                    return true; // occluded
            }
        } else {
            stack[stackPtr++] = node.leftChild;
            stack[stackPtr++] = node.rightChild;
        }
    }
    return false; // visible
}
