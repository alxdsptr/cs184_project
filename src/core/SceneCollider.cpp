#include "core/SceneCollider.h"
#include "accel/SAH_BVH.h"
#include "accel/BVHNode.h"
#include "scene/Scene.h"
#include "scene/Mesh.h"
#include <cmath>
#include <cstdint>

void SceneCollider::clear() {
    m_positions.clear();
    m_indices.clear();
    m_bvh = BVHData{};
    m_bounds = AABB{};
}

void SceneCollider::build(const Scene& scene) {
    clear();

    const auto& meshes = scene.getMeshes();
    size_t totalVerts = 0;
    size_t totalIdx   = 0;
    for (const auto& m : meshes) {
        totalVerts += m.positions.size();
        totalIdx   += m.indices.size();
    }
    m_positions.reserve(totalVerts);
    m_indices.reserve(totalIdx);

    uint32_t vertOffset = 0;
    for (const auto& m : meshes) {
        for (float3 p : m.positions) {
            m_positions.push_back(p);
            m_bounds.expand(p);
        }
        for (uint32_t i : m.indices) {
            m_indices.push_back(i + vertOffset);
        }
        vertOffset += (uint32_t)m.positions.size();
    }

    if (m_indices.empty()) return;

    SAH_BVH builder;
    uint32_t triCount = (uint32_t)(m_indices.size() / 3);
    m_bvh = builder.build(m_positions.data(), m_indices.data(), triCount);
}

bool SceneCollider::raycast(float3 origin, float3 dir, float maxDist,
                            float& tHit, float3& nHit, float tmin) const
{
    if (m_bvh.nodes.empty()) return false;

    float3 invDir = safeInvDir(dir);
    float closestT = maxDist;
    bool anyHit = false;
    float3 hitN = make_float3(0, 1, 0);

    // Iterative traversal — same shape as bvh_closestHit, but host-only and
    // we re-fetch the original (unreordered) indices via orderedPrimIndices.
    const BVHNode* nodes = m_bvh.nodes.data();
    const uint32_t* prim = m_bvh.orderedPrimIndices.data();
    const float3* positions = m_positions.data();
    const uint32_t* indices = m_indices.data();

    uint32_t stack[64];
    int sp = 0;
    stack[sp++] = m_bvh.rootIndex;

    while (sp > 0) {
        uint32_t nodeIdx = stack[--sp];
        const BVHNode& node = nodes[nodeIdx];
        if (!node.bounds.intersect(origin, invDir, tmin, closestT)) continue;

        if (node.isLeaf()) {
            for (uint32_t i = 0; i < node.primCount; i++) {
                uint32_t triIdx = prim ? prim[node.primOffset + i]
                                       : (node.primOffset + i);
                uint32_t i0 = indices[triIdx * 3 + 0];
                uint32_t i1 = indices[triIdx * 3 + 1];
                uint32_t i2 = indices[triIdx * 3 + 2];
                float3 v0 = positions[i0];
                float3 v1 = positions[i1];
                float3 v2 = positions[i2];

                float3 e1 = v1 - v0;
                float3 e2 = v2 - v0;
                float3 h  = cross(dir, e2);
                float a   = dot(e1, h);
                if (fabsf(a) < 1e-8f) continue;
                float f = 1.0f / a;
                float3 s = origin - v0;
                float u = f * dot(s, h);
                if (u < 0.0f || u > 1.0f) continue;
                float3 q = cross(s, e1);
                float v = f * dot(dir, q);
                if (v < 0.0f || u + v > 1.0f) continue;
                float t = f * dot(e2, q);
                if (t > tmin && t < closestT) {
                    closestT = t;
                    anyHit = true;
                    float3 gn = normalize(cross(e1, e2));
                    if (dot(gn, dir) > 0.0f) gn = -gn;
                    hitN = gn;
                }
            }
        } else {
            stack[sp++] = node.leftChild;
            stack[sp++] = node.rightChild;
        }
    }

    if (anyHit) {
        tHit = closestT;
        nHit = hitN;
    }
    return anyHit;
}
