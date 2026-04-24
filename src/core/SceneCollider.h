#pragma once
#include "accel/BVH.h"
#include "accel/AABB.h"
#include "core/Types.h"
#include <vector>
#include <cstdint>

class Scene;

class SceneCollider {
public:
    void build(const Scene& scene);
    void clear();
    bool ready() const { return !m_indices.empty(); }

    const AABB& bounds() const { return m_bounds; }

    // Cast a ray against the scene. Returns true on hit; writes hit distance
    // into `tHit` (in [tmin, maxDist]) and the geometric face normal into
    // `nHit` (oriented opposite to the ray direction).
    bool raycast(float3 origin, float3 dir, float maxDist,
                 float& tHit, float3& nHit,
                 float tmin = 1e-3f) const;

private:
    std::vector<float3>   m_positions;
    std::vector<uint32_t> m_indices;
    BVHData               m_bvh;
    AABB                  m_bounds;
};
