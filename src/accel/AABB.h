#pragma once
#include "core/Types.h"
#include "core/Math.h"

struct AABB {
    float3 bmin;
    float3 bmax;

    HD AABB() : bmin(make_float3(1e30f,1e30f,1e30f)), bmax(make_float3(-1e30f,-1e30f,-1e30f)) {}
    HD AABB(float3 mn, float3 mx) : bmin(mn), bmax(mx) {}

    HD void expand(float3 p) {
        bmin = make_float3(fminf(bmin.x,p.x), fminf(bmin.y,p.y), fminf(bmin.z,p.z));
        bmax = make_float3(fmaxf(bmax.x,p.x), fmaxf(bmax.y,p.y), fmaxf(bmax.z,p.z));
    }

    HD void expand(const AABB& other) {
        expand(other.bmin);
        expand(other.bmax);
    }

    HD float surfaceArea() const {
        float3 d = bmax - bmin;
        return 2.0f * (d.x*d.y + d.y*d.z + d.z*d.x);
    }

    HD float3 center() const {
        return (bmin + bmax) * 0.5f;
    }

    // Slab-based ray-AABB intersection
    HD bool intersect(float3 origin, float3 invDir, float tmin, float tmax) const {
        float3 t0 = (bmin - origin) * invDir;
        float3 t1 = (bmax - origin) * invDir;
        float3 tNear = make_float3(fminf(t0.x,t1.x), fminf(t0.y,t1.y), fminf(t0.z,t1.z));
        float3 tFar  = make_float3(fmaxf(t0.x,t1.x), fmaxf(t0.y,t1.y), fmaxf(t0.z,t1.z));
        float tN = fmaxf(fmaxf(tNear.x, tNear.y), fmaxf(tNear.z, tmin));
        float tF = fminf(fminf(tFar.x, tFar.y), fminf(tFar.z, tmax));
        return tN <= tF;
    }
};

// Multiply float3 componentwise (for invDir)
inline HD float3 safeInvDir(float3 dir) {
    return make_float3(
        1.0f / (fabsf(dir.x) > 1e-8f ? dir.x : (dir.x >= 0 ? 1e-8f : -1e-8f)),
        1.0f / (fabsf(dir.y) > 1e-8f ? dir.y : (dir.y >= 0 ? 1e-8f : -1e-8f)),
        1.0f / (fabsf(dir.z) > 1e-8f ? dir.z : (dir.z >= 0 ? 1e-8f : -1e-8f))
    );
}
