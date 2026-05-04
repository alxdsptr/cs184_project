#pragma once
#include <cuda_runtime.h>
#include "core/Math.h"

struct GPUAreaLight {
    float3 v0;
    float3 e1;
    float3 e2;
    float3 normal;
    // For uniform emitters: direct emission. For textured emitters: an
    // albedo/intensity multiplier applied to the texel color fetched via
    // emissiveTex at NEE time.
    float3 emission;
    float  area;
    float  weight;

    // UVs for sampling the emissive texture. Valid when emissiveTex != 0.
    float2 uv0;
    float2 uv1;
    float2 uv2;

    cudaTextureObject_t emissiveTex;  // 0 = no texture, use `emission` directly
};

// Runtime "spotlight override" applied to every triangle area light. When
// `enabled == 0` returns 1 unconditionally — vanilla area-light behavior.
// Otherwise restricts emission to a cone around the triangle normal, with a
// smooth cubic fade across the outer slice of the cone so beam edges don't
// look like cardboard. `cosCutoff` is cos(halfAngle); `softness` ∈ [0,1] is
// the fraction of (1 - cosCutoff) used for the fade ramp.
__device__ inline float spotlightAttenuation(int enabled, float cosCutoff,
                                             float softness, float lNdot)
{
    if (!enabled) return 1.0f;
    float halfWidth = fmaxf((1.0f - cosCutoff) * softness, 1e-6f);
    float t = (lNdot - cosCutoff) / halfWidth;
    t = fminf(fmaxf(t, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
