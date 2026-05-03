#pragma once
#include "core/Types.h"

// Scene-global homogeneous participating medium.
// `density` is a scalar multiplier over sigma_a/sigma_s to make lookdev easier.
struct HomogeneousMedium {
    bool   enabled = false;
    float3 sigmaA = make_float3(0.0f, 0.0f, 0.0f);
    float3 sigmaS = make_float3(0.0f, 0.0f, 0.0f);
    float  density = 1.0f;
    float  anisotropy = 0.0f;      // Henyey-Greenstein g in [-0.99, 0.99]
    float  maxDistance = 1e6f;     // Clamp for very long segments (env/miss)
};
