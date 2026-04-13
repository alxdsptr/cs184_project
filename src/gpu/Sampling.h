#pragma once
#include "core/Types.h"
#include "core/Math.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// Cosine-weighted hemisphere sampling
inline D float3 sampleCosineHemisphere(float u1, float u2, float& pdf) {
    float r   = sqrtf(u1);
    float phi = 2.0f * M_PI_F * u2;
    float x = r * cosf(phi);
    float z = r * sinf(phi);
    float y = sqrtf(fmaxf(0.0f, 1.0f - u1));
    pdf = y / M_PI_F;
    return make_float3(x, y, z);
}

// Build orthonormal basis from normal
inline D void buildONB(float3 N, float3& T, float3& B) {
    float3 a = (fabsf(N.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    B = normalize(cross(N, a));
    T = cross(B, N);
}

// Transform direction from local (Y-up) to world space using ONB
inline D float3 localToWorld(float3 local, float3 T, float3 N, float3 B) {
    return T * local.x + N * local.y + B * local.z;
}
