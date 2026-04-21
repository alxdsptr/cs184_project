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

// Sample a tangent-space normal map and rotate the interpolated normal into
// the perturbed shading frame. tangent4.w stores the bitangent handedness
// (±1) packed at scene-load time; w == 0 means "no tangent available", in
// which case we return the input N unchanged.
inline D float3 applyNormalMap(
    float3 N, float4 tangent4, cudaTextureObject_t normalTex, float2 uv)
{
    if (normalTex == 0 || tangent4.w == 0.0f) return N;

    float3 T = make_float3(tangent4.x, tangent4.y, tangent4.z);
    // Re-orthogonalize T against N (Gram-Schmidt) so the TBN stays orthonormal
    // even after vertex-normal interpolation across the triangle.
    T = T - N * dot(N, T);
    float Tlen2 = dot(T, T);
    if (Tlen2 < 1e-8f) return N;
    T = T * rsqrtf(Tlen2);
    float3 B = cross(N, T) * tangent4.w;

    float4 nm = tex2D<float4>(normalTex, uv.x, uv.y);
    float3 ts = make_float3(nm.x * 2.0f - 1.0f,
                            nm.y * 2.0f - 1.0f,
                            nm.z * 2.0f - 1.0f);
    float tsLen2 = dot(ts, ts);
    if (tsLen2 < 1e-8f) return N;
    ts = ts * rsqrtf(tsLen2);

    float3 perturbed = T * ts.x + B * ts.y + N * ts.z;
    float pLen2 = dot(perturbed, perturbed);
    if (pLen2 < 1e-8f) return N;
    return perturbed * rsqrtf(pLen2);
}
