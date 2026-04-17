#pragma once

// NRD front-end helpers ported from NRD.hlsli / REBLUR_Config.hlsli.
//
// NRD options compiled into our build (see external/NRD CMake at the top):
//   NRD_NORMAL_ENCODING    = 2  (R8G8B8A8_UNORM, oct-encoded normals)
//   NRD_ROUGHNESS_ENCODING = 1  (linear roughness, alpha channel)
//
// Keep this header self-contained so the path-trace kernel can include it
// without dragging in Vulkan or SDK headers.

#include "core/Math.h"
#include <cuda_runtime.h>
#include <cstdint>

#ifndef NRD_M_PI_F
#  define NRD_M_PI_F 3.14159265358979323846f
#endif

namespace nrd_helpers {

// ── Normal encoding 2: oct-wrap into 2×[0,1] then scale+bias to UNORM ───
__device__ __forceinline__ float2 octWrap(float2 v) {
    float2 s = { v.x >= 0.0f ? 1.0f : -1.0f,
                 v.y >= 0.0f ? 1.0f : -1.0f };
    return make_float2((1.0f - fabsf(v.y)) * s.x,
                       (1.0f - fabsf(v.x)) * s.y);
}

__device__ __forceinline__ float2 encodeOctNormal(float3 n) {
    float absSum = fabsf(n.x) + fabsf(n.y) + fabsf(n.z);
    float inv = 1.0f / fmaxf(absSum, 1e-6f);
    float2 p = make_float2(n.x * inv, n.y * inv);
    if (n.z < 0.0f) p = octWrap(p);
    // Map [-1,1] → [0,1] for UNORM storage.
    return make_float2(p.x * 0.5f + 0.5f, p.y * 0.5f + 0.5f);
}

// Pack (normal, roughness) into an RGBA8 UNORM texel. Layout:
//   RG = oct-encoded normal (2×8 bit)
//   B  = material ID / unused → 0
//   A  = linear roughness
__device__ __forceinline__ float4 packNormalRoughness(float3 normal, float roughness) {
    float2 oct = encodeOctNormal(normal);
    return make_float4(oct.x, oct.y, 0.0f, fminf(fmaxf(roughness, 0.0f), 1.0f));
}

// NRD "YCoCg" radiance packing is optional; RELAX accepts straight RGB in
// the RGB channels with hitT in alpha. Simpler — we go with that.
__device__ __forceinline__ float4 packRadianceHitDist(float3 radiance, float hitDist) {
    // NRD RELAX clamps hitT internally; we clamp to a sane range here to keep
    // FP16 output from blowing up on glancing misses.
    float hd = fminf(fmaxf(hitDist, 0.0f), 65504.0f);
    return make_float4(radiance.x, radiance.y, radiance.z, hd);
}

// View-space Z (signed, linear). NRD CommonSettings::isOrthoProjection=false
// expects positive values in front of camera by default; the RELAX plugin
// normalizes internally. We pass positive values.
__device__ __forceinline__ float computeViewZ(float3 hitPos, float3 camPos, float3 camFwd) {
    return dot(hitPos - camPos, camFwd);
}

// Pixel-space motion vector from current and previous clip-space positions.
// Convention: screen-space delta = prev - curr (NRD treats MV as reprojection
// offset, "where was this pixel last frame" — hence prev minus curr).
__device__ __forceinline__ float2 computeMotionVectorPx(
    float3 hitPos,
    const float4x4& viewProj,
    const float4x4& prevViewProj,
    uint32_t width, uint32_t height)
{
    float3 clipCurr = mat4_transformPoint(viewProj,     hitPos);
    float3 clipPrev = mat4_transformPoint(prevViewProj, hitPos);
    float2 curr = make_float2((clipCurr.x + 1.0f) * 0.5f * width,
                              (1.0f - clipCurr.y) * 0.5f * height);
    float2 prev = make_float2((clipPrev.x + 1.0f) * 0.5f * width,
                              (1.0f - clipPrev.y) * 0.5f * height);
    return make_float2(prev.x - curr.x, prev.y - curr.y);
}

} // namespace nrd_helpers
