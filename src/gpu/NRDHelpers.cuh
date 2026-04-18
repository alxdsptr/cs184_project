#pragma once

// NRD front-end helpers ported from NRD.hlsli / REBLUR_Config.hlsli.
//
// NRD options compiled into our build (overridden in top-level CMakeLists.txt):
//   NRD_NORMAL_ENCODING    = 0  (R8G8B8A8_UNORM, plain XYZ normal in RGB)
//   NRD_ROUGHNESS_ENCODING = 1  (linear roughness, alpha channel)
//
// These MUST match the VkFormat used for the normal-roughness aux image
// (VulkanSharedAuxBuffers::m_normalRoughness, currently R8G8B8A8_UNORM). If
// the encoding or image format changes, update both sides together — NRD's
// front-end unpacker reads the texel according to NRD_NORMAL_ENCODING and
// silently returns garbage otherwise.
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

// Pack (normal, roughness) into an RGBA8 UNORM texel.
// Layout matches NRD's NRD_NORMAL_ENCODING_RGBA8_UNORM (= 0) front-end
// unpacker, which does `p.xyz * 2 - 1` and then normalizes:
//   RGB = (normal * 0.5 + 0.5)    — plain XYZ remapped to [0,1]
//   A   = linear roughness         — matches NRD_ROUGHNESS_ENCODING_LINEAR
__device__ __forceinline__ float4 packNormalRoughness(float3 normal, float roughness) {
    return make_float4(
        normal.x * 0.5f + 0.5f,
        normal.y * 0.5f + 0.5f,
        normal.z * 0.5f + 0.5f,
        fminf(fmaxf(roughness, 0.0f), 1.0f));
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
