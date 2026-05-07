#pragma once
// Vulkan-shared g-buffer surface writers used by every primary-hit path:
// PathTraceKernel.cu / PathTraceKernelSplit.cu / OptiXPrograms.cu.
// Layout matches the surface formats declared in PathTraceKernel.h
// (PrimaryHitSurfaces / SplitSurfaceOutputs):
//   viewZ          : R32F   (linear meters; sentinel 1e6 for sky)
//   motionVectors  : RG16F  (pixel-space, prev-curr; zero for sky)
//   ndcDepth       : R32F   (clip.z*0.5 + 0.5; 1.0 for sky)
//
// Helpers are header-only (`__device__ inline`) so OptiX raygens and CUDA
// kernels share a single source of truth without ODR conflicts. Each TU that
// includes this header must already pull in `<cuda_runtime.h>` (CUDA) or
// `<surface_indirect_functions.h>` (OptiX) for surf2Dwrite to be available;
// since both consumers do, no extra include is needed here.

#include "core/Math.h"
#include "core/Camera.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Pack two float→__half values into a ushort2 for an RG16F surface write.
// Avoids pointer-reinterpret tricks that have caused alignment issues in some
// OptiX toolchain versions; uses __half_as_ushort directly.
__device__ inline ushort2 packRG16F(float x, float y) {
    ushort2 r;
    r.x = __half_as_ushort(__float2half(x));
    r.y = __half_as_ushort(__float2half(y));
    return r;
}

// Write the primary-hit g-buffer entries (viewZ, motion vector, NDC depth) to
// any surfaces that are bound (zero-handle = skip). DLSS / NRD only need the
// first sample's hit to win — the caller gates with `firstBounce && !gbufferWritten`
// so this is called once per pixel.
__device__ inline void writePrimaryGBufferSurfaces(
    cudaSurfaceObject_t viewZSurf,
    cudaSurfaceObject_t mvSurf,
    cudaSurfaceObject_t ndcDepthSurf,
    uint32_t x, uint32_t y,
    float viewZ, float2 mvPx, float clipCurrZ)
{
    if (viewZSurf) {
        surf2Dwrite<float>(viewZ, viewZSurf, x * 4, y);
    }
    if (mvSurf) {
        surf2Dwrite<ushort2>(packRG16F(mvPx.x, mvPx.y), mvSurf, x * 4, y);
    }
    if (ndcDepthSurf) {
        // DLSS expects post-perspective clip.z/clip.w in [0,1] (near=0, far=1).
        float ndcZ = clampf(clipCurrZ * 0.5f + 0.5f, 0.0f, 1.0f);
        surf2Dwrite<float>(ndcZ, ndcDepthSurf, x * 4, y);
    }
}

// Sky-pixel sentinel: viewZ = 1e6 (beyond denoising range), motion = 0
// (sky doesn't reproject by camera), NDC depth = 1 (far plane). Same gating
// rules as writePrimaryGBufferSurfaces — call once per pixel from the miss
// branch when firstBounce is still true.
__device__ inline void writeSkyGBufferSentinel(
    cudaSurfaceObject_t viewZSurf,
    cudaSurfaceObject_t mvSurf,
    cudaSurfaceObject_t ndcDepthSurf,
    uint32_t x, uint32_t y)
{
    if (viewZSurf) {
        surf2Dwrite<float>(1.0e6f, viewZSurf, x * 4, y);
    }
    if (mvSurf) {
        ushort2 zero = make_ushort2(0, 0);
        surf2Dwrite<ushort2>(zero, mvSurf, x * 4, y);
    }
    if (ndcDepthSurf) {
        surf2Dwrite<float>(1.0f, ndcDepthSurf, x * 4, y);
    }
}

// Pure math: project hit position through current and previous view-projection
// matrices, derive linear viewZ, screen-space motion vector (prev - curr to
// match DLSS / NRD's "where was this pixel last frame" convention), and the
// post-perspective clip Z used for the NDC-depth surface write.
__device__ inline void computePrimaryReproject(
    const CameraParams& camera,
    float3 hitPos, float3 hitPosPrev,
    uint32_t width, uint32_t height,
    float& outViewZ, float2& outMvPx, float& outClipCurrZ)
{
    outViewZ = dot(hitPos - camera.position, camera.forward);
    float3 clipCurr = mat4_transformPoint(camera.viewProjMatrix,     hitPos);
    float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hitPosPrev);
    float2 screenCurr = make_float2((clipCurr.x + 1.0f) * 0.5f * (float)width,
                                    (1.0f - clipCurr.y) * 0.5f * (float)height);
    float2 screenPrev = make_float2((clipPrev.x + 1.0f) * 0.5f * (float)width,
                                    (1.0f - clipPrev.y) * 0.5f * (float)height);
    outMvPx = screenPrev - screenCurr;
    outClipCurrZ = clipCurr.z;
}
