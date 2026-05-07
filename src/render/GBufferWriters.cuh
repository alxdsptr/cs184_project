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

// Pack a float4 into a ushort4 carrying four __half bit patterns. RGBA16F
// surfaces are 8 bytes/texel — never write `float4` (16 bytes) at `x * 8`,
// the second half spills into the next pixel and silently corrupts the NRD
// inputs (looks exactly like "the denoiser has no effect").
__device__ inline ushort4 packRGBA16F(float4 v) {
    ushort4 r;
    r.x = __half_as_ushort(__float2half(v.x));
    r.y = __half_as_ushort(__float2half(v.y));
    r.z = __half_as_ushort(__float2half(v.z));
    r.w = __half_as_ushort(__float2half(v.w));
    return r;
}

// Pack four [0,1] floats into a single uint32 with byte layout (LE):
// byte0=r, byte1=g, byte2=b, byte3=a. Workaround: surf2Dwrite<uchar4> in
// OptiX-compiled device code emits a PTX store that faults with
// "misaligned address" on Ampere+ (verified on RTX 4070, OptiX 9.0); writing
// the same 4 bytes as a uint32_t works. CUDA-compiled code can use the
// uchar4 path directly and doesn't need this.
__device__ inline uint32_t packRGBA8_uint(float r, float g, float b, float a) {
    uint32_t br = (uint32_t)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t bg = (uint32_t)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t bb = (uint32_t)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t ba = (uint32_t)(fminf(fmaxf(a, 0.0f), 1.0f) * 255.0f + 0.5f);
    return (ba << 24) | (bb << 16) | (bg << 8) | br;
}

// ── Conditional surface writes (no-op if the handle is zero) ─────
// Skip the `if (surf)` boilerplate at every call site. surf2Dwrite expects a
// BYTE offset for x — the second arg of each helper is the texel-stride
// already factored in.

// R32F: 4 bytes/texel.
__device__ inline void writeR32F(cudaSurfaceObject_t surf, uint32_t x, uint32_t y, float v) {
    if (surf) surf2Dwrite<float>(v, surf, x * 4, y);
}

// RG16F: 4 bytes/texel, written as a packed ushort2 (a pair of __half).
__device__ inline void writeRG16F(cudaSurfaceObject_t surf, uint32_t x, uint32_t y, float2 v) {
    if (surf) surf2Dwrite<ushort2>(packRG16F(v.x, v.y), surf, x * 4, y);
}

// RGBA16F: 8 bytes/texel, written as a packed ushort4 (four __half).
__device__ inline void writeRGBA16F(cudaSurfaceObject_t surf, uint32_t x, uint32_t y, float4 v) {
    if (surf) surf2Dwrite<ushort4>(packRGBA16F(v), surf, x * 8, y);
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
    writeR32F(viewZSurf, x, y, viewZ);
    writeRG16F(mvSurf,   x, y, mvPx);
    // DLSS expects post-perspective clip.z/clip.w in [0,1] (near=0, far=1).
    writeR32F(ndcDepthSurf, x, y, clampf(clipCurrZ * 0.5f + 0.5f, 0.0f, 1.0f));
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
    writeR32F(viewZSurf,    x, y, 1.0e6f);
    writeRG16F(mvSurf,      x, y, make_float2(0.0f, 0.0f));
    writeR32F(ndcDepthSurf, x, y, 1.0f);
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
