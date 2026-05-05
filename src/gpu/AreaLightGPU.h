#pragma once
#include <cuda_runtime.h>

struct GPUAreaLight {
    // Current-frame world-space triangle. For static lights this never
    // changes from the upload-time value. For animated lights the per-frame
    // light-update kernel rewrites these from the *_rest fields below by
    // applying that mesh's pose delta.
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

    // Rest-pose world triangle (the t=0 pose, also what static lights keep
    // forever). Used by the per-frame light update kernel to recompute the
    // current-frame triangle as `meshDelta[meshIndex] * rest`. Static lights
    // (meshIndex == -1) are left at their upload-time values.
    float3 v0_rest;
    float3 e1_rest;
    float3 e2_rest;
    float3 normal_rest;
    // Mesh this light was generated from. -1 = static (no per-frame update).
    int    meshIndex;
};
