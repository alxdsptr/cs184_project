#pragma once
#include <cuda_runtime.h>

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
