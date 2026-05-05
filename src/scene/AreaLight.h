#pragma once
#include <cuda_runtime.h>

struct TriangleAreaLight {
    float3 v0      = make_float3(0.0f, 0.0f, 0.0f);
    float3 e1      = make_float3(0.0f, 0.0f, 0.0f);
    float3 e2      = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal  = make_float3(0.0f, 0.0f, 1.0f);
    // Fallback emission used when the light has no emissive texture (i.e. a
    // uniform emitter). For textured emitters this stores albedo × intensity
    // to multiply the fetched texel color at runtime.
    float3 emission = make_float3(0.0f, 0.0f, 0.0f);
    float  area    = 0.0f;
    float  weight  = 0.0f;

    // UVs for runtime emissive-texture sampling. Unused if emissiveTexObj == 0.
    float2 uv0 = make_float2(0.0f, 0.0f);
    float2 uv1 = make_float2(0.0f, 0.0f);
    float2 uv2 = make_float2(0.0f, 0.0f);

    // Runtime CUDA texture handle for the emissive texture (0 = no texture).
    // This is populated by Application after TextureManager binds CUDA texture
    // objects, then read back by DeviceScene::upload() when building the GPU
    // area light array.
    cudaTextureObject_t emissiveTexObj = 0;

    // Material index this light was generated from. Used by Application to
    // look up the emissive cudaTextureObject_t after texture upload. -1 for
    // lights that do not need a texture (uniform emitters).
    int materialIndex = -1;

    // True if the source mesh sits under a non-animated SceneNode. Static
    // emitters get NEE direct sampling via the light BVH; dynamic emitters
    // are excluded from the BVH (their world-space bounds change every frame)
    // and contribute to illumination only through BSDF hits + MIS.
    bool isStatic = true;
};
