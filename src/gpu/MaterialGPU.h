#pragma once
#include "core/Types.h"
#include <cuda_runtime.h>

struct GPUMaterial {
    float3 albedo;
    float  roughness;
    float  metallic;
    float  _pad0;
    float3 emission;
    float  emissionStrength;
    float  ior;
    float  transmission;
    float  _pad1;
    float  _pad2;
    cudaTextureObject_t albedoTex;
    cudaTextureObject_t normalTex;
    cudaTextureObject_t metallicRoughTex;
    cudaTextureObject_t emissiveTex;
};
