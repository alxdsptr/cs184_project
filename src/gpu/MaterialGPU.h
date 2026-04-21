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
    int    pureDiffuse;             // nonzero = bypass Cook-Torrance specular lobe
    int    useSpecularGlossiness;   // nonzero = use SG workflow (F0 from specularColor/Tex)
    int    specularGlossAlphaIsGlossiness; // nonzero = sample tex.a as per-pixel glossiness
    float3 specularColor;           // F0 multiplier (used when useSpecularGlossiness != 0)
    float  glossiness;              // 1 - roughness multiplier
    cudaTextureObject_t albedoTex;
    cudaTextureObject_t normalTex;
    cudaTextureObject_t metallicRoughTex;
    cudaTextureObject_t emissiveTex;
    cudaTextureObject_t specularGlossTex; // RGB=F0, A=glossiness
};
