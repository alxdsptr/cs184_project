#pragma once
#include <string>
#include <cuda_runtime.h>

struct PBRMaterial {
    float3 albedo      = {0.8f, 0.8f, 0.8f};
    float  roughness   = 0.5f;
    float  metallic    = 0.0f;
    float3 emission    = {0.0f, 0.0f, 0.0f};
    float  emissionStrength = 0.0f;
    float  ior         = 1.5f;
    float  transmission = 0.0f;

    std::string albedoTexPath;
    std::string normalTexPath;
    std::string metallicRoughTexPath;
    std::string emissiveTexPath;
};
