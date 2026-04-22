#pragma once
#include <cuda_runtime.h>

struct GPUPointLight {
    float3 position;
    float3 color;
    float  intensity;
    float  constantAttenuation;
    float  linearAttenuation;
    float  quadraticAttenuation;
    int    enabled;  // 0 = disabled (skip in NEE & overlay dims the marker)
};
