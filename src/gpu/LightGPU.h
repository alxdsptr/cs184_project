#pragma once
#include <cuda_runtime.h>

struct GPUPointLight {
    float3 position;
    float3 color;
    float  intensity;
    float  constantAttenuation;
    float  linearAttenuation;
    float  quadraticAttenuation;
};

struct GPUDirectionalLight {
    float3 direction;
    float3 color;
};
