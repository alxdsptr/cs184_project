#pragma once
#include <cuda_runtime.h>

struct PointLight {
    float3 position = make_float3(0.0f, 0.0f, 0.0f);
    float3 color    = make_float3(1.0f, 1.0f, 1.0f);
    float  intensity = 1.0f;
    float  constantAttenuation  = 1.0f;
    float  linearAttenuation    = 0.0f;
    float  quadraticAttenuation = 0.0f;
};

struct DirectionalLight {
    float3 direction = make_float3(0.0f, -1.0f, 0.0f);
    float3 color     = make_float3(1.0f, 1.0f, 1.0f);
};
