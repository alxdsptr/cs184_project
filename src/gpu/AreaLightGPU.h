#pragma once
#include <cuda_runtime.h>

struct GPUAreaLight {
    float3 v0;
    float3 e1;
    float3 e2;
    float3 normal;
    float3 emission;
    float  area;
    float  weight;
};
