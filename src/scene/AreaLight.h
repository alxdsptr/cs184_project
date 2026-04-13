#pragma once
#include <cuda_runtime.h>

struct TriangleAreaLight {
    float3 v0      = make_float3(0.0f, 0.0f, 0.0f);
    float3 e1      = make_float3(0.0f, 0.0f, 0.0f);
    float3 e2      = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal  = make_float3(0.0f, 0.0f, 1.0f);
    float3 emission = make_float3(0.0f, 0.0f, 0.0f);
    float  area    = 0.0f;
    float  weight  = 0.0f;
};
