#pragma once
#include "core/Types.h"

struct Ray {
    float3 origin;
    float  tmin;
    float3 direction;
    float  tmax;
};

struct HitRecord {
    float3 position;
    float3 normal;
    float3 shadingNormal;
    float2 uv;
    float  t;
    int    materialIndex;
    int    primitiveIndex;
    bool   frontFace;
};

struct RayPayload {
    float3   throughput;
    float3   radiance;
    Ray      ray;
    uint32_t rngState;
    int      depth;
    bool     terminated;
};

// BDPT-ready: path vertex for storing complete subpaths
struct PathVertex {
    float3 position;
    float3 normal;
    float2 uv;
    int    materialIndex;
    float3 throughput;
    float  pdfFwd;
    float  pdfRev;
    bool   isDelta;
    bool   isOnLight;
};
