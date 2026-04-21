#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

struct TriangleMesh {
    std::vector<float3>   positions;
    std::vector<float3>   normals;
    std::vector<float4>   tangents; // xyz = tangent, w = bitangent sign (+/-1)
    std::vector<float2>   uvs;
    std::vector<uint32_t> indices;
    int materialIndex = 0;
};
