#pragma once
#include "gpu/AreaLightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/LightGPU.h"
#include "accel/BVHNode.h"
#include <cuda_runtime.h>

struct DeviceSceneData {
    float3*      d_positions      = nullptr;
    float3*      d_normals        = nullptr;
    float2*      d_uvs            = nullptr;
    uint32_t*    d_indices         = nullptr;
    GPUMaterial* d_materials       = nullptr;
    int*         d_materialIndices = nullptr;
    uint32_t     totalTriangles    = 0;
    uint32_t     totalVertices     = 0;
    uint32_t     materialCount     = 0;
    GPUPointLight* d_pointLights   = nullptr;
    uint32_t       pointLightCount = 0;
    GPUAreaLight* d_areaLights     = nullptr;
    float*       d_areaLightCDF    = nullptr;
    uint32_t     areaLightCount    = 0;
    float        areaLightTotalWeight = 0.0f;
    int*         d_triangleAreaLightIndex = nullptr;
    BVHNode*     d_bvhNodes        = nullptr;
    uint32_t     bvhRootIndex      = 0;
};

class Scene;

class DeviceScene {
public:
    void upload(const Scene& scene);
    void free();
    DeviceSceneData getData() const { return m_data; }

private:
    DeviceSceneData m_data;
};
