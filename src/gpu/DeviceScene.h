#pragma once
#include "gpu/AreaLightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/LightGPU.h"
#include "accel/BVHNode.h"
#include <cuda_runtime.h>

struct DeviceSceneData {
    float3*      d_positions      = nullptr;
    float3*      d_normals        = nullptr;
    float4*      d_tangents       = nullptr; // xyz = tangent, w = bitangent sign
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

    // HDR environment map (equirectangular, float4 texture)
    cudaTextureObject_t envMapTex   = 0;

    // Precomputed L2 (3rd-order) Spherical Harmonics radiance coefficients of
    // the environment map. Nine RGB coefficients (float3 x 9) laid out in the
    // canonical order (l,m) = (0,0) (1,-1) (1,0) (1,1) (2,-2) (2,-1) (2,0)
    // (2,1) (2,2). Used by `evalSHIrradiance` for cheap, noise-free diffuse
    // environment irradiance at any surface normal. When `envUseSH` is 0 or
    // `d_shEnvCoeffs` is null, the renderer falls back to stochastic envmap
    // sampling.
    float3*  d_shEnvCoeffs = nullptr;
    int      envUseSH      = 0;
};

class Scene;

class DeviceScene {
public:
    void upload(const Scene& scene);
    void free();
    DeviceSceneData getData() const { return m_data; }

    // Rewrite only the `enabled` flag of every point light. Cheap — the
    // point-light array is tiny. Used by the debug picker to toggle lights
    // at runtime without reuploading the full scene.
    void updatePointLightsEnabled(const bool* enabledFlags, uint32_t count);

private:
    DeviceSceneData m_data;
};
