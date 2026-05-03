#pragma once
#include "gpu/AreaLightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/LightGPU.h"
#include "accel/BVHNode.h"
#include "accel/LightBVHNode.h"
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

    // Spatial acceleration structure over area lights used for importance
    // sampling many lights from a shading point. When d_lightBVHNodes is
    // non-null the path tracer descends the tree (stochastic, shading-point
    // dependent) instead of binary-searching d_areaLightCDF.
    LightBVHNode* d_lightBVHNodes   = nullptr;
    uint32_t      lightBVHRootIndex = 0;
    // orderedLightIndices[i] = original area-light index at ordered slot i
    // (a leaf stores [primOffset, primOffset+primCount) slots, and each slot
    // indexes this array to get the real GPUAreaLight ID).
    uint32_t*     d_lightOrderedIndices = nullptr;
    // Inverse map: lightIndexToSlot[origLightIdx] = ordered slot (for MIS
    // PDF lookup when a BSDF-sampled ray hits a known emissive triangle).
    uint32_t*     d_lightIndexToSlot    = nullptr;

    // ReSTIR DI — when non-null, the main kernel reads the precomputed
    // reservoir at bounce-0 NEE instead of running fresh RIS. Struct type is
    // forward-declared opaque (void*) to avoid pulling render/ReSTIR.h into
    // every translation unit that sees DeviceSceneData.
    void*         d_restirReservoirs    = nullptr;
    int           restirEnabled         = 0;

    // ReSTIR GI — per-pixel resolved indirect-radiance buffer produced by
    // the ReSTIR GI pipeline (init candidates → temporal → spatial → shade).
    // When non-null AND restirGIEnabled != 0, the main kernel consumes this
    // value as the "indirect contribution from the primary hit" and skips
    // its own continuation bounces (sample 0 only — extra spp still path-
    // traces normally so we don't double-count). Layout: float3 per pixel
    // in row-major width × height.
    float3*       d_restirGIIndirect    = nullptr;
    int           restirGIEnabled       = 0;

    // ReSTIR PT (Lin et al. 2022) — same shape as the GI buffer above. A
    // per-pixel float3 indirect-radiance image produced by the PT pipeline
    // (init → temporal → spatial → shade), consumed at the primary hit on
    // sample 0 in lieu of continuation bounces. Mutually exclusive with
    // `d_restirGIIndirect`: when both are bound, PT wins (longer path
    // postfix → strictly more information than GI's 1-bounce NEE).
    float3*       d_restirPTIndirect    = nullptr;
    int           restirPTEnabled       = 0;

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

    // Debug visualization mode for diagnosing normal-map issues. When non-zero,
    // the path-trace kernel short-circuits at the primary hit and writes a
    // false-colour image instead of tracing light transport. 0 = off (normal
    // render); 1 = perturbed world-space normal as RGB; 2 = interpolated
    // tangent handedness (green=+1, blue=-1, red=drifted); 3 = back-face flag
    // after normal-map perturbation (red if dot(N, rayDir) > 0).
    int      debugNormalViz = 0;

    // Master switch for tangent-space normal mapping. When 0, the kernel
    // ignores every material's normalTex and shades against the interpolated
    // vertex normal. Handy for A/B-ing the effect of normal maps.
    int      enableNormalMap = 1;

    // Debug normal-arrow overlay: sparse grid of (position, shading-normal)
    // samples captured at the primary hit. Laid out as N float4 pairs:
    //   [2*i + 0].xyz = world-space position,    .w = valid flag (1=hit, 0=miss)
    //   [2*i + 1].xyz = world-space perturbed N, .w = unused
    // `debugArrowStride` is the pixel stride between sample cells on both
    // axes (e.g. 24 → one arrow every 24×24 pixels). Capacity is
    // ceil(W/stride) * ceil(H/stride). When d_debugArrows is null or
    // debugArrowStride <= 0 the kernel writes nothing.
    float4*  d_debugArrows     = nullptr;
    int      debugArrowStride  = 0;
    int      debugArrowWidth   = 0;   // ceil(W/stride)
    int      debugArrowHeight  = 0;   // ceil(H/stride)
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
