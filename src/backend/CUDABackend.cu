#include "backend/CUDABackend.h"
#include "scene/Scene.h"
#include "accel/SAH_BVH.h"
#include "core/Math.h"
#include "render/PathTraceKernel.h"
#include "render/Tonemapping.h"
#include "util/CudaCheck.h"
#include "util/Log.h"

void CUDABackend::buildAccelerationStructure(Scene& scene) {
    // Upload geometry + materials to GPU
    m_deviceScene.upload(scene);
    auto data = m_deviceScene.getData();

    if (data.totalTriangles == 0) {
        LOG_WARN("CUDABackend: no triangles to build BVH");
        return;
    }

    // Build SAH BVH on CPU using the flattened position/index data
    // We need host copies for the CPU BVH builder
    const auto& meshes = scene.getMeshes();

    // Flatten positions and indices on host (same layout as DeviceScene)
    std::vector<float3>   hostPositions;
    std::vector<uint32_t> hostIndices;
    uint32_t vertexOffset = 0;
    for (auto& mesh : meshes) {
        for (auto& p : mesh.positions) hostPositions.push_back(p);
        for (auto idx : mesh.indices)
            hostIndices.push_back(idx + vertexOffset);
        vertexOffset += (uint32_t)mesh.positions.size();
    }

    SAH_BVH builder;
    BVHData bvhData = builder.build(hostPositions.data(), hostIndices.data(), data.totalTriangles);

    // The BVH reorders primitives. We need to reorder the device index buffer,
    // material indices, and area light indices to match the BVH's orderedPrimIndices.
    std::vector<uint32_t> reorderedIndices(data.totalTriangles * 3);
    std::vector<int>      reorderedMatIndices(data.totalTriangles);
    std::vector<int>      reorderedAreaLightIndices(data.totalTriangles);

    // Build host material index and area light index arrays (same order as DeviceScene)
    std::vector<int> hostMatIndices;
    std::vector<int> hostAreaLightIndices;
    const auto& materials = scene.getMaterials();
    const auto& areaLights = scene.getAreaLights();
    uint32_t areaLightIdx = 0;
    for (auto& mesh : meshes) {
        const auto& mat = materials[(size_t)mesh.materialIndex];
        bool emissiveMesh = mat.emissionStrength > 0.0f &&
                            (mat.emission.x > 0.0f || mat.emission.y > 0.0f || mat.emission.z > 0.0f);

        uint32_t triCount = (uint32_t)mesh.indices.size() / 3;
        for (uint32_t t = 0; t < triCount; t++) {
            hostMatIndices.push_back(mesh.materialIndex);
            if (emissiveMesh && areaLightIdx < (uint32_t)areaLights.size()) {
                uint32_t li0 = mesh.indices[t * 3 + 0];
                uint32_t li1 = mesh.indices[t * 3 + 1];
                uint32_t li2 = mesh.indices[t * 3 + 2];
                float3 lv0 = mesh.positions[li0];
                float3 lv1 = mesh.positions[li1];
                float3 lv2 = mesh.positions[li2];
                float3 le1 = lv1 - lv0;
                float3 le2 = lv2 - lv0;
                float triArea = 0.5f * length(cross(le1, le2));
                if (triArea > 1e-8f) {
                    hostAreaLightIndices.push_back((int)areaLightIdx++);
                } else {
                    hostAreaLightIndices.push_back(-1);
                }
            } else {
                hostAreaLightIndices.push_back(-1);
            }
        }
    }

    for (uint32_t i = 0; i < (uint32_t)bvhData.orderedPrimIndices.size(); i++) {
        uint32_t origTri = bvhData.orderedPrimIndices[i];
        reorderedIndices[i*3+0] = hostIndices[origTri*3+0];
        reorderedIndices[i*3+1] = hostIndices[origTri*3+1];
        reorderedIndices[i*3+2] = hostIndices[origTri*3+2];
        reorderedMatIndices[i]  = hostMatIndices[origTri];
        reorderedAreaLightIndices[i] = hostAreaLightIndices[origTri];
    }

    // Re-upload reordered indices, material indices, and area light indices
    CUDA_CHECK(cudaMemcpy(data.d_indices, reorderedIndices.data(),
                           data.totalTriangles * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_materialIndices, reorderedMatIndices.data(),
                           data.totalTriangles * sizeof(int), cudaMemcpyHostToDevice));
    if (data.d_triangleAreaLightIndex) {
        CUDA_CHECK(cudaMemcpy(data.d_triangleAreaLightIndex, reorderedAreaLightIndices.data(),
                               data.totalTriangles * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Upload BVH nodes
    BVHNode* d_nodes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_nodes, bvhData.nodes.size() * sizeof(BVHNode)));
    CUDA_CHECK(cudaMemcpy(d_nodes, bvhData.nodes.data(),
                           bvhData.nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));

    // Store in device scene data (need to update internal state)
    // We modify the device scene's BVH pointers via the data struct
    data.d_bvhNodes   = d_nodes;
    data.bvhRootIndex = bvhData.rootIndex;

    // Store the updated data back - we need to keep the BVH node pointer
    m_bvhNodes    = d_nodes;
    m_bvhRoot     = bvhData.rootIndex;

    LOG_INFO("CUDABackend: BVH built and uploaded (%u nodes)", (uint32_t)bvhData.nodes.size());
}

void CUDABackend::launchPathTrace(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    float4* d_accumBuffer,
    float4* d_outputBuffer,
    AuxBufferPtrs auxBuffers,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerPixel,
    PrimaryHitSurfaces gbufferSurfaces,
    bool skipEmissiveInNEE,
    DebugHeatmapPtrs heatmap)
{
    // Patch in BVH data (since DeviceScene may not have it directly)
    DeviceSceneData patchedScene = scene;
    patchedScene.d_bvhNodes   = m_bvhNodes;
    patchedScene.bvhRootIndex = m_bvhRoot;

    launchPathTraceKernel(
        patchedScene, camera,
        d_accumBuffer, d_outputBuffer, auxBuffers,
        width, height, sampleIndex, enableEnvironment, maxBounces,
        samplesPerPixel,
        gbufferSurfaces,
        skipEmissiveInNEE,
        heatmap
    );
}

#ifdef PATHTRACER_NRD_DLSS_ENABLED
void CUDABackend::launchPathTraceSplit(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    SplitSurfaceOutputs surfaces,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerPixel,
    bool skipEmissiveInNEE)
{
    // Patch in CUDA SAH-BVH (the split kernel uses scene.d_bvhNodes directly,
    // mirroring launchPathTrace above). The DeviceScene the renderer passes in
    // does not carry the BVH pointer — the backend owns it.
    DeviceSceneData patchedScene = scene;
    patchedScene.d_bvhNodes   = m_bvhNodes;
    patchedScene.bvhRootIndex = m_bvhRoot;

    launchPathTraceKernelSplit(
        patchedScene, camera, surfaces,
        width, height, sampleIndex, enableEnvironment, maxBounces,
        samplesPerPixel,
        skipEmissiveInNEE);
}
#endif

void CUDABackend::traceOcclusionRays(
    const float3* d_origins,
    const float3* d_targets,
    bool* d_visible,
    uint32_t rayCount)
{
    (void)d_origins; (void)d_targets; (void)d_visible; (void)rayCount;
    // BDPT stub -- will launch occlusion kernel using BVH any-hit
}
