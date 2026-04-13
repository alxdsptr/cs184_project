#include "backend/CUDABackend.h"
#include "scene/Scene.h"
#include "accel/SAH_BVH.h"
#include "render/PathTraceKernel.h"
#include "render/Tonemapping.h"
#include "util/CudaCheck.h"
#include "util/Log.h"

void CUDABackend::buildAccelerationStructure(const Scene& scene) {
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

    // The BVH reorders primitives. We need to reorder the device index buffer
    // and material indices to match the BVH's orderedPrimIndices.
    std::vector<uint32_t> reorderedIndices(data.totalTriangles * 3);
    std::vector<int>      reorderedMatIndices(data.totalTriangles);

    // Build host material index array (same order as DeviceScene)
    std::vector<int> hostMatIndices;
    for (auto& mesh : meshes) {
        uint32_t triCount = (uint32_t)mesh.indices.size() / 3;
        for (uint32_t t = 0; t < triCount; t++)
            hostMatIndices.push_back(mesh.materialIndex);
    }

    for (uint32_t i = 0; i < (uint32_t)bvhData.orderedPrimIndices.size(); i++) {
        uint32_t origTri = bvhData.orderedPrimIndices[i];
        reorderedIndices[i*3+0] = hostIndices[origTri*3+0];
        reorderedIndices[i*3+1] = hostIndices[origTri*3+1];
        reorderedIndices[i*3+2] = hostIndices[origTri*3+2];
        reorderedMatIndices[i]  = hostMatIndices[origTri];
    }

    // Re-upload reordered indices and material indices
    CUDA_CHECK(cudaMemcpy(data.d_indices, reorderedIndices.data(),
                           data.totalTriangles * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_materialIndices, reorderedMatIndices.data(),
                           data.totalTriangles * sizeof(int), cudaMemcpyHostToDevice));

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
    uint32_t sampleIndex)
{
    // Patch in BVH data (since DeviceScene may not have it directly)
    DeviceSceneData patchedScene = scene;
    patchedScene.d_bvhNodes   = m_bvhNodes;
    patchedScene.bvhRootIndex = m_bvhRoot;

    launchPathTraceKernel(
        patchedScene, camera,
        d_accumBuffer, d_outputBuffer, auxBuffers,
        width, height, sampleIndex
    );
}

void CUDABackend::traceOcclusionRays(
    const float3* d_origins,
    const float3* d_targets,
    bool* d_visible,
    uint32_t rayCount)
{
    (void)d_origins; (void)d_targets; (void)d_visible; (void)rayCount;
    // BDPT stub -- will launch occlusion kernel using BVH any-hit
}
