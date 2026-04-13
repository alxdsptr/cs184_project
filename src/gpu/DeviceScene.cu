#include "gpu/DeviceScene.h"
#include "scene/Scene.h"
#include "util/CudaCheck.h"
#include "util/Log.h"

void DeviceScene::upload(const Scene& scene) {
    free(); // release any prior data

    const auto& meshes = scene.getMeshes();
    const auto& materials = scene.getMaterials();

    // Count totals
    uint32_t totalVerts = 0, totalTris = 0;
    for (auto& m : meshes) {
        totalVerts += (uint32_t)m.positions.size();
        totalTris  += (uint32_t)m.indices.size() / 3;
    }

    m_data.totalVertices  = totalVerts;
    m_data.totalTriangles = totalTris;
    m_data.materialCount  = (uint32_t)materials.size();

    // Flatten all meshes into contiguous arrays
    std::vector<float3>   allPositions;
    std::vector<float3>   allNormals;
    std::vector<float2>   allUVs;
    std::vector<uint32_t> allIndices;
    std::vector<int>      allMatIndices; // per-triangle

    allPositions.reserve(totalVerts);
    allNormals.reserve(totalVerts);
    allUVs.reserve(totalVerts);
    allIndices.reserve(totalTris * 3);
    allMatIndices.reserve(totalTris);

    uint32_t vertexOffset = 0;
    for (auto& mesh : meshes) {
        for (auto& p : mesh.positions) allPositions.push_back(p);

        if (!mesh.normals.empty()) {
            for (auto& n : mesh.normals) allNormals.push_back(n);
        } else {
            for (size_t j = 0; j < mesh.positions.size(); j++)
                allNormals.push_back(make_float3(0, 1, 0));
        }

        if (!mesh.uvs.empty()) {
            for (auto& uv : mesh.uvs) allUVs.push_back(uv);
        } else {
            for (size_t j = 0; j < mesh.positions.size(); j++)
                allUVs.push_back(make_float2(0, 0));
        }

        uint32_t triCount = (uint32_t)mesh.indices.size() / 3;
        for (auto idx : mesh.indices)
            allIndices.push_back(idx + vertexOffset);
        for (uint32_t t = 0; t < triCount; t++)
            allMatIndices.push_back(mesh.materialIndex);

        vertexOffset += (uint32_t)mesh.positions.size();
    }

    // Upload positions
    CUDA_CHECK(cudaMalloc(&m_data.d_positions, totalVerts * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(m_data.d_positions, allPositions.data(),
                           totalVerts * sizeof(float3), cudaMemcpyHostToDevice));

    // Upload normals
    CUDA_CHECK(cudaMalloc(&m_data.d_normals, totalVerts * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(m_data.d_normals, allNormals.data(),
                           totalVerts * sizeof(float3), cudaMemcpyHostToDevice));

    // Upload UVs
    CUDA_CHECK(cudaMalloc(&m_data.d_uvs, totalVerts * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(m_data.d_uvs, allUVs.data(),
                           totalVerts * sizeof(float2), cudaMemcpyHostToDevice));

    // Upload indices
    CUDA_CHECK(cudaMalloc(&m_data.d_indices, totalTris * 3 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(m_data.d_indices, allIndices.data(),
                           totalTris * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Upload per-triangle material indices
    CUDA_CHECK(cudaMalloc(&m_data.d_materialIndices, totalTris * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(m_data.d_materialIndices, allMatIndices.data(),
                           totalTris * sizeof(int), cudaMemcpyHostToDevice));

    // Upload materials (convert PBRMaterial -> GPUMaterial)
    std::vector<GPUMaterial> gpuMats(materials.size());
    for (size_t i = 0; i < materials.size(); i++) {
        auto& src = materials[i];
        auto& dst = gpuMats[i];
        dst.albedo           = src.albedo;
        dst.roughness        = src.roughness;
        dst.metallic         = src.metallic;
        dst.emission         = src.emission;
        dst.emissionStrength = src.emissionStrength;
        dst.ior              = src.ior;
        dst.transmission     = src.transmission;
        // Texture objects are set separately by the caller after TextureManager loads them
        dst.albedoTex        = 0;
        dst.normalTex        = 0;
        dst.metallicRoughTex = 0;
        dst.emissiveTex      = 0;
    }

    CUDA_CHECK(cudaMalloc(&m_data.d_materials, materials.size() * sizeof(GPUMaterial)));
    CUDA_CHECK(cudaMemcpy(m_data.d_materials, gpuMats.data(),
                           materials.size() * sizeof(GPUMaterial), cudaMemcpyHostToDevice));

    LOG_INFO("GPU upload: %u vertices, %u triangles, %u materials",
             totalVerts, totalTris, (uint32_t)materials.size());
}

void DeviceScene::free() {
    if (m_data.d_positions)       { cudaFree(m_data.d_positions); }
    if (m_data.d_normals)         { cudaFree(m_data.d_normals); }
    if (m_data.d_uvs)             { cudaFree(m_data.d_uvs); }
    if (m_data.d_indices)         { cudaFree(m_data.d_indices); }
    if (m_data.d_materials)       { cudaFree(m_data.d_materials); }
    if (m_data.d_materialIndices) { cudaFree(m_data.d_materialIndices); }
    if (m_data.d_bvhNodes)        { cudaFree(m_data.d_bvhNodes); }
    m_data = DeviceSceneData{};
}
