#include "gpu/DeviceScene.h"
#include "scene/Scene.h"
#include "util/CudaCheck.h"
#include "util/Log.h"

void DeviceScene::upload(const Scene& scene) {
    free(); // release any prior data

    const auto& meshes = scene.getMeshes();
    const auto& materials = scene.getMaterials();
    const auto& lights = scene.getLights();
    const auto& areaLights = scene.getAreaLights();

    // Count totals
    uint32_t totalVerts = 0, totalTris = 0;
    for (auto& m : meshes) {
        totalVerts += (uint32_t)m.positions.size();
        totalTris  += (uint32_t)m.indices.size() / 3;
    }

    m_data.totalVertices  = totalVerts;
    m_data.totalTriangles = totalTris;
    m_data.materialCount  = (uint32_t)materials.size();
    m_data.pointLightCount = (uint32_t)lights.size();
    m_data.areaLightCount = (uint32_t)areaLights.size();

    // Flatten all meshes into contiguous arrays
    std::vector<float3>   allPositions;
    std::vector<float3>   allNormals;
    std::vector<float2>   allUVs;
    std::vector<uint32_t> allIndices;
    std::vector<int>      allMatIndices; // per-triangle
    std::vector<int>      allAreaLightIndices; // per-triangle

    allPositions.reserve(totalVerts);
    allNormals.reserve(totalVerts);
    allUVs.reserve(totalVerts);
    allIndices.reserve(totalTris * 3);
    allMatIndices.reserve(totalTris);
    allAreaLightIndices.reserve(totalTris);

    uint32_t vertexOffset = 0;
    uint32_t areaLightIndex = 0;
    for (auto& mesh : meshes) {
        const auto& mat = materials[(size_t)mesh.materialIndex];
        bool emissiveMesh = mat.emissionStrength > 0.0f &&
                            (mat.emission.x > 0.0f || mat.emission.y > 0.0f || mat.emission.z > 0.0f);

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
        for (uint32_t t = 0; t < triCount; t++) {
            allMatIndices.push_back(mesh.materialIndex);
            if (emissiveMesh && areaLightIndex < areaLights.size()) {
                // Check if this triangle matches the next area light (non-degenerate).
                // SceneLoader skips degenerate triangles (area <= 1e-8), so we must
                // replicate that check to keep indices in sync.
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
                    allAreaLightIndices.push_back((int)areaLightIndex++);
                } else {
                    allAreaLightIndices.push_back(-1);
                }
            } else {
                allAreaLightIndices.push_back(-1);
            }
        }

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

    CUDA_CHECK(cudaMalloc(&m_data.d_triangleAreaLightIndex, totalTris * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(m_data.d_triangleAreaLightIndex, allAreaLightIndices.data(),
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
        dst.pureDiffuse      = src.pureDiffuse ? 1 : 0;
        dst.albedoTex        = src.albedoTexObj;
        dst.normalTex        = src.normalTexObj;
        dst.metallicRoughTex = src.metallicRoughTexObj;
        dst.emissiveTex      = src.emissiveTexObj;
    }

    CUDA_CHECK(cudaMalloc(&m_data.d_materials, materials.size() * sizeof(GPUMaterial)));
    CUDA_CHECK(cudaMemcpy(m_data.d_materials, gpuMats.data(),
                           materials.size() * sizeof(GPUMaterial), cudaMemcpyHostToDevice));

    if (!lights.empty()) {
        std::vector<GPUPointLight> gpuLights(lights.size());
        for (size_t i = 0; i < lights.size(); i++) {
            gpuLights[i].position = lights[i].position;
            gpuLights[i].color = lights[i].color;
            gpuLights[i].intensity = lights[i].intensity;
            gpuLights[i].constantAttenuation = lights[i].constantAttenuation;
            gpuLights[i].linearAttenuation = lights[i].linearAttenuation;
            gpuLights[i].quadraticAttenuation = lights[i].quadraticAttenuation;
        }

        CUDA_CHECK(cudaMalloc(&m_data.d_pointLights, lights.size() * sizeof(GPUPointLight)));
        CUDA_CHECK(cudaMemcpy(m_data.d_pointLights, gpuLights.data(),
                               lights.size() * sizeof(GPUPointLight), cudaMemcpyHostToDevice));
    }

    if (!areaLights.empty()) {
        std::vector<GPUAreaLight> gpuAreaLights(areaLights.size());
        std::vector<float> cdf(areaLights.size());
        float totalWeight = 0.0f;
        for (size_t i = 0; i < areaLights.size(); i++) {
            const auto& src = areaLights[i];
            auto& dst = gpuAreaLights[i];
            dst.v0 = src.v0;
            dst.e1 = src.e1;
            dst.e2 = src.e2;
            dst.normal = src.normal;
            dst.emission = src.emission;
            dst.area = src.area;
            dst.weight = src.weight;
            totalWeight += src.weight;
            cdf[i] = totalWeight;
        }

        if (totalWeight > 0.0f) {
            for (auto& value : cdf) {
                value /= totalWeight;
            }
            m_data.areaLightTotalWeight = totalWeight;

            CUDA_CHECK(cudaMalloc(&m_data.d_areaLights, areaLights.size() * sizeof(GPUAreaLight)));
            CUDA_CHECK(cudaMemcpy(m_data.d_areaLights, gpuAreaLights.data(),
                                   areaLights.size() * sizeof(GPUAreaLight), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMalloc(&m_data.d_areaLightCDF, areaLights.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(m_data.d_areaLightCDF, cdf.data(),
                                   areaLights.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    LOG_INFO("GPU upload: %u vertices, %u triangles, %u materials, %u lights, %u area lights",
             totalVerts, totalTris, (uint32_t)materials.size(), (uint32_t)lights.size(), (uint32_t)areaLights.size());
}

void DeviceScene::free() {
    if (m_data.d_positions)       { cudaFree(m_data.d_positions); }
    if (m_data.d_normals)         { cudaFree(m_data.d_normals); }
    if (m_data.d_uvs)             { cudaFree(m_data.d_uvs); }
    if (m_data.d_indices)         { cudaFree(m_data.d_indices); }
    if (m_data.d_materials)       { cudaFree(m_data.d_materials); }
    if (m_data.d_materialIndices) { cudaFree(m_data.d_materialIndices); }
    if (m_data.d_pointLights)     { cudaFree(m_data.d_pointLights); }
    if (m_data.d_areaLights)      { cudaFree(m_data.d_areaLights); }
    if (m_data.d_areaLightCDF)    { cudaFree(m_data.d_areaLightCDF); }
    if (m_data.d_triangleAreaLightIndex) { cudaFree(m_data.d_triangleAreaLightIndex); }
    if (m_data.d_bvhNodes)        { cudaFree(m_data.d_bvhNodes); }
    m_data = DeviceSceneData{};
}
