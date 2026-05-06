#include "gpu/DeviceScene.h"
#include "scene/Scene.h"
#include "core/Math.h"
#include "util/CudaCheck.h"
#include "util/Log.h"
#include "accel/LightBVH.h"

// Apply a host float4x4 to a float3 (treated as a point with implicit w=1).
static float3 transformPos(const float4x4& M, float3 p) {
    return mat4_transformPoint(M, p);
}

// Apply the cofactor (inverse-transpose) of M's upper-3x3 to a normal, then
// normalize. Used to bake worldRest into rest-pose normals.
static float3 transformNormal(const float4x4& M, float3 n) {
    float a00 = M.m[0][0], a01 = M.m[0][1], a02 = M.m[0][2];
    float a10 = M.m[1][0], a11 = M.m[1][1], a12 = M.m[1][2];
    float a20 = M.m[2][0], a21 = M.m[2][1], a22 = M.m[2][2];
    float c00 =  (a11 * a22 - a12 * a21);
    float c01 = -(a10 * a22 - a12 * a20);
    float c02 =  (a10 * a21 - a11 * a20);
    float c10 = -(a01 * a22 - a02 * a21);
    float c11 =  (a00 * a22 - a02 * a20);
    float c12 = -(a00 * a21 - a01 * a20);
    float c20 =  (a01 * a12 - a02 * a11);
    float c21 = -(a00 * a12 - a02 * a10);
    float c22 =  (a00 * a11 - a01 * a10);
    float x = c00*n.x + c01*n.y + c02*n.z;
    float y = c10*n.x + c11*n.y + c12*n.z;
    float z = c20*n.x + c21*n.y + c22*n.z;
    float L = sqrtf(x*x + y*y + z*z);
    if (L > 1e-12f) { float inv = 1.0f/L; x *= inv; y *= inv; z *= inv; }
    else { x = n.x; y = n.y; z = n.z; }
    return make_float3(x, y, z);
}

// Apply cofactor of upper-3x3 to a tangent (then renormalize the .xyz part;
// .w handedness is preserved).
static float4 transformTangent(const float4x4& M, float4 t) {
    // Use plain rotation (upper-3x3) for tangents; for non-uniform scale this
    // is approximate, but FBX rigid-body keyframes here are uniform-scale.
    float x = M.m[0][0]*t.x + M.m[0][1]*t.y + M.m[0][2]*t.z;
    float y = M.m[1][0]*t.x + M.m[1][1]*t.y + M.m[1][2]*t.z;
    float z = M.m[2][0]*t.x + M.m[2][1]*t.y + M.m[2][2]*t.z;
    float L = sqrtf(x*x + y*y + z*z);
    if (L > 1e-12f) { float inv = 1.0f/L; x *= inv; y *= inv; z *= inv; }
    return make_float4(x, y, z, t.w);
}

void DeviceScene::upload(const Scene& scene) {
    free(); // release any prior data

    const auto& meshes            = scene.getMeshes();
    const auto& materials         = scene.getMaterials();
    const auto& lights            = scene.getLights();
    const auto& directionalLights = scene.getDirectionalLights();
    const auto& areaLights        = scene.getAreaLights();
    const auto& bindings          = scene.getMeshBindings();
    const auto& nodes             = scene.getNodes();

    m_data.medium = scene.getMedium();

    uint32_t totalVerts = 0, totalTris = 0;
    for (auto& m : meshes) {
        totalVerts += (uint32_t)m.positions.size();
        totalTris  += (uint32_t)m.indices.size() / 3;
    }
    m_data.totalVertices  = totalVerts;
    m_data.totalTriangles = totalTris;
    m_data.materialCount  = (uint32_t)materials.size();
    m_data.pointLightCount = (uint32_t)lights.size();
    m_data.directionalLightCount = (uint32_t)directionalLights.size();
    m_data.areaLightCount = (uint32_t)areaLights.size();

    // Flatten — same merged-buffer layout the rest of the renderer expects.
    // Geometry is baked to *world space* using each mesh's node->worldRest;
    // for static meshes that's the final pose, for animated meshes it's the
    // pose at t=0 (the rest pose). The pose-update kernel will overwrite the
    // animated-vertex slots with the per-frame world-space pose.
    std::vector<float3>   posWorldRest;          // baked for static; rest pose for animated
    std::vector<float3>   normalsWorldRest;
    std::vector<float4>   allTangents;
    std::vector<float2>   allUVs;
    std::vector<uint32_t> allIndices;
    std::vector<int>      allMatIndices;
    std::vector<int>      allAreaLightIndices;
    std::vector<uint32_t> perVertexMeshIndex;    // staticSentinel or mesh index

    posWorldRest.reserve(totalVerts);
    normalsWorldRest.reserve(totalVerts);
    allTangents.reserve(totalVerts);
    allUVs.reserve(totalVerts);
    allIndices.reserve(totalTris * 3);
    allMatIndices.reserve(totalTris);
    allAreaLightIndices.reserve(totalTris);
    perVertexMeshIndex.reserve(totalVerts);

    bool sceneHasAnimation = scene.hasAnimation();
    uint32_t animatedMeshCount = 0;
    uint32_t animatedVertexCount = 0;

    uint32_t vertexOffset = 0;
    uint32_t areaLightIndex = 0;
    const uint32_t kStaticSentinel = 0xFFFFFFFFu;

    for (size_t mi = 0; mi < meshes.size(); mi++) {
        const auto& mesh = meshes[mi];
        const auto& mat  = materials[(size_t)mesh.materialIndex];
        bool emissiveMesh = mat.emissionStrength > 0.0f &&
                            (mat.emission.x > 0.0f || mat.emission.y > 0.0f || mat.emission.z > 0.0f);

        int ni = (mi < bindings.size()) ? bindings[mi].nodeIndex : -1;
        const float4x4& W = (ni >= 0 && (size_t)ni < nodes.size())
                                ? nodes[(size_t)ni].worldRest
                                : float4x4::identity();
        bool meshAnimated = (mi < bindings.size()) ? bindings[mi].animated : false;

        // Positions (mesh-local -> world space at rest pose).
        for (auto& p : mesh.positions) posWorldRest.push_back(transformPos(W, p));

        // Normals — same transform, cofactor + renormalize.
        if (!mesh.normals.empty()) {
            for (auto& n : mesh.normals) normalsWorldRest.push_back(transformNormal(W, n));
        } else {
            for (size_t j = 0; j < mesh.positions.size(); j++)
                normalsWorldRest.push_back(make_float3(0, 1, 0));
        }

        // Tangents — rotate by upper-3x3.
        if (!mesh.tangents.empty()) {
            for (auto& t : mesh.tangents) allTangents.push_back(transformTangent(W, t));
        } else {
            for (size_t j = 0; j < mesh.positions.size(); j++)
                allTangents.push_back(make_float4(1, 0, 0, 0));
        }

        if (!mesh.uvs.empty()) {
            for (auto& uv : mesh.uvs) allUVs.push_back(uv);
        } else {
            for (size_t j = 0; j < mesh.positions.size(); j++)
                allUVs.push_back(make_float2(0, 0));
        }

        // Per-vertex mesh index for the pose-update kernel.
        uint32_t pvIdx = meshAnimated ? (uint32_t)mi : kStaticSentinel;
        for (size_t j = 0; j < mesh.positions.size(); j++) {
            perVertexMeshIndex.push_back(pvIdx);
        }
        if (meshAnimated) {
            animatedMeshCount++;
            animatedVertexCount += (uint32_t)mesh.positions.size();
        }

        uint32_t triCount = (uint32_t)mesh.indices.size() / 3;
        for (auto idx : mesh.indices) allIndices.push_back(idx + vertexOffset);
        for (uint32_t t = 0; t < triCount; t++) {
            allMatIndices.push_back(mesh.materialIndex);
            if (emissiveMesh && areaLightIndex < areaLights.size()) {
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

    // ── Pose update setup (animation only) ────────────────────────────
    // When the scene has any animated meshes, allocate PoseUpdate buffers
    // and have m_data.d_positions / d_normals / d_positionsPrev *alias* into
    // them (so the existing flat-buffer reads in the OptiX/CUDA kernels see
    // the pose-update output without any code change). The rest-pose host
    // arrays we just built become the kernel's d_positionsRest / d_normalsRest.
    bool useAnimation = sceneHasAnimation && animatedMeshCount > 0;
    if (useAnimation) {
        poseUpdateAlloc(m_pose, totalVerts, (uint32_t)meshes.size());
        m_pose.animatedVertexCount = animatedVertexCount;

        CUDA_CHECK(cudaMemcpy(m_pose.d_positionsRest, posWorldRest.data(),
                              totalVerts * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_pose.d_normalsRest, normalsWorldRest.data(),
                              totalVerts * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_pose.d_perVertexMeshIndex, perVertexMeshIndex.data(),
                              totalVerts * sizeof(uint32_t), cudaMemcpyHostToDevice));
        // Seed the curr/prev buffers to the rest pose so the very first
        // launch sees something sensible (kernel's firstFrame=true will
        // overwrite the animated slots; static slots keep these values).
        CUDA_CHECK(cudaMemcpy(m_pose.d_positionsCurr, posWorldRest.data(),
                              totalVerts * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_pose.d_normalsCurr, normalsWorldRest.data(),
                              totalVerts * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_pose.d_positionsPrev, posWorldRest.data(),
                              totalVerts * sizeof(float3), cudaMemcpyHostToDevice));

        // Alias rendering pointers into pose buffers (DeviceScene does NOT
        // free these — m_pose owns them).
        m_data.d_positions     = m_pose.d_positionsCurr;
        m_data.d_normals       = m_pose.d_normalsCurr;
        m_data.d_positionsPrev = m_pose.d_positionsPrev;
        m_ownsPositions = false;
        m_ownsNormals   = false;
    } else {
        // No animation — original layout. Allocate and own d_positions /
        // d_normals directly.
        CUDA_CHECK(cudaMalloc(&m_data.d_positions, totalVerts * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(m_data.d_positions, posWorldRest.data(),
                              totalVerts * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&m_data.d_normals, totalVerts * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy(m_data.d_normals, normalsWorldRest.data(),
                              totalVerts * sizeof(float3), cudaMemcpyHostToDevice));
        m_data.d_positionsPrev = nullptr;  // motion vectors fall back to static reprojection
        m_ownsPositions = true;
        m_ownsNormals   = true;
    }

    CUDA_CHECK(cudaMalloc(&m_data.d_tangents, totalVerts * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(m_data.d_tangents, allTangents.data(),
                          totalVerts * sizeof(float4), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&m_data.d_uvs, totalVerts * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(m_data.d_uvs, allUVs.data(),
                          totalVerts * sizeof(float2), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&m_data.d_indices, totalTris * 3 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(m_data.d_indices, allIndices.data(),
                          totalTris * 3 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&m_data.d_materialIndices, totalTris * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(m_data.d_materialIndices, allMatIndices.data(),
                          totalTris * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&m_data.d_triangleAreaLightIndex, totalTris * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(m_data.d_triangleAreaLightIndex, allAreaLightIndices.data(),
                          totalTris * sizeof(int), cudaMemcpyHostToDevice));

    // Materials.
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
        dst.useSpecularGlossiness = src.useSpecularGlossiness ? 1 : 0;
        dst.specularGlossAlphaIsGlossiness = src.specularGlossAlphaIsGlossiness ? 1 : 0;
        dst.useFBXCustomPacking = src.useFBXCustomPacking ? 1 : 0;
        dst.useFBXUEPacking  = src.useFBXUEPacking ? 1 : 0;
        dst.specularColor    = src.specularColor;
        dst.glossiness       = src.glossiness;
        dst.albedoTex        = src.albedoTexObj;
        dst.normalTex        = src.normalTexObj;
        dst.metallicRoughTex = src.metallicRoughTexObj;
        dst.emissiveTex      = src.emissiveTexObj;
        dst.specularGlossTex = src.specularGlossTexObj;
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

    if (!directionalLights.empty()) {
        std::vector<GPUDirectionalLight> gpuDirectionalLights(directionalLights.size());
        for (size_t i = 0; i < directionalLights.size(); i++) {
            gpuDirectionalLights[i].direction = directionalLights[i].direction;
            gpuDirectionalLights[i].color = directionalLights[i].color;
        }

        CUDA_CHECK(cudaMalloc(&m_data.d_directionalLights,
                              directionalLights.size() * sizeof(GPUDirectionalLight)));
        CUDA_CHECK(cudaMemcpy(m_data.d_directionalLights, gpuDirectionalLights.data(),
                              directionalLights.size() * sizeof(GPUDirectionalLight),
                              cudaMemcpyHostToDevice));
    }

    // Area lights — full set goes through both BSDF-hit emissive lookup AND
    // the NEE / light-BVH path. Animated lights get their world triangle and
    // BVH leaf AABBs refreshed each frame by the light-update + BVH-refit
    // kernels in PoseUpdate / LightBVHRefit. Their _rest fields hold the
    // upload-time pose so the per-frame update is `meshDelta * rest`.
    if (!areaLights.empty()) {
        std::vector<GPUAreaLight> gpuAreaLights(areaLights.size());
        // CDF as a fallback when the light BVH descent fails. Built over the
        // full set including animated emitters — for animated lights area
        // (and therefore weight) is approximately preserved under rigid-body
        // animation, so a static CDF is a reasonable approximation.
        std::vector<float> cdf(areaLights.size(), 0.0f);
        float totalWeight = 0.0f;
        for (size_t i = 0; i < areaLights.size(); i++) {
            const auto& src = areaLights[i];
            auto& dst = gpuAreaLights[i];
            // Initialise both current and rest poses to the upload value.
            // The light-update kernel will overwrite v0/e1/e2/normal each
            // frame for animated lights (meshIndex >= 0); v0_rest/e1_rest/
            // e2_rest stay frozen for use as the kernel's source operand.
            dst.v0 = src.v0;
            dst.e1 = src.e1;
            dst.e2 = src.e2;
            dst.normal = src.normal;
            dst.v0_rest = src.v0;
            dst.e1_rest = src.e1;
            dst.e2_rest = src.e2;
            dst.normal_rest = src.normal;
            dst.meshIndex = src.meshIndex;
            dst.emission = src.emission;
            dst.area = src.area;
            dst.weight = src.weight;
            dst.uv0 = src.uv0;
            dst.uv1 = src.uv1;
            dst.uv2 = src.uv2;
            dst.emissiveTex = src.emissiveTexObj;
            totalWeight += dst.weight;
            cdf[i] = totalWeight;
        }

        if (totalWeight > 0.0f) {
            for (auto& value : cdf) value /= totalWeight;
            m_data.areaLightTotalWeight = totalWeight;

            CUDA_CHECK(cudaMalloc(&m_data.d_areaLights, areaLights.size() * sizeof(GPUAreaLight)));
            CUDA_CHECK(cudaMemcpy(m_data.d_areaLights, gpuAreaLights.data(),
                                  areaLights.size() * sizeof(GPUAreaLight), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMalloc(&m_data.d_areaLightCDF, areaLights.size() * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(m_data.d_areaLightCDF, cdf.data(),
                                  areaLights.size() * sizeof(float), cudaMemcpyHostToDevice));

            // Light BVH — built over the full set with real weights. Animated
            // lights' bounds will drift each frame; LightBVHRefit::refit()
            // walks the leaves up to the root rebuilding AABBs to track them.
            // Topology stays fixed (build-time partitioning is preserved) so
            // the stochastic descent + lightIndexToSlot map remain valid.
            std::vector<AABB>  lightBounds(areaLights.size());
            std::vector<float> lightWeights(areaLights.size());
            for (size_t i = 0; i < areaLights.size(); i++) {
                const auto& src = areaLights[i];
                float3 v0 = src.v0;
                float3 v1 = src.v0 + src.e1;
                float3 v2 = src.v0 + src.e2;
                AABB b;
                b.expand(v0); b.expand(v1); b.expand(v2);
                lightBounds[i]  = b;
                lightWeights[i] = src.weight;
            }

            LightBVH builder;
            LightBVHData lbvh = builder.build(lightBounds.data(),
                                              lightWeights.data(),
                                              (uint32_t)areaLights.size());

            std::vector<uint32_t> lightIndexToSlot(areaLights.size(), 0);
            for (uint32_t slot = 0; slot < lbvh.orderedLightIndices.size(); slot++) {
                lightIndexToSlot[lbvh.orderedLightIndices[slot]] = slot;
            }

            CUDA_CHECK(cudaMalloc(&m_data.d_lightBVHNodes,
                                  lbvh.nodes.size() * sizeof(LightBVHNode)));
            CUDA_CHECK(cudaMemcpy(m_data.d_lightBVHNodes, lbvh.nodes.data(),
                                  lbvh.nodes.size() * sizeof(LightBVHNode),
                                  cudaMemcpyHostToDevice));
            m_data.lightBVHRootIndex = lbvh.rootIndex;

            CUDA_CHECK(cudaMalloc(&m_data.d_lightOrderedIndices,
                                  lbvh.orderedLightIndices.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(m_data.d_lightOrderedIndices,
                                  lbvh.orderedLightIndices.data(),
                                  lbvh.orderedLightIndices.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMalloc(&m_data.d_lightIndexToSlot,
                                  lightIndexToSlot.size() * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(m_data.d_lightIndexToSlot,
                                  lightIndexToSlot.data(),
                                  lightIndexToSlot.size() * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));

            // ── Light-BVH refit hookup (animation only) ─────────────────
            // Wire the per-level node-index arrays into PoseUpdateData so
            // the per-frame lightBVHRefitLaunch() can wave-front merge the
            // BVH from leaves up to root. Nothing to do for static scenes —
            // their bounds were finalised at build time and never go stale.
            if (useAnimation && !lbvh.nodesByLevel.empty()) {
                m_pose.d_areaLights        = m_data.d_areaLights;
                m_pose.areaLightCount      = (uint32_t)areaLights.size();
                m_pose.d_lightBVHNodes     = m_data.d_lightBVHNodes;
                m_pose.d_orderedLightIndices = m_data.d_lightOrderedIndices;
                m_pose.lightBVHRootIndex   = lbvh.rootIndex;
                m_pose.lightBVHNodeCount   = (uint32_t)lbvh.nodes.size();
                m_pose.d_lightBVHLevel.resize(lbvh.nodesByLevel.size(), nullptr);
                m_pose.lightBVHLevelSize.resize(lbvh.nodesByLevel.size(), 0);
                for (size_t lv = 0; lv < lbvh.nodesByLevel.size(); lv++) {
                    const auto& level = lbvh.nodesByLevel[lv];
                    m_pose.lightBVHLevelSize[lv] = (uint32_t)level.size();
                    if (level.empty()) continue;
                    CUDA_CHECK(cudaMalloc(&m_pose.d_lightBVHLevel[lv],
                                          level.size() * sizeof(uint32_t)));
                    CUDA_CHECK(cudaMemcpy(m_pose.d_lightBVHLevel[lv],
                                          level.data(),
                                          level.size() * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice));
                }
                LOG_INFO("Light BVH refit wired: %u nodes across %zu levels",
                         (unsigned)lbvh.nodes.size(), lbvh.nodesByLevel.size());
            }
        }
    }

    LOG_INFO("GPU upload: %u vertices, %u triangles, %u materials, %u lights, %u area lights, animation=%s (animated meshes=%u verts=%u)",
             totalVerts, totalTris,
             (uint32_t)materials.size(), (uint32_t)lights.size(), (uint32_t)areaLights.size(),
             useAnimation ? "yes" : "no",
             animatedMeshCount, animatedVertexCount);
}

void DeviceScene::refreshAnimationPointers() {
    if (m_pose.vertexCount == 0) return;
    // d_positions/d_normals/d_positionsPrev all alias into m_pose buffers,
    // which keep the same address across pose updates — nothing to do.
    m_data.d_positions     = m_pose.d_positionsCurr;
    m_data.d_normals       = m_pose.d_normalsCurr;
    m_data.d_positionsPrev = m_pose.d_positionsPrev;
}

void DeviceScene::free() {
    if (m_ownsPositions && m_data.d_positions) { cudaFree(m_data.d_positions); }
    if (m_ownsNormals   && m_data.d_normals)   { cudaFree(m_data.d_normals); }
    m_data.d_positions = nullptr;
    m_data.d_normals = nullptr;
    m_data.d_positionsPrev = nullptr;
    m_ownsPositions = true;
    m_ownsNormals = true;

    if (m_data.d_tangents)        { cudaFree(m_data.d_tangents); }
    if (m_data.d_uvs)             { cudaFree(m_data.d_uvs); }
    if (m_data.d_indices)         { cudaFree(m_data.d_indices); }
    if (m_data.d_materials)       { cudaFree(m_data.d_materials); }
    if (m_data.d_materialIndices) { cudaFree(m_data.d_materialIndices); }
    if (m_data.d_pointLights)     { cudaFree(m_data.d_pointLights); }
    if (m_data.d_directionalLights) { cudaFree(m_data.d_directionalLights); }
    if (m_data.d_areaLights)      { cudaFree(m_data.d_areaLights); }
    if (m_data.d_areaLightCDF)    { cudaFree(m_data.d_areaLightCDF); }
    if (m_data.d_triangleAreaLightIndex) { cudaFree(m_data.d_triangleAreaLightIndex); }
    if (m_data.d_bvhNodes)        { cudaFree(m_data.d_bvhNodes); }
    if (m_data.d_lightBVHNodes)   { cudaFree(m_data.d_lightBVHNodes); }
    if (m_data.d_lightOrderedIndices){ cudaFree(m_data.d_lightOrderedIndices); }
    if (m_data.d_lightIndexToSlot)   { cudaFree(m_data.d_lightIndexToSlot); }

    poseUpdateFree(m_pose);

    m_data = DeviceSceneData{};
}
