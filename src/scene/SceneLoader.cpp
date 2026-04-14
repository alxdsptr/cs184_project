#include "scene/SceneLoader.h"
#include "scene/PbrtLoader.h"
#include "core/Math.h"
#include "util/Log.h"
#include "utils.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>
#include <unordered_map>

using namespace scene_loader_util;

static void processNode(
    const aiScene* aiScn, const aiNode* node,
    Scene& scene, const std::string& baseDir, int forcedMaterialIndex = -1)
{
    for (unsigned i = 0; i < node->mNumMeshes; i++) {
        const aiMesh* aiM = aiScn->mMeshes[node->mMeshes[i]];

        TriangleMesh mesh;
        mesh.materialIndex = forcedMaterialIndex >= 0 ? forcedMaterialIndex : (int)aiM->mMaterialIndex;

        // Positions
        mesh.positions.resize(aiM->mNumVertices);
        for (unsigned v = 0; v < aiM->mNumVertices; v++) {
            mesh.positions[v] = toFloat3(aiM->mVertices[v]);
            scene.getBounds().expand(mesh.positions[v]);
        }

        // Normals
        if (aiM->HasNormals()) {
            mesh.normals.resize(aiM->mNumVertices);
            for (unsigned v = 0; v < aiM->mNumVertices; v++)
                mesh.normals[v] = toFloat3(aiM->mNormals[v]);
        }

        // UVs (first set)
        if (aiM->HasTextureCoords(0)) {
            mesh.uvs.resize(aiM->mNumVertices);
            for (unsigned v = 0; v < aiM->mNumVertices; v++)
                mesh.uvs[v] = make_float2(aiM->mTextureCoords[0][v].x, aiM->mTextureCoords[0][v].y);
        }

        // Indices
        for (unsigned f = 0; f < aiM->mNumFaces; f++) {
            const aiFace& face = aiM->mFaces[f];
            if (face.mNumIndices == 3) {
                mesh.indices.push_back(face.mIndices[0]);
                mesh.indices.push_back(face.mIndices[1]);
                mesh.indices.push_back(face.mIndices[2]);
            }
        }

        scene.getMeshes().push_back(std::move(mesh));
    }

    for (unsigned i = 0; i < node->mNumChildren; i++)
        processNode(aiScn, node->mChildren[i], scene, baseDir, forcedMaterialIndex);
}

bool SceneLoader::load(const std::string& path, Scene& scene) {
    std::string ext = lowerString(std::filesystem::path(path).extension().string());
    if (ext == ".pbrt") {
        return loadPbrtScene(path, scene);
    }

    Assimp::Importer importer;
    unsigned flags =
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_JoinIdenticalVertices |
        aiProcess_PreTransformVertices |
        aiProcess_FlipUVs;

    const aiScene* aiScn = importer.ReadFile(path, flags);
    if (!aiScn || !aiScn->mRootNode || (aiScn->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        LOG_ERROR("Assimp: %s", importer.GetErrorString());
        return false;
    }

    // Apply unit scaling for FBX files (handle cm vs generic unit conversion)
    applyUnitScaling(const_cast<aiScene*>(aiScn), ext);

    std::string baseDir = std::filesystem::path(path).parent_path().string();

    if (aiScn->mNumCameras > 0) {
        const aiCamera* aiCam = aiScn->mCameras[0];
        if (aiCam) {
            SceneCamera& camera = scene.getCamera();
            camera.valid = true;

            const aiNode* cameraNode = findNodeByName(aiScn->mRootNode, aiCam->mName);
            aiMatrix4x4 cameraWorld = cameraNode ? computeWorldTransform(cameraNode) : aiMatrix4x4();

            aiVector3D position = aiCam->mPosition;
            float3 forward3 = toFloat3(aiCam->mLookAt);
            float3 up3 = toFloat3(aiCam->mUp);
            if (cameraNode) {
                position = cameraWorld * position;
                forward3 = transformDirection(cameraWorld, aiCam->mLookAt);
                up3 = transformDirection(cameraWorld, aiCam->mUp);
            }

            if (length(forward3) > 1e-6f) {
                camera.forward = normalize(forward3);
            }
            if (length(up3) > 1e-6f) {
                camera.up = normalize(up3);
            }

            camera.position = toFloat3(position);
            if (aiCam->mHorizontalFOV > 1e-6f) {
                camera.horizontalFovRadians = aiCam->mHorizontalFOV;
            }
            camera.aspect = aiCam->mAspect;
            if (aiCam->mClipPlaneNear > 1e-6f) {
                camera.nearPlane = aiCam->mClipPlaneNear;
            }
            if (aiCam->mClipPlaneFar > camera.nearPlane) {
                camera.farPlane = aiCam->mClipPlaneFar;
            }

            if (aiCam->mOrthographicWidth > 0.0f) {
                LOG_WARN("Assimp camera %s is orthographic; using its direction but treating it as perspective", aiCam->mName.C_Str());
            }

            LOG_INFO("Loaded camera: %s", aiCam->mName.C_Str());
        }
    }

    // Parse COLLADA radiance extensions (Assimp ignores <extra> radiance data)
    std::unordered_map<std::string, float3> colladaRadiance;
    if (ext == ".dae") {
        colladaRadiance = parseColladaRadiance(path);
    }

    // Materials
    for (unsigned i = 0; i < aiScn->mNumMaterials; i++) {
        const aiMaterial* aiMat = aiScn->mMaterials[i];
        PBRMaterial mat;
        aiString matName;
        aiMat->Get(AI_MATKEY_NAME, matName);
        const std::string materialName = matName.C_Str();

        // Base color
        aiColor3D color(0.8f, 0.8f, 0.8f);
        if (aiMat->Get(AI_MATKEY_BASE_COLOR, color) != aiReturn_SUCCESS)
            aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        mat.albedo = toFloat3(color);

        // Metallic / roughness
        float metallic = 0.0f, roughness = 0.5f;
        bool hasPbrMetallic = (aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == aiReturn_SUCCESS);
        aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);

        // For FBX specular/shininess workflow, Assimp does NOT produce a
        // reliable metallic value. The reflectivity property is also unreliable
        // as it can be 1.0 for all materials (e.g. Bistro).
        // Strategy: only trust metallic from Assimp if it was explicitly set
        // AND is not the suspicious default of exactly 1.0 for a non-PBR format.
        // Metallic-roughness textures (loaded later) will override this at
        // render time in the kernel.
        if (!hasPbrMetallic) {
            metallic = 0.0f;
        }

        mat.metallic  = metallic;
        mat.roughness = roughness;

        // Emission — check multiple sources in priority order:
        // 1. COLLADA <radiance> extension (parsed from XML, Assimp ignores it)
        // 2. PBR emissive intensity (glTF)
        // 3. Assimp emissive color if bright enough to be a real light source
        auto colladaIt = colladaRadiance.find(materialName);
        if (colladaIt != colladaRadiance.end()) {
            mat.emission = colladaIt->second;
            mat.emissionStrength = 1.0f;
            LOG_INFO("Material '%s': using COLLADA radiance (%.1f, %.1f, %.1f)",
                     materialName.c_str(),
                     mat.emission.x, mat.emission.y, mat.emission.z);
        } else {
            aiColor3D emissive(0, 0, 0);
            aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
            mat.emission = toFloat3(emissive);
            float emissiveLum = luminance(toFloat3(emissive));

            float emissiveIntensity = 0.0f;
            bool hasPbrEmissiveIntensity =
                (aiMat->Get(AI_MATKEY_EMISSIVE_INTENSITY, emissiveIntensity) == aiReturn_SUCCESS);

            if (hasPbrEmissiveIntensity && emissiveIntensity > 0.0f && emissiveLum > 0.0f) {
                mat.emissionStrength = emissiveIntensity;
            } else if (emissiveLum > 0.8f) {
                mat.emissionStrength = 1.0f;
            } else {
                mat.emission = make_float3(0.0f, 0.0f, 0.0f);
            }
        }

        // IOR
        float ior = 1.5f;
        aiMat->Get(AI_MATKEY_REFRACTI, ior);
        mat.ior = ior;

        // Texture paths
        mat.albedoTexPath        = getTexturePath(aiMat, aiTextureType_BASE_COLOR, baseDir);
        if (mat.albedoTexPath.empty())
            mat.albedoTexPath    = getTexturePath(aiMat, aiTextureType_DIFFUSE, baseDir);
        mat.normalTexPath        = getTexturePath(aiMat, aiTextureType_NORMALS, baseDir);
        mat.metallicRoughTexPath = getTexturePath(aiMat, aiTextureType_METALNESS, baseDir);
        mat.emissiveTexPath      = getTexturePath(aiMat, aiTextureType_EMISSIVE, baseDir);

        // If the material has an emissive texture, it is explicitly meant to emit
        // light. Enable emission even when the scalar emissive color from Assimp
        // was zero (common in FBX files where the texture carries all the data).
        if (!mat.emissiveTexPath.empty() && mat.emissionStrength <= 0.0f) {
            mat.emission = make_float3(1.0f, 1.0f, 1.0f);
            mat.emissionStrength = 1.0f;
        }

        scene.getMaterials().push_back(std::move(mat));
    }

    // Lights
    for (unsigned i = 0; i < aiScn->mNumLights; i++) {
        const aiLight* aiL = aiScn->mLights[i];
        if (!aiL) continue;

        LOG_INFO("Light[%u] '%s': type=%d color=(%.2f,%.2f,%.2f) atten=(%.4f,%.4f,%.4f)",
                 i, aiL->mName.C_Str(), (int)aiL->mType,
                 aiL->mColorDiffuse.r, aiL->mColorDiffuse.g, aiL->mColorDiffuse.b,
                 aiL->mAttenuationConstant, aiL->mAttenuationLinear, aiL->mAttenuationQuadratic);

        // Currently only point and spot lights are supported (treat spot as point)
        if (aiL->mType != aiLightSource_POINT && aiL->mType != aiLightSource_SPOT) {
            LOG_WARN("Skipping unsupported light type %d for %s",
                     (int)aiL->mType, aiL->mName.C_Str());
            continue;
        }

        PointLight light;

        aiVector3D lightPos = aiL->mPosition;
        const aiNode* lightNode = findNodeByName(aiScn->mRootNode, aiL->mName);
        if (lightNode) {
            aiMatrix4x4 world = computeWorldTransform(lightNode);
            lightPos = world * aiL->mPosition;
        }

        light.position = toFloat3(lightPos);
        light.color = toFloat3(aiL->mColorDiffuse);
        if (light.color.x <= 0.0f && light.color.y <= 0.0f && light.color.z <= 0.0f) {
            light.color = make_float3(1.0f, 1.0f, 1.0f);
        }
        light.intensity = 1.0f;
        light.constantAttenuation = aiL->mAttenuationConstant;
        light.linearAttenuation = aiL->mAttenuationLinear;
        light.quadraticAttenuation = aiL->mAttenuationQuadratic;

        scene.getLights().push_back(light);
    }

    // Meshes
    processNode(aiScn, aiScn->mRootNode, scene, baseDir);

    // Emissive triangle area lights
    const auto& meshes = scene.getMeshes();
    const auto& materials = scene.getMaterials();
    for (const auto& mesh : meshes) {
        if (mesh.materialIndex < 0 || (size_t)mesh.materialIndex >= materials.size()) {
            continue;
        }

        const PBRMaterial& mat = materials[(size_t)mesh.materialIndex];
        if (mat.emissionStrength <= 0.0f) {
            continue;
        }

        // Skip materials that rely on emissive textures for area light creation.
        // We can't know per-triangle emission from a texture at scene-load time,
        // so these are handled via BSDF path hits in the kernel instead.
        if (!mat.emissiveTexPath.empty()) {
            continue;
        }

        float3 emission = mat.emission * mat.emissionStrength;
        float emissionLum = luminance(emission);
        if (emissionLum <= 0.0f) {
            continue;
        }

        uint32_t triCount = (uint32_t)mesh.indices.size() / 3;
        for (uint32_t t = 0; t < triCount; t++) {
            uint32_t i0 = mesh.indices[t * 3 + 0];
            uint32_t i1 = mesh.indices[t * 3 + 1];
            uint32_t i2 = mesh.indices[t * 3 + 2];

            float3 v0 = mesh.positions[i0];
            float3 v1 = mesh.positions[i1];
            float3 v2 = mesh.positions[i2];
            float3 e1 = v1 - v0;
            float3 e2 = v2 - v0;
            float3 n = cross(e1, e2);
            float area = 0.5f * length(n);
            if (area <= 1e-8f) {
                continue;
            }

            TriangleAreaLight light;
            light.v0 = v0;
            light.e1 = e1;
            light.e2 = e2;
            light.normal = normalize(n);
            light.emission = emission;
            light.area = area;
            light.weight = area * emissionLum;
            scene.getAreaLights().push_back(light);
        }
    }

    LOG_INFO("Loaded: %s (%u meshes, %u materials, %u lights, %u triangles, %u vertices)",
             path.c_str(),
             (unsigned)scene.getMeshes().size(),
             (unsigned)scene.getMaterials().size(),
             (unsigned)scene.getLights().size(),
             scene.totalTriangles(),
             scene.totalVertices());

    LOG_INFO("Emissive triangle lights: %u", (unsigned)scene.getAreaLights().size());

    return true;
}
