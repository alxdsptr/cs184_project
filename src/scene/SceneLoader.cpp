#include "scene/SceneLoader.h"
#include "core/Math.h"
#include "util/Log.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/matrix4x4.h>
#include <filesystem>

static float3 toFloat3(const aiVector3D& v) { return make_float3(v.x, v.y, v.z); }
static float3 toFloat3(const aiColor3D& c)  { return make_float3(c.r, c.g, c.b); }

static bool containsLightName(const std::string& name) {
    return name.find("light") != std::string::npos || name.find("Light") != std::string::npos;
}

static float luminance(const float3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

static const aiNode* findNodeByName(const aiNode* node, const aiString& name) {
    if (!node) return nullptr;
    if (node->mName == name) return node;
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        const aiNode* found = findNodeByName(node->mChildren[i], name);
        if (found) return found;
    }
    return nullptr;
}

static aiMatrix4x4 computeWorldTransform(const aiNode* node) {
    aiMatrix4x4 world;
    if (!node) return world;

    world = node->mTransformation;
    const aiNode* parent = node->mParent;
    while (parent) {
        world = parent->mTransformation * world;
        parent = parent->mParent;
    }
    return world;
}

static void processNode(
    const aiScene* aiScn, const aiNode* node,
    Scene& scene, const std::string& baseDir)
{
    for (unsigned i = 0; i < node->mNumMeshes; i++) {
        const aiMesh* aiM = aiScn->mMeshes[node->mMeshes[i]];

        TriangleMesh mesh;
        mesh.materialIndex = (int)aiM->mMaterialIndex;

        // Positions
        mesh.positions.resize(aiM->mNumVertices);
        for (unsigned v = 0; v < aiM->mNumVertices; v++)
            mesh.positions[v] = toFloat3(aiM->mVertices[v]);

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
        processNode(aiScn, node->mChildren[i], scene, baseDir);
}

static std::string getTexturePath(const aiMaterial* mat, aiTextureType type, const std::string& baseDir) {
    if (mat->GetTextureCount(type) > 0) {
        aiString str;
        mat->GetTexture(type, 0, &str);
        std::filesystem::path p = std::filesystem::path(baseDir) / str.C_Str();
        return p.string();
    }
    return "";
}

bool SceneLoader::load(const std::string& path, Scene& scene) {
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

    std::string baseDir = std::filesystem::path(path).parent_path().string();

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
        aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
        aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
        mat.metallic  = metallic;
        mat.roughness = roughness;

        // Emission
        aiColor3D emissive(0, 0, 0);
        aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissive);
        mat.emission = toFloat3(emissive);
        if (emissive.r > 0 || emissive.g > 0 || emissive.b > 0)
            mat.emissionStrength = 1.0f;

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

        if (mat.emissionStrength <= 0.0f && containsLightName(materialName)) {
            mat.emission = make_float3(10.0f, 10.0f, 10.0f);
            mat.emissionStrength = 1.0f;
        }

        scene.getMaterials().push_back(std::move(mat));
    }

    // Lights (currently import point lights)
    for (unsigned i = 0; i < aiScn->mNumLights; i++) {
        const aiLight* aiL = aiScn->mLights[i];
        if (!aiL) continue;

        if (aiL->mType != aiLightSource_POINT) {
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
