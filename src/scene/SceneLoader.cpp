#include "scene/SceneLoader.h"
#include "util/Log.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>

static float3 toFloat3(const aiVector3D& v) { return make_float3(v.x, v.y, v.z); }
static float3 toFloat3(const aiColor3D& c)  { return make_float3(c.r, c.g, c.b); }

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

        scene.getMaterials().push_back(std::move(mat));
    }

    // Meshes
    processNode(aiScn, aiScn->mRootNode, scene, baseDir);

    LOG_INFO("Loaded: %s (%u meshes, %u materials, %u triangles, %u vertices)",
             path.c_str(),
             (unsigned)scene.getMeshes().size(),
             (unsigned)scene.getMaterials().size(),
             scene.totalTriangles(),
             scene.totalVertices());

    return true;
}
