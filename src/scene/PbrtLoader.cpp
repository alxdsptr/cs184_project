#include "scene/PbrtLoader.h"

#include "core/Math.h"
#include "util/Log.h"
#include "utils.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <filesystem>
#include <fstream>
#include <unordered_map>

using namespace scene_loader_util;

namespace {
void processNodeForPbrt(
    const aiScene* aiScn,
    const aiNode* node,
    Scene& scene,
    const std::string& baseDir,
    int forcedMaterialIndex = -1)
{
    for (unsigned i = 0; i < node->mNumMeshes; i++) {
        const aiMesh* aiM = aiScn->mMeshes[node->mMeshes[i]];

        TriangleMesh mesh;
        mesh.materialIndex = forcedMaterialIndex >= 0 ? forcedMaterialIndex : (int)aiM->mMaterialIndex;

        mesh.positions.resize(aiM->mNumVertices);
        for (unsigned v = 0; v < aiM->mNumVertices; v++) {
            mesh.positions[v] = toFloat3(aiM->mVertices[v]);
            scene.getBounds().expand(mesh.positions[v]);
        }

        if (aiM->HasNormals()) {
            mesh.normals.resize(aiM->mNumVertices);
            for (unsigned v = 0; v < aiM->mNumVertices; v++) {
                mesh.normals[v] = toFloat3(aiM->mNormals[v]);
            }
        }

        if (aiM->HasTextureCoords(0)) {
            mesh.uvs.resize(aiM->mNumVertices);
            for (unsigned v = 0; v < aiM->mNumVertices; v++) {
                mesh.uvs[v] = make_float2(aiM->mTextureCoords[0][v].x, aiM->mTextureCoords[0][v].y);
            }
        }

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

    for (unsigned i = 0; i < node->mNumChildren; i++) {
        processNodeForPbrt(aiScn, node->mChildren[i], scene, baseDir, forcedMaterialIndex);
    }
}
}

bool loadPbrtScene(const std::string& path, Scene& scene) {
    std::ifstream in(path);
    if (!in.is_open()) {
        LOG_ERROR("Failed to open PBRT scene: %s", path.c_str());
        return false;
    }

    struct PendingShape {
        std::string material;
        std::string relativePath;
    };

    std::filesystem::path scenePath(path);
    std::filesystem::path sceneDir = scenePath.parent_path();

    std::unordered_map<std::string, PBRMaterial> namedMaterials;
    std::vector<std::string> materialOrder;
    std::vector<PendingShape> pendingShapes;

    std::string activeMaterial;
    std::string definingMaterial;
    bool waitingForPlyFilename = false;
    std::string pendingShapeMaterial;
    bool waitingForCameraFov = false;
    bool hasCameraTransform = false;
    float cameraTransform[16] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float cameraFovDegrees = 0.0f;

    std::string line;
    while (std::getline(in, line)) {
        std::string s = trimString(line);
        if (s.empty() || s[0] == '#') {
            continue;
        }

        if (s.rfind("MakeNamedMaterial", 0) == 0) {
            auto quoted = extractQuotedStrings(s);
            if (!quoted.empty()) {
                definingMaterial = quoted[0];
                if (namedMaterials.find(definingMaterial) == namedMaterials.end()) {
                    namedMaterials.emplace(definingMaterial, PBRMaterial{});
                    materialOrder.push_back(definingMaterial);
                }
            }
            continue;
        }

        if (!definingMaterial.empty() && s.find("\"rgb reflectance\"") != std::string::npos) {
            auto values = extractBracketFloats(s);
            if (values.size() >= 3) {
                auto it = namedMaterials.find(definingMaterial);
                if (it != namedMaterials.end()) {
                    it->second.albedo = make_float3(values[0], values[1], values[2]);
                }
            }
            continue;
        }

        if (s.rfind("NamedMaterial", 0) == 0) {
            auto quoted = extractQuotedStrings(s);
            if (!quoted.empty()) {
                activeMaterial = quoted[0];
                if (namedMaterials.find(activeMaterial) == namedMaterials.end()) {
                    namedMaterials.emplace(activeMaterial, PBRMaterial{});
                    materialOrder.push_back(activeMaterial);
                }
            }
            continue;
        }

        if (s.rfind("Shape \"plymesh\"", 0) == 0) {
            waitingForPlyFilename = true;
            pendingShapeMaterial = activeMaterial;
            continue;
        }

        if (waitingForPlyFilename && s.find("\"string filename\"") != std::string::npos) {
            auto quoted = extractQuotedStrings(s);
            if (quoted.size() >= 2) {
                pendingShapes.push_back({pendingShapeMaterial, quoted[1]});
            }
            waitingForPlyFilename = false;
            continue;
        }

        if (s.rfind("Camera", 0) == 0 && s.find("\"perspective\"") != std::string::npos) {
            waitingForCameraFov = true;
            continue;
        }

        if (waitingForCameraFov && s.find("\"float fov\"") != std::string::npos) {
            auto values = extractBracketFloats(s);
            if (!values.empty()) {
                cameraFovDegrees = values[0];
            }
            waitingForCameraFov = false;
            continue;
        }

        if (s.rfind("Transform", 0) == 0) {
            auto values = extractBracketFloats(s);
            if (values.size() >= 16) {
                for (size_t i = 0; i < 16; i++) {
                    cameraTransform[i] = values[i];
                }
                hasCameraTransform = true;
            }
            continue;
        }
    }

    std::unordered_map<std::string, int> materialIndices;
    for (const std::string& name : materialOrder) {
        materialIndices[name] = (int)scene.getMaterials().size();
        scene.getMaterials().push_back(namedMaterials[name]);
    }
    if (scene.getMaterials().empty()) {
        scene.getMaterials().push_back(PBRMaterial{});
    }

    unsigned loadedMeshFiles = 0;
    for (const auto& shape : pendingShapes) {
        std::filesystem::path meshPath = resolvePbrtMeshPath(sceneDir, shape.relativePath);
        if (meshPath.empty()) {
            LOG_WARN("PBRT mesh not found: %s", shape.relativePath.c_str());
            continue;
        }

        int materialIndex = 0;
        auto it = materialIndices.find(shape.material);
        if (it != materialIndices.end()) {
            materialIndex = it->second;
        }

        Assimp::Importer importer;
        const unsigned flags =
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_JoinIdenticalVertices |
            aiProcess_FlipUVs;

        const aiScene* aiScn = importer.ReadFile(meshPath.string(), flags);
        if (!aiScn || !aiScn->mRootNode || (aiScn->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
            LOG_WARN("Failed to import PBRT mesh via Assimp: %s (%s)",
                     meshPath.string().c_str(), importer.GetErrorString());
            continue;
        }

        processNodeForPbrt(aiScn, aiScn->mRootNode, scene, meshPath.parent_path().string(), materialIndex);
        loadedMeshFiles++;
    }

    if (loadedMeshFiles == 0) {
        LOG_ERROR("PBRT scene loaded no meshes: %s", path.c_str());
        return false;
    }

    if (hasCameraTransform || cameraFovDegrees > 1e-6f) {
        SceneCamera& camera = scene.getCamera();
        camera.valid = true;
        if (hasCameraTransform) {
            camera.position = make_float3(cameraTransform[12], cameraTransform[13], cameraTransform[14]);

            float3 forward = make_float3(-cameraTransform[8], -cameraTransform[9], -cameraTransform[10]);
            float3 up = make_float3(cameraTransform[4], cameraTransform[5], cameraTransform[6]);
            if (length(forward) > 1e-6f) {
                camera.forward = normalize(forward);
            }
            if (length(up) > 1e-6f) {
                camera.up = normalize(up);
            }
        }

        if (cameraFovDegrees > 1e-6f) {
            constexpr float kPi = 3.14159265358979323846f;
            camera.horizontalFovRadians = cameraFovDegrees * (kPi / 180.0f);
        }
    }

    LOG_INFO("Loaded PBRT: %s (%u meshes, %u materials, %u triangles, %u vertices)",
             path.c_str(),
             (unsigned)scene.getMeshes().size(),
             (unsigned)scene.getMaterials().size(),
             scene.totalTriangles(),
             scene.totalVertices());

    return true;
}
