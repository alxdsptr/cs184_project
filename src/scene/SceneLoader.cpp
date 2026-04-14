#include "scene/SceneLoader.h"
#include "core/Math.h"
#include "util/Log.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/matrix4x4.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <cmath>

static float3 toFloat3(const aiVector3D& v) { return make_float3(v.x, v.y, v.z); }
static float3 toFloat3(const aiColor3D& c)  { return make_float3(c.r, c.g, c.b); }

static float3 transformDirection(const aiMatrix4x4& m, const aiVector3D& v) {
    return make_float3(
        m.a1 * v.x + m.a2 * v.y + m.a3 * v.z,
        m.b1 * v.x + m.b2 * v.y + m.b3 * v.z,
        m.c1 * v.x + m.c2 * v.y + m.c3 * v.z
    );
}

static float luminance(const float3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// Parse COLLADA (.dae) files for <radiance> emission data in <extra> blocks.
// Assimp ignores these CGL extensions, so we extract them manually.
// Returns a map from material name -> radiance float3.
static std::unordered_map<std::string, float3> parseColladaRadiance(const std::string& path) {
    std::unordered_map<std::string, float3> result;

    std::ifstream in(path);
    if (!in.is_open()) return result;

    std::string content((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    in.close();

    // Step 1: Find effects that have <radiance> in their <extra> block.
    // Map: effect-id -> radiance value
    std::unordered_map<std::string, float3> effectRadiance;
    {
        // Find each <effect id="..."> and check for <radiance>
        size_t pos = 0;
        while (true) {
            size_t effectStart = content.find("<effect ", pos);
            if (effectStart == std::string::npos) break;

            // Find the id attribute
            size_t idPos = content.find("id=\"", effectStart);
            size_t effectEnd = content.find("</effect>", effectStart);
            if (idPos == std::string::npos || effectEnd == std::string::npos) {
                pos = effectStart + 1;
                continue;
            }
            if (idPos > effectEnd) { pos = effectEnd; continue; }

            size_t idStart = idPos + 4;
            size_t idEnd = content.find('"', idStart);
            if (idEnd == std::string::npos) break;
            std::string effectId = content.substr(idStart, idEnd - idStart);

            // Look for <radiance> within this effect
            size_t radPos = content.find("<radiance>", effectStart);
            if (radPos != std::string::npos && radPos < effectEnd) {
                size_t radStart = radPos + 10; // strlen("<radiance>")
                size_t radEnd = content.find("</radiance>", radStart);
                if (radEnd != std::string::npos && radEnd < effectEnd) {
                    std::stringstream ss(content.substr(radStart, radEnd - radStart));
                    float r = 0, g = 0, b = 0;
                    if (ss >> r >> g >> b) {
                        effectRadiance[effectId] = make_float3(r, g, b);
                        LOG_INFO("COLLADA: effect '%s' has radiance (%.1f, %.1f, %.1f)",
                                 effectId.c_str(), r, g, b);
                    }
                }
            }

            pos = effectEnd;
        }
    }

    if (effectRadiance.empty()) return result;

    // Step 2: Find <material> elements that reference these effects via
    //         <instance_effect url="#effect-id"/>, mapping material name -> radiance
    {
        size_t pos = 0;
        while (true) {
            size_t matStart = content.find("<material ", pos);
            if (matStart == std::string::npos) break;

            size_t matEnd = content.find("</material>", matStart);
            if (matEnd == std::string::npos) matEnd = content.find("/>", matStart);
            if (matEnd == std::string::npos) break;

            // Extract name attribute
            size_t namePos = content.find("name=\"", matStart);
            if (namePos != std::string::npos && namePos < matEnd) {
                size_t nameStart = namePos + 6;
                size_t nameEnd = content.find('"', nameStart);
                if (nameEnd != std::string::npos) {
                    std::string matName = content.substr(nameStart, nameEnd - nameStart);

                    // Find instance_effect url
                    size_t instPos = content.find("<instance_effect", matStart);
                    if (instPos != std::string::npos && instPos < matEnd) {
                        size_t urlPos = content.find("url=\"#", instPos);
                        if (urlPos != std::string::npos) {
                            size_t urlStart = urlPos + 6;
                            size_t urlEnd = content.find('"', urlStart);
                            if (urlEnd != std::string::npos) {
                                std::string effectRef = content.substr(urlStart, urlEnd - urlStart);
                                auto it = effectRadiance.find(effectRef);
                                if (it != effectRadiance.end()) {
                                    result[matName] = it->second;
                                    LOG_INFO("COLLADA: material '%s' -> radiance (%.1f, %.1f, %.1f)",
                                             matName.c_str(), it->second.x, it->second.y, it->second.z);
                                }
                            }
                        }
                    }
                }
            }

            pos = matEnd;
        }
    }

    return result;
}

static std::string lowerString(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return value;
}

static std::string trimString(std::string value) {
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && isSpace((unsigned char)value.front())) value.erase(value.begin());
    while (!value.empty() && isSpace((unsigned char)value.back())) value.pop_back();
    return value;
}

static std::vector<std::string> extractQuotedStrings(const std::string& line) {
    std::vector<std::string> values;
    size_t pos = 0;
    while (true) {
        size_t start = line.find('"', pos);
        if (start == std::string::npos) break;
        size_t end = line.find('"', start + 1);
        if (end == std::string::npos) break;
        values.push_back(line.substr(start + 1, end - start - 1));
        pos = end + 1;
    }
    return values;
}

static std::vector<float> extractBracketFloats(const std::string& line) {
    std::vector<float> values;
    size_t left = line.find('[');
    size_t right = line.find(']', left == std::string::npos ? 0 : left + 1);
    if (left == std::string::npos || right == std::string::npos || right <= left) {
        return values;
    }

    std::stringstream ss(line.substr(left + 1, right - left - 1));
    float v = 0.0f;
    while (ss >> v) {
        values.push_back(v);
    }
    return values;
}

static std::filesystem::path resolvePbrtMeshPath(const std::filesystem::path& sceneDir, const std::string& relativeMeshPath) {
    std::filesystem::path rel(relativeMeshPath);
    if (rel.is_absolute() && std::filesystem::exists(rel)) {
        return rel;
    }

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(sceneDir / rel);
    if (!sceneDir.empty() && sceneDir.has_parent_path()) {
        candidates.push_back(sceneDir.parent_path() / rel);
    }
    if (!sceneDir.empty() && sceneDir.has_parent_path() && sceneDir.parent_path().has_parent_path()) {
        candidates.push_back(sceneDir.parent_path().parent_path() / rel);
    }

    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return {};
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

static void applyUnitScaling(aiScene* aiScn, const std::string& ext) {
    if (!aiScn) return;

    // FBX files often store units in metadata and may use cm as default
    // OBJ files don't have standard unit metadata, so we normalize FBX to match
    double unitScale = 1.0;
    bool applied = false;
    
    if (ext == ".fbx") {
        // Check FBX metadata for unit information
        if (aiScn->mMetaData) {
            double metaScale = 1.0;
            if (aiScn->mMetaData->Get("UnitScaleFactor", metaScale)) {
                if (metaScale > 1e-6) {
                    unitScale = 1.0 / metaScale;
                    applied = true;
                    LOG_INFO("FBX UnitScaleFactor metadata: %.6f, applying scale: %.6f", metaScale, unitScale);
                }
            }
        }

        // If metadata didn't provide scale, auto-detect based on coordinate magnitudes
        if (!applied) {
            float maxCoord = 0.0f;
            float minCoord = 1e10f;
            float avgCoord = 0.0f;
            unsigned totalVerts = 0;

            for (unsigned m = 0; m < aiScn->mNumMeshes; m++) {
                const aiMesh* mesh = aiScn->mMeshes[m];
                for (unsigned v = 0; v < mesh->mNumVertices; v++) {
                    const aiVector3D& pos = mesh->mVertices[v];
                    float mag = std::sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z);
                    maxCoord = std::max(maxCoord, mag);
                    minCoord = std::min(minCoord, mag);
                    avgCoord += mag;
                    totalVerts++;
                }
            }

            if (totalVerts > 0) {
                avgCoord /= totalVerts;
                LOG_INFO("FBX coordinate analysis: max=%.2f, min=%.2f, avg=%.2f", 
                         maxCoord, minCoord < 1e9f ? minCoord : 0.0f, avgCoord);
                
                // If coordinates are very large (typical for cm-based FBX), apply 0.01 scale
                if (maxCoord > 100.0f) {
                    unitScale = 0.01;
                    applied = true;
                    LOG_INFO("Detected large FBX coordinates (max: %.1f). Applying 0.01 scale factor.", maxCoord);
                }
            }
        }
    }

    // Apply scale to all mesh vertices if scaling factor is significant
    if (applied && std::abs(unitScale - 1.0) > 1e-6) {
        LOG_INFO("Applying unit scale factor: %.6f to %s file", unitScale, ext.c_str());
        
        for (unsigned m = 0; m < aiScn->mNumMeshes; m++) {
            aiMesh* mesh = aiScn->mMeshes[m];
            for (unsigned v = 0; v < mesh->mNumVertices; v++) {
                mesh->mVertices[v] *= (float)unitScale;
            }
        }

        // Also scale camera positions and lights
        for (unsigned c = 0; c < aiScn->mNumCameras; c++) {
            aiCamera* cam = aiScn->mCameras[c];
            if (cam) {
                cam->mPosition *= (float)unitScale;
                cam->mLookAt *= (float)unitScale;
            }
        }

        for (unsigned l = 0; l < aiScn->mNumLights; l++) {
            aiLight* light = aiScn->mLights[l];
            if (light) {
                light->mPosition *= (float)unitScale;
                light->mDirection *= (float)unitScale;
            }
        }
    } else if (!applied) {
        LOG_INFO("No unit scaling applied to %s file (units appear standard)", ext.c_str());
    }
}

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

static std::string getTexturePath(const aiMaterial* mat, aiTextureType type, const std::string& baseDir) {
    if (mat->GetTextureCount(type) > 0) {
        aiString str;
        mat->GetTexture(type, 0, &str);
        std::string texturePath = str.C_Str();
        
        // Check if texture format is supported
        std::string lowerPath = texturePath;
        std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(),
                      [](unsigned char c) { return std::tolower(c); });
        
        // Skip unsupported formats (EXR requires special library)
        if (lowerPath.find(".exr") != std::string::npos) {
            LOG_WARN("Texture format .exr not supported. Skipping: %s", texturePath.c_str());
            return "";
        }
        
        // Try different candidate paths
        std::vector<std::filesystem::path> candidates;
        
        // 1. Direct path as-is
        candidates.push_back(texturePath);
        
        // 2. Relative to baseDir (FBX directory)
        candidates.push_back(std::filesystem::path(baseDir) / texturePath);
        
        // 3. Try to extract just the filename and search in baseDir
        std::filesystem::path texFile = texturePath;
        std::string filename = texFile.filename().string();
        candidates.push_back(std::filesystem::path(baseDir) / filename);
        
        // 4. Try to find it in parent directories (for multi-level model + texture dirs)
        if (!baseDir.empty()) {
            auto basePathObj = std::filesystem::path(baseDir);
            
            // Search up to 2 levels up with the filename alone
            for (int i = 0; i < 2 && basePathObj.has_parent_path(); i++) {
                basePathObj = basePathObj.parent_path();
                candidates.push_back(basePathObj / filename);
                
                // Also try the original relative path from parent dirs
                candidates.push_back(basePathObj / texturePath);
            }
        }
        
        // Find first existing path
        for (const auto& candidate : candidates) {
            if (std::filesystem::exists(candidate)) {
                std::string result = candidate.string();
                LOG_INFO("Resolved texture: %s -> %s", texturePath.c_str(), result.c_str());
                return result;
            }
        }
        
        // If nothing found, log warning and return the original attempt
        LOG_WARN("Failed to locate texture: %s (tried %zu paths)", texturePath.c_str(), candidates.size());
        return (std::filesystem::path(baseDir) / texturePath).string();
    }
    return "";
}

static bool loadPbrt(const std::string& path, Scene& scene) {
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

        processNode(aiScn, aiScn->mRootNode, scene, meshPath.parent_path().string(), materialIndex);
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

bool SceneLoader::load(const std::string& path, Scene& scene) {
    std::string ext = lowerString(std::filesystem::path(path).extension().string());
    if (ext == ".pbrt") {
        return loadPbrt(path, scene);
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
