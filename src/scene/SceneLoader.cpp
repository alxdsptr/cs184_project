#include "scene/SceneLoader.h"
#include "scene/SceneTreeBuilder.h"
#include "scene/Animation.h"
#include "scene/PbrtLoader.h"
#include "core/Math.h"
#include "util/Log.h"
#include "utils.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

using namespace scene_loader_util;

// Process meshes attached to one aiNode (no recursion — the SceneNode tree is
// already built; we walk the aiNode tree separately to attach meshes to the
// right SceneNode). Geometry is left in *mesh-local* space; DeviceScene + the
// pose-update kernel handle world-space transformation.
//
// scenenodeIndex maps aiNode* -> SceneNode index (already built by
// SceneTreeBuilder). Pivot-chain intermediates map to the canonical node so
// any meshes mistakenly attached to one (shouldn't happen for FBX, defensive
// only) still land in a sensible place.
static void attachNodeMeshes(
    const aiScene* aiScn, const aiNode* aiN,
    Scene& scene,
    const std::unordered_map<const aiNode*, int>& sceneNodeIndex,
    int forcedMaterialIndex = -1)
{
    auto it = sceneNodeIndex.find(aiN);
    int myNodeIdx = (it != sceneNodeIndex.end()) ? it->second : -1;

    for (unsigned i = 0; i < aiN->mNumMeshes; i++) {
        const aiMesh* aiM = aiScn->mMeshes[aiN->mMeshes[i]];

        TriangleMesh mesh;
        mesh.materialIndex = forcedMaterialIndex >= 0 ? forcedMaterialIndex : (int)aiM->mMaterialIndex;

        // Positions — kept in mesh-local space (no PreTransformVertices).
        mesh.positions.resize(aiM->mNumVertices);
        for (unsigned v = 0; v < aiM->mNumVertices; v++) {
            mesh.positions[v] = toFloat3(aiM->mVertices[v]);
        }

        // Normals — also mesh-local.
        if (aiM->HasNormals()) {
            mesh.normals.resize(aiM->mNumVertices);
            for (unsigned v = 0; v < aiM->mNumVertices; v++)
                mesh.normals[v] = toFloat3(aiM->mNormals[v]);
        }

        // UVs (first set).
        if (aiM->HasTextureCoords(0)) {
            mesh.uvs.resize(aiM->mNumVertices);
            for (unsigned v = 0; v < aiM->mNumVertices; v++)
                mesh.uvs[v] = make_float2(aiM->mTextureCoords[0][v].x, aiM->mTextureCoords[0][v].y);
        }

        // Tangents (computed by aiProcess_CalcTangentSpace). Pack handedness
        // into .w so the kernel can reconstruct B = sign * cross(N, T). These
        // are mesh-local too — they rotate with the mesh transform.
        if (aiM->HasTangentsAndBitangents()) {
            mesh.tangents.resize(aiM->mNumVertices);
            for (unsigned v = 0; v < aiM->mNumVertices; v++) {
                float3 t = toFloat3(aiM->mTangents[v]);
                float3 b = toFloat3(aiM->mBitangents[v]);
                float3 n = aiM->HasNormals() ? toFloat3(aiM->mNormals[v]) : make_float3(0,1,0);
                float sign = (dot(cross(n, t), b) < 0.0f) ? -1.0f : 1.0f;
                mesh.tangents[v] = make_float4(t.x, t.y, t.z, sign);
            }
        }

        // Indices.
        for (unsigned f = 0; f < aiM->mNumFaces; f++) {
            const aiFace& face = aiM->mFaces[f];
            if (face.mNumIndices == 3) {
                mesh.indices.push_back(face.mIndices[0]);
                mesh.indices.push_back(face.mIndices[1]);
                mesh.indices.push_back(face.mIndices[2]);
            }
        }

        // Track binding before we move-out the mesh.
        MeshNodeBinding binding;
        binding.nodeIndex = myNodeIdx;
        binding.vertexCount = (uint32_t)mesh.positions.size();
        // vertexOffset is filled in by DeviceScene at upload time (it depends
        // on the flatten order); leave 0 for now.
        binding.vertexOffset = 0;
        // animated flag is filled in after animation parsing.
        binding.animated = false;

        scene.getMeshes().push_back(std::move(mesh));
        scene.getMeshBindings().push_back(binding);
    }

    // Recurse into children — we need to walk the FULL aiNode tree, including
    // pivot-chain intermediates, because meshes can in principle hang off any
    // aiNode (though for FBX they only attach to logical leaves).
    for (unsigned i = 0; i < aiN->mNumChildren; i++) {
        attachNodeMeshes(aiScn, aiN->mChildren[i], scene, sceneNodeIndex, forcedMaterialIndex);
    }
}

// ── Animation extraction ─────────────────────────────────────
//
// Convert aiAnimation channels into AnimationClip channels. Channels in
// pivot-chain intermediates (`Foo_$AssimpFbx$_Translation` etc.) target the
// same canonical node — we merge them so the final AnimationClip has at most
// one channel per logical node, carrying the full TRS data for the chain.
//
// `nameToNodeIdx` maps both the canonical name and the pivot intermediate
// names to the canonical node index (built by SceneTreeBuilder).
static AnimationClip extractAnimation(
    const aiAnimation* aiA,
    const std::unordered_map<std::string, int>& nameToNodeIdx)
{
    AnimationClip clip;
    clip.name = aiA->mName.C_Str();
    float tps = (aiA->mTicksPerSecond > 1e-6) ? (float)aiA->mTicksPerSecond : 30.0f;
    clip.ticksPerSecond = tps;
    clip.duration = (float)aiA->mDuration / tps;

    // Map canonical node idx -> channel slot in `clip.channels`.
    std::unordered_map<int, int> nodeToSlot;

    auto getOrCreateSlot = [&](int nodeIdx) -> int {
        auto it = nodeToSlot.find(nodeIdx);
        if (it != nodeToSlot.end()) return it->second;
        int slot = (int)clip.channels.size();
        clip.channels.emplace_back();
        clip.nodeIndices.push_back(nodeIdx);
        nodeToSlot[nodeIdx] = slot;
        return slot;
    };

    for (unsigned c = 0; c < aiA->mNumChannels; c++) {
        const aiNodeAnim* ch = aiA->mChannels[c];
        std::string chName = ch->mNodeName.C_Str();
        auto it = nameToNodeIdx.find(chName);
        if (it == nameToNodeIdx.end()) {
            // Channel targets a node we don't know about — skip.
            continue;
        }
        int nodeIdx = it->second;
        int slot = getOrCreateSlot(nodeIdx);
        AnimChannelTrack& tr = clip.channels[slot];

        // Append keys, converting ticks to seconds.
        for (unsigned k = 0; k < ch->mNumPositionKeys; k++) {
            tr.posTimes.push_back((float)ch->mPositionKeys[k].mTime / tps);
            const aiVector3D& v = ch->mPositionKeys[k].mValue;
            tr.posValues.push_back(make_float3(v.x, v.y, v.z));
        }
        for (unsigned k = 0; k < ch->mNumRotationKeys; k++) {
            tr.rotTimes.push_back((float)ch->mRotationKeys[k].mTime / tps);
            const aiQuaternion& q = ch->mRotationKeys[k].mValue;
            tr.rotValues.push_back(make_float4(q.x, q.y, q.z, q.w));
        }
        for (unsigned k = 0; k < ch->mNumScalingKeys; k++) {
            tr.scaleTimes.push_back((float)ch->mScalingKeys[k].mTime / tps);
            const aiVector3D& v = ch->mScalingKeys[k].mValue;
            tr.scaleValues.push_back(make_float3(v.x, v.y, v.z));
        }
    }

    // Note: when multiple intermediate channels (Translation, Rotation,
    // Scaling) target the same canonical node, we end up with disjoint key
    // arrays — pos comes from the _Translation channel, rot from _Rotation,
    // scale from _Scaling. The animation evaluator samples each
    // independently and composes them, which is exactly what we want.
    return clip;
}

// Recursively mark all SceneNodes in the subtree of `nodeIdx` as `animated`.
// The mark is inherited because a child's world transform depends on its
// animated parent's pose, even if the child itself has no track.
static void markSubtreeAnimated(int nodeIdx, std::vector<SceneNode>& nodes) {
    if (nodeIdx < 0 || (size_t)nodeIdx >= nodes.size()) return;
    nodes[nodeIdx].animated = true;
    for (size_t i = nodeIdx + 1; i < nodes.size(); i++) {
        if (nodes[i].parent >= 0 && (size_t)nodes[i].parent < i && nodes[nodes[i].parent].animated) {
            nodes[i].animated = true;
        }
    }
}

bool SceneLoader::load(const std::string& path, Scene& scene, SGWorkflowMode sgMode,
                       float texturedEmissiveTargetLum) {
    std::string ext = lowerString(std::filesystem::path(path).extension().string());
    if (ext == ".pbrt") {
        return loadPbrtScene(path, scene);
    }

    Assimp::Importer importer;
    // Note: NO aiProcess_PreTransformVertices — we need the node hierarchy
    // intact so we can resolve animation channels. The pose-update kernel
    // applies per-mesh transforms at runtime.
    unsigned flags =
        aiProcess_Triangulate |
        aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_JoinIdenticalVertices |
        aiProcess_FlipUVs;

    const aiScene* aiScn = importer.ReadFile(path, flags);
    if (!aiScn || !aiScn->mRootNode || (aiScn->mFlags & AI_SCENE_FLAGS_INCOMPLETE)) {
        LOG_ERROR("Assimp: %s", importer.GetErrorString());
        return false;
    }

    // Apply unit scaling for FBX files (handles cm vs m + the
    // large-coordinate heuristic). With PreTransformVertices dropped from the
    // import flags, applyUnitScaling now also scales node translation columns
    // and animation translation keys so the hierarchy-applied result matches
    // the previously-flat geometry.
    applyUnitScaling(const_cast<aiScene*>(aiScn), ext);

    std::string baseDir = std::filesystem::path(path).parent_path().string();

    // ── Build scene-graph tree (parent-before-child, pivots collapsed) ──
    std::unordered_map<const aiNode*, int> nodeIndex;
    std::unordered_map<std::string, int>   logicalNameToIdx;
    std::unordered_map<std::string, int>   anyNameToIdx;
    buildSceneTree(aiScn, scene, nodeIndex, logicalNameToIdx, anyNameToIdx);

    // Camera (transform via the cached worldRest of the camera's node, which
    // already includes the pivot-chain composition).
    if (aiScn->mNumCameras > 0) {
        const aiCamera* aiCam = aiScn->mCameras[0];
        if (aiCam) {
            SceneCamera& camera = scene.getCamera();
            camera.valid = true;

            const aiNode* cameraNode = findNodeByName(aiScn->mRootNode, aiCam->mName);
            int camIdx = -1;
            if (cameraNode) {
                auto it = nodeIndex.find(cameraNode);
                if (it != nodeIndex.end()) camIdx = it->second;
            }
            float4x4 camWorld = camIdx >= 0 ? scene.getNodes()[camIdx].worldRest
                                            : float4x4::identity();

            float3 position = toFloat3(aiCam->mPosition);
            float3 forward3 = toFloat3(aiCam->mLookAt);
            float3 up3      = toFloat3(aiCam->mUp);
            // Transform via worldRest: position is a point, forward/up are
            // directions.
            position = mat4_transformPoint(camWorld, position);
            auto rotateDir = [&](float3 d) {
                float x = camWorld.m[0][0]*d.x + camWorld.m[0][1]*d.y + camWorld.m[0][2]*d.z;
                float y = camWorld.m[1][0]*d.x + camWorld.m[1][1]*d.y + camWorld.m[1][2]*d.z;
                float z = camWorld.m[2][0]*d.x + camWorld.m[2][1]*d.y + camWorld.m[2][2]*d.z;
                return make_float3(x, y, z);
            };
            forward3 = rotateDir(forward3);
            up3      = rotateDir(up3);

            if (length(forward3) > 1e-6f) camera.forward = normalize(forward3);
            if (length(up3) > 1e-6f)      camera.up = normalize(up3);
            camera.position = position;
            if (aiCam->mHorizontalFOV > 1e-6f) camera.horizontalFovRadians = aiCam->mHorizontalFOV;
            camera.aspect = aiCam->mAspect;
            if (aiCam->mClipPlaneNear > 1e-6f) camera.nearPlane = aiCam->mClipPlaneNear;
            if (aiCam->mClipPlaneFar > camera.nearPlane) camera.farPlane = aiCam->mClipPlaneFar;
            if (aiCam->mOrthographicWidth > 0.0f) {
                LOG_WARN("Assimp camera %s is orthographic; using its direction but treating it as perspective", aiCam->mName.C_Str());
            }

            LOG_INFO("Loaded camera: %s pos=(%.3f,%.3f,%.3f) fwd=(%.3f,%.3f,%.3f) up=(%.3f,%.3f,%.3f)",
                     aiCam->mName.C_Str(),
                     camera.position.x, camera.position.y, camera.position.z,
                     camera.forward.x, camera.forward.y, camera.forward.z,
                     camera.up.x, camera.up.y, camera.up.z);
            // Bounds for sanity.
        }
    }

    // Parse COLLADA radiance extensions (Assimp ignores <extra> radiance data)
    std::unordered_map<std::string, float3> colladaRadiance;
    std::unordered_set<std::string> colladaCGLAreaLights;
    if (ext == ".dae") {
        colladaRadiance = parseColladaRadiance(path);
        colladaCGLAreaLights = parseColladaCGLAreaLights(path);
    }

    // CPU-decoded emissive textures, keyed by resolved path. Populated lazily
    // during material load (for adaptive emissionStrength normalisation) and
    // reused again during area-light construction (for per-triangle weights).
    struct DecodedEmissiveTex {
        std::vector<unsigned char> pixels;
        int width  = 0;
        int height = 0;
        float3 avgRGB = {0, 0, 0};
        float  avgLum = 0.0f;
        bool   valid  = false;
        bool   loaded = false;
    };
    std::unordered_map<std::string, DecodedEmissiveTex> emissiveTexCache;

    auto getDecodedEmissive = [&](const std::string& texPath) -> DecodedEmissiveTex& {
        auto it = emissiveTexCache.find(texPath);
        if (it != emissiveTexCache.end()) return it->second;
        DecodedEmissiveTex dt;
        dt.loaded = true;
        dt.valid = loadTexturePixelsRGBA8(texPath, dt.pixels, dt.width, dt.height);
        if (dt.valid) {
            dt.avgRGB = computeAverageTextureRGB(dt.pixels.data(), dt.width, dt.height);
            dt.avgLum = luminance(dt.avgRGB);
            LOG_DEBUG("Decoded emissive texture: %s (%dx%d) avgRGB=(%.4f,%.4f,%.4f) avgLum=%.4f",
                      texPath.c_str(), dt.width, dt.height,
                      dt.avgRGB.x, dt.avgRGB.y, dt.avgRGB.z, dt.avgLum);
        } else {
            LOG_WARN("Failed to CPU-decode emissive texture: %s", texPath.c_str());
        }
        auto [ins, _] = emissiveTexCache.emplace(texPath, std::move(dt));
        return ins->second;
    };

    const float kTargetTexturedEmissiveLum = std::max(1e-3f, texturedEmissiveTargetLum);
    const float kMinEmissionStrength       = 1.0f;
    const float kMaxEmissionStrength       = 1000.0f;
    LOG_DEBUG("Adaptive textured-emissive target luminance: %.2f", kTargetTexturedEmissiveLum);

    // Materials — unchanged from the pre-animation loader. Trimmed comments
    // here for brevity; behaviour is identical.
    for (unsigned i = 0; i < aiScn->mNumMaterials; i++) {
        const aiMaterial* aiMat = aiScn->mMaterials[i];
        PBRMaterial mat;
        aiString matName;
        aiMat->Get(AI_MATKEY_NAME, matName);
        const std::string materialName = matName.C_Str();

        aiColor3D color(0.8f, 0.8f, 0.8f);
        if (aiMat->Get(AI_MATKEY_BASE_COLOR, color) != aiReturn_SUCCESS)
            aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
        mat.albedo = toFloat3(color);

        float metallic = 0.0f, roughness = 0.5f;
        bool hasPbrMetallic = (aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metallic) == aiReturn_SUCCESS);
        aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
        if (!hasPbrMetallic) metallic = 0.0f;
        mat.metallic  = metallic;
        mat.roughness = roughness;

        auto colladaIt = colladaRadiance.find(materialName);
        if (colladaIt != colladaRadiance.end()) {
            mat.emission = colladaIt->second;
            mat.emissionStrength = 1.0f;
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

        float ior = 1.5f;
        aiMat->Get(AI_MATKEY_REFRACTI, ior);
        mat.ior = ior;

        float transmissionFactor = 0.0f;
        if (aiMat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmissionFactor) == aiReturn_SUCCESS
            && transmissionFactor > 0.0f) {
            mat.transmission = transmissionFactor;
        }
        float opacity = 1.0f;
        if (aiMat->Get(AI_MATKEY_OPACITY, opacity) == aiReturn_SUCCESS
            && opacity < 0.99f && mat.transmission <= 0.0f) {
            mat.transmission = 1.0f - opacity;
        }

        mat.albedoTexPath        = getTexturePath(aiMat, aiTextureType_BASE_COLOR, baseDir);
        if (mat.albedoTexPath.empty())
            mat.albedoTexPath    = getTexturePath(aiMat, aiTextureType_DIFFUSE, baseDir);
        mat.normalTexPath        = getTexturePath(aiMat, aiTextureType_NORMALS, baseDir);
        mat.metallicRoughTexPath = getTexturePath(aiMat, aiTextureType_METALNESS, baseDir);
        mat.emissiveTexPath      = getTexturePath(aiMat, aiTextureType_EMISSIVE, baseDir);
        mat.specularGlossTexPath = getTexturePath(aiMat, aiTextureType_SPECULAR, baseDir);

        if (sgMode != SGWorkflowMode::Off
            && !mat.specularGlossTexPath.empty()
            && mat.metallicRoughTexPath.empty()
            && !hasPbrMetallic)
        {
            mat.useSpecularGlossiness = true;
            mat.useFBXCustomPacking = (sgMode == SGWorkflowMode::FbxC4D);
            mat.useFBXUEPacking     = (sgMode == SGWorkflowMode::FbxUE);
            mat.metallic = 0.0f;

            aiColor3D specColor(1.0f, 1.0f, 1.0f);
            aiMat->Get(AI_MATKEY_COLOR_SPECULAR, specColor);
            float specLum = luminance(toFloat3(specColor));
            mat.specularColor = (specLum > 1e-4f) ? toFloat3(specColor)
                                                  : make_float3(1.0f, 1.0f, 1.0f);

            std::vector<unsigned char> sgPixels;
            int sgW = 0, sgH = 0;
            bool decoded = loadTexturePixelsRGBA8(mat.specularGlossTexPath, sgPixels, sgW, sgH);
            float alphaMean = 1.0f;
            bool  alphaCarriesGloss = false;
            if (decoded && !sgPixels.empty()) {
                int aMin = 255, aMax = 0;
                double aSum = 0.0;
                size_t n = (size_t)sgW * (size_t)sgH;
                for (size_t k = 0; k < n; k++) {
                    int a = sgPixels[k * 4 + 3];
                    aMin = std::min(aMin, a);
                    aMax = std::max(aMax, a);
                    aSum += a;
                }
                alphaMean = (float)(aSum / (double)n) * (1.0f / 255.0f);
                alphaCarriesGloss = (aMax - aMin) >= 4;
            }

            float shinStrength = 0.0f;
            float shininess    = 0.0f;
            bool hasShinStrength = (aiMat->Get(AI_MATKEY_SHININESS_STRENGTH, shinStrength) == aiReturn_SUCCESS);
            aiMat->Get(AI_MATKEY_SHININESS, shininess);
            if (hasShinStrength && shinStrength > 0.0f) {
                mat.glossiness = std::min(1.0f, shinStrength);
            } else if (shininess > 0.0f) {
                float g = log2f(fmaxf(shininess, 1.0f)) / 13.0f;
                mat.glossiness = std::max(0.0f, std::min(1.0f, g));
            } else {
                mat.glossiness = 0.5f;
            }

            if (!alphaCarriesGloss) {
                mat.glossiness = std::min(mat.glossiness, alphaMean);
                mat.specularGlossAlphaIsGlossiness = false;
            } else {
                mat.specularGlossAlphaIsGlossiness = true;
            }

            mat.roughness = std::max(0.045f, 1.0f - mat.glossiness);
        }

        if (!mat.emissiveTexPath.empty()) {
            mat.emission = make_float3(1.0f, 1.0f, 1.0f);

            const DecodedEmissiveTex& dt = getDecodedEmissive(mat.emissiveTexPath);
            float normMetric = std::max(dt.avgRGB.x,
                                        std::max(dt.avgRGB.y, dt.avgRGB.z));
            if (dt.valid && normMetric > 1e-6f) {
                float strength = kTargetTexturedEmissiveLum / normMetric;
                strength = std::max(kMinEmissionStrength,
                                    std::min(kMaxEmissionStrength, strength));
                mat.emissionStrength = strength;
            } else if (dt.valid && dt.avgLum <= 1e-6f) {
                mat.emission = make_float3(0.0f, 0.0f, 0.0f);
                mat.emissionStrength = 0.0f;
            } else {
                mat.emissionStrength = 10.0f;
            }
        }

        if (ext == ".dae" && !hasPbrMetallic && mat.metallicRoughTexPath.empty()) {
            aiColor3D specularColor(0.0f, 0.0f, 0.0f);
            aiMat->Get(AI_MATKEY_COLOR_SPECULAR, specularColor);
            float specLum = luminance(toFloat3(specularColor));
            if (specLum < 0.01f && mat.transmission <= 0.0f && mat.emissionStrength <= 0.0f) {
                mat.pureDiffuse = true;
                mat.metallic = 0.0f;
            }
        }

        if (mat.emissionStrength > 0.0f && mat.transmission > 0.0f) {
            mat.transmission = 0.0f;
        }

        scene.getMaterials().push_back(std::move(mat));
    }

    // Lights — same as before, but transform position via the SceneNode's
    // worldRest (now that we have one).
    for (unsigned i = 0; i < aiScn->mNumLights; i++) {
        const aiLight* aiL = aiScn->mLights[i];
        if (!aiL) continue;

        LOG_DEBUG("Light[%u] '%s': type=%d color=(%.2f,%.2f,%.2f) atten=(%.4f,%.4f,%.4f)",
                  i, aiL->mName.C_Str(), (int)aiL->mType,
                  aiL->mColorDiffuse.r, aiL->mColorDiffuse.g, aiL->mColorDiffuse.b,
                  aiL->mAttenuationConstant, aiL->mAttenuationLinear, aiL->mAttenuationQuadratic);

        if (aiL->mType == aiLightSource_DIRECTIONAL) {
            DirectionalLight light;

            aiVector3D lightDir = aiL->mDirection;
            const aiNode* lightNode = findNodeByName(aiScn->mRootNode, aiL->mName);
            if (lightNode) {
                aiMatrix4x4 world = computeWorldTransform(lightNode);
                aiMatrix3x3 rotation(world);
                lightDir = rotation * lightDir;
            }

            if (lightDir.SquareLength() > 1e-12f) {
                light.direction = normalize(-toFloat3(lightDir));
            } else {
                light.direction = make_float3(0.0f, -1.0f, 0.0f);
            }

            light.color = toFloat3(aiL->mColorDiffuse);
            light.color *= 0.15f; // Temporarily downscaling for BistroExterior.fbx
            if (light.color.x <= 0.0f && light.color.y <= 0.0f && light.color.z <= 0.0f) {
                light.color = make_float3(1.0f, 1.0f, 1.0f);
            }

            LOG_INFO("  Resolved directional light '%s' -> direction=(%.4f,%.4f,%.4f) color=(%.4f,%.4f,%.4f)",
                     aiL->mName.C_Str(),
                     light.direction.x, light.direction.y, light.direction.z,
                     light.color.x, light.color.y, light.color.z);

            scene.getDirectionalLights().push_back(light);
            continue;
        }

        // Currently only point and spot lights are supported (treat spot as point)
        if (aiL->mType != aiLightSource_POINT && aiL->mType != aiLightSource_SPOT) {
            LOG_WARN("Skipping unsupported light type %d for %s",
                     (int)aiL->mType, aiL->mName.C_Str());
            continue;
        }

        // Skip lights that the COLLADA file marks as area lights via the CGL
        // <extra> extension — those are provided by an emissive mesh elsewhere
        // in the scene, and loading them here would double-count the emitter.
        // If the file contains ANY CGL area-light markers we skip every point
        // light in the file: name matching against Assimp's aiLight::mName is
        // unreliable across Assimp versions (it may hold the light id, the
        // instancing node's id, or the node's name), and .dae files that use
        // the CGL extension are authored so the real illumination comes from
        // emissive mesh geometry — keeping the point light on top double-lit
        // the scene (visible on CBbunny.dae as a washed-out look).
        if (!colladaCGLAreaLights.empty()) {
            LOG_DEBUG("Skipping point light '%s': COLLADA file uses CGL area-light extension",
                      aiL->mName.C_Str());
            continue;
        }

        PointLight light;
        float3 pos = toFloat3(aiL->mPosition);
        const aiNode* lightNode = findNodeByName(aiScn->mRootNode, aiL->mName);
        if (lightNode) {
            auto it = nodeIndex.find(lightNode);
            if (it != nodeIndex.end()) {
                pos = mat4_transformPoint(scene.getNodes()[it->second].worldRest, pos);
            }
        }
        light.position = pos;
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

    // ── Meshes (mesh-local geometry; node binding recorded) ─────────
    attachNodeMeshes(aiScn, aiScn->mRootNode, scene, nodeIndex);

    // ── Animations ──────────────────────────────────────────────────
    if (aiScn->mNumAnimations > 0) {
        // We currently only consume the first clip — that's all the FBX has
        // ("Take 001"). Multi-clip support would slot additional clips here.
        AnimationClip clip = extractAnimation(aiScn->mAnimations[0], anyNameToIdx);
        // Mark animated nodes (and propagate to descendants — a child of a
        // moving node is itself moving even without its own track).
        for (int ni : clip.nodeIndices) {
            if (ni >= 0 && (size_t)ni < scene.getNodes().size()) {
                scene.getNodes()[ni].animated = true;
            }
        }
        // Iterate forward (parent-before-child) and propagate `animated`.
        auto& nodesRef = scene.getNodes();
        for (size_t i = 0; i < nodesRef.size(); i++) {
            int p = nodesRef[i].parent;
            if (p >= 0 && nodesRef[p].animated) nodesRef[i].animated = true;
        }
        scene.getAnimations().push_back(std::move(clip));
        LOG_INFO("Loaded animation: %u channels (post-collapse: %zu node tracks), duration=%.2fs @ %.1f tps",
                 aiScn->mAnimations[0]->mNumChannels,
                 scene.getAnimations().back().channels.size(),
                 scene.getAnimations().back().duration,
                 scene.getAnimations().back().ticksPerSecond);
    }

    // Patch each mesh binding's `animated` flag from its bound node, and at
    // the same time compute scene bounds in *world space* using the rest pose
    // (the BVH/light-BVH/etc. need world-space AABBs; geometry stays mesh-
    // local, but we transform a copy here to expand bounds).
    auto& meshesRef    = scene.getMeshes();
    auto& bindingsRef  = scene.getMeshBindings();
    auto& nodesRef     = scene.getNodes();
    for (size_t i = 0; i < bindingsRef.size(); i++) {
        int ni = bindingsRef[i].nodeIndex;
        if (ni >= 0 && (size_t)ni < nodesRef.size()) {
            bindingsRef[i].animated = nodesRef[ni].animated;
        }
        const float4x4& W = (ni >= 0) ? nodesRef[ni].worldRest : float4x4::identity();
        const auto& mesh = meshesRef[i];
        for (auto& p : mesh.positions) {
            scene.getBounds().expand(mat4_transformPoint(W, p));
        }
    }

    // ── Emissive triangle area lights ─────────────────────────────
    // Triangle world-space positions are computed via the node's worldRest.
    // For *animated* meshes we still emit area-light entries, but the kernel
    // will only see them at their rest-pose location. To avoid them
    // double-contributing or contributing from a stale position when the
    // mesh moves, we mark them isStatic=false and skip including them in
    // the light BVH / NEE; they still emit via BSDF hits onto the moving
    // geometry, which is radiometrically correct.
    const auto& meshes = scene.getMeshes();
    const auto& materials = scene.getMaterials();
    const auto& bindings = scene.getMeshBindings();

    for (size_t mi = 0; mi < meshes.size(); mi++) {
        const auto& mesh = meshes[mi];
        if (mesh.materialIndex < 0 || (size_t)mesh.materialIndex >= materials.size()) continue;

        const PBRMaterial& mat = materials[(size_t)mesh.materialIndex];
        if (mat.emissionStrength <= 0.0f) continue;

        bool hasTexture = !mat.emissiveTexPath.empty();
        const DecodedEmissiveTex* decoded = nullptr;
        if (hasTexture) {
            const DecodedEmissiveTex& dt = getDecodedEmissive(mat.emissiveTexPath);
            if (dt.valid) decoded = &dt;
        }

        float3 baseEmission = mat.emission * mat.emissionStrength;
        float baseEmissionLum = luminance(baseEmission);
        if (!hasTexture && baseEmissionLum <= 0.0f) continue;

        int ni = (mi < bindings.size()) ? bindings[mi].nodeIndex : -1;
        const float4x4& W = (ni >= 0) ? nodesRef[ni].worldRest : float4x4::identity();
        bool meshAnimated = (mi < bindings.size()) ? bindings[mi].animated : false;

        uint32_t triCount = (uint32_t)mesh.indices.size() / 3;
        for (uint32_t t = 0; t < triCount; t++) {
            uint32_t i0 = mesh.indices[t * 3 + 0];
            uint32_t i1 = mesh.indices[t * 3 + 1];
            uint32_t i2 = mesh.indices[t * 3 + 2];

            float3 v0 = mat4_transformPoint(W, mesh.positions[i0]);
            float3 v1 = mat4_transformPoint(W, mesh.positions[i1]);
            float3 v2 = mat4_transformPoint(W, mesh.positions[i2]);
            float3 e1 = v1 - v0;
            float3 e2 = v2 - v0;
            float3 n = cross(e1, e2);
            float area = 0.5f * length(n);
            if (area <= 1e-8f) continue;

            TriangleAreaLight light;
            light.v0 = v0;
            light.e1 = e1;
            light.e2 = e2;
            light.normal = normalize(n);
            light.area = area;
            light.isStatic = !meshAnimated;
            // Animated lights remember which mesh's pose-delta drives them,
            // so the per-frame light-update kernel can refresh their world
            // triangle. Static lights keep meshIndex = -1 and stay frozen.
            light.meshIndex = meshAnimated ? (int)mi : -1;

            if (decoded && !mesh.uvs.empty() &&
                i0 < mesh.uvs.size() && i1 < mesh.uvs.size() && i2 < mesh.uvs.size())
            {
                float2 uv0 = mesh.uvs[i0];
                float2 uv1 = mesh.uvs[i1];
                float2 uv2 = mesh.uvs[i2];
                float avgTexLum = rasterizeTriangleAvgLuminance(
                    uv0, uv1, uv2,
                    decoded->pixels.data(), decoded->width, decoded->height);
                if (avgTexLum <= 0.0f) continue;

                light.uv0 = uv0;
                light.uv1 = uv1;
                light.uv2 = uv2;
                light.emission = make_float3(mat.emissionStrength,
                                             mat.emissionStrength,
                                             mat.emissionStrength);
                light.weight = area * avgTexLum * mat.emissionStrength;
                light.materialIndex = mesh.materialIndex;
            } else {
                if (baseEmissionLum <= 0.0f) continue;
                light.emission = baseEmission;
                light.weight = area * baseEmissionLum;
            }

            scene.getAreaLights().push_back(light);
        }
    }

    {
        const auto& bnd = scene.getBounds();
        LOG_INFO("Scene bounds: min=(%.3f,%.3f,%.3f) max=(%.3f,%.3f,%.3f)",
                 bnd.bmin.x, bnd.bmin.y, bnd.bmin.z,
                 bnd.bmax.x, bnd.bmax.y, bnd.bmax.z);
    }
    LOG_INFO("Loaded: %s (%u meshes, %u materials, %u lights, %u triangles, %u vertices, %u animations)",
             path.c_str(),
             (unsigned)scene.getMeshes().size(),
             (unsigned)scene.getMaterials().size(),
             (unsigned)scene.getLights().size(),
             scene.totalTriangles(),
             scene.totalVertices(),
             (unsigned)scene.getAnimations().size());

    // Count animated/static area lights for sanity.
    {
        size_t nStatic = 0, nDyn = 0;
        for (const auto& l : scene.getAreaLights()) {
            if (l.isStatic) nStatic++; else nDyn++;
        }
        LOG_INFO("Emissive triangle lights: %zu (static=%zu dynamic=%zu)",
                 scene.getAreaLights().size(), nStatic, nDyn);
    }

    return true;
}
