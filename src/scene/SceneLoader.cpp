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

        // Tangents (computed by aiProcess_CalcTangentSpace). Pack handedness
        // into .w so the kernel can reconstruct B = sign * cross(N, T).
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
            LOG_INFO("Decoded emissive texture: %s (%dx%d) avgRGB=(%.4f,%.4f,%.4f) avgLum=%.4f",
                     texPath.c_str(), dt.width, dt.height,
                     dt.avgRGB.x, dt.avgRGB.y, dt.avgRGB.z, dt.avgLum);
        } else {
            LOG_WARN("Failed to CPU-decode emissive texture: %s", texPath.c_str());
        }
        auto [ins, _] = emissiveTexCache.emplace(texPath, std::move(dt));
        return ins->second;
    };

    // Adaptive emissionStrength for textured emitters: we target a roughly
    // constant *textured linear luminance* across scenes (Bistro, MEASURE_SEVEN,
    // etc.) so one "emissionStrength=50 hardcoded" doesn't over/under-expose
    // depending on texture content. Strength = target / avgLum, clamped to a
    // sensible range so black-background logos don't explode and white lamp
    // plates don't drop to 0.
    const float kTargetTexturedEmissiveLum = 30.0f;
    const float kMinEmissionStrength       = 1.0f;
    const float kMaxEmissionStrength       = 1000.0f;

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

        // Transmission / glass detection
        // 1. glTF KHR_materials_transmission extension
        float transmissionFactor = 0.0f;
        if (aiMat->Get(AI_MATKEY_TRANSMISSION_FACTOR, transmissionFactor) == aiReturn_SUCCESS
            && transmissionFactor > 0.0f) {
            mat.transmission = transmissionFactor;
        }
        // 2. Opacity < 1 implies partial transmission (common in FBX)
        float opacity = 1.0f;
        if (aiMat->Get(AI_MATKEY_OPACITY, opacity) == aiReturn_SUCCESS
            && opacity < 0.99f && mat.transmission <= 0.0f) {
            mat.transmission = 1.0f - opacity;
        }
        if (mat.transmission > 0.0f) {
            LOG_INFO("Material '%s': transmission=%.3f ior=%.3f",
                     materialName.c_str(), mat.transmission, mat.ior);
        }

        // TransparencyFactor from FBX can appear on many materials including
        // emissive ones (e.g. light bulbs in Bistro). Log a notice and
        // disable transmission for emissive materials — they should glow,
        // not refract.
        float transparencyFactor = 0.0f;
        aiMat->Get(AI_MATKEY_TRANSPARENCYFACTOR, transparencyFactor);
        if (transparencyFactor > 0.0f) {
            LOG_INFO("Material '%s': transparencyFactor=%.3f", materialName.c_str(), transparencyFactor);
        }

        // Texture paths
        mat.albedoTexPath        = getTexturePath(aiMat, aiTextureType_BASE_COLOR, baseDir);
        if (mat.albedoTexPath.empty())
            mat.albedoTexPath    = getTexturePath(aiMat, aiTextureType_DIFFUSE, baseDir);
        mat.normalTexPath        = getTexturePath(aiMat, aiTextureType_NORMALS, baseDir);
        mat.metallicRoughTexPath = getTexturePath(aiMat, aiTextureType_METALNESS, baseDir);
        mat.emissiveTexPath      = getTexturePath(aiMat, aiTextureType_EMISSIVE, baseDir);

        // If the material has an emissive texture, it is explicitly meant to
        // emit light. Enable emission even when the scalar emissive color from
        // Assimp was zero (common in FBX files where the texture carries all
        // the data). Adaptive: scale emissionStrength so every textured
        // emitter ends up with roughly the same average linear luminance in
        // world space, regardless of how bright/dark its texture is. This is
        // what keeps Bistro's lamps from blowing out while MEASURE_SEVEN's
        // logo-on-black decals stay visible at the same default "strength".
        if (!mat.emissiveTexPath.empty() && mat.emissionStrength <= 0.0f) {
            mat.emission = make_float3(1.0f, 1.0f, 1.0f);

            const DecodedEmissiveTex& dt = getDecodedEmissive(mat.emissiveTexPath);
            if (dt.valid && dt.avgLum > 1e-6f) {
                float strength = kTargetTexturedEmissiveLum / dt.avgLum;
                strength = std::max(kMinEmissionStrength,
                                    std::min(kMaxEmissionStrength, strength));
                mat.emissionStrength = strength;
                LOG_INFO("Material '%s': adaptive emissionStrength=%.2f "
                         "(avgTexLum=%.4f, target=%.2f)",
                         materialName.c_str(), strength, dt.avgLum,
                         kTargetTexturedEmissiveLum);
            } else if (dt.valid && dt.avgLum <= 1e-6f) {
                // Texture is entirely black — the material isn't actually
                // emissive even though the FBX tagged it so. Skip it rather
                // than letting mat.emission=(1,1,1) make it glow white.
                mat.emission = make_float3(0.0f, 0.0f, 0.0f);
                mat.emissionStrength = 0.0f;
                LOG_INFO("Material '%s': emissive texture is all black, "
                         "treating as non-emissive",
                         materialName.c_str());
            } else {
                // Texture couldn't be decoded at all (e.g. missing file). Keep
                // a conservative strength so the material visibly emits white
                // light — easier to spot and fix than an invisibly dark lamp.
                mat.emissionStrength = 10.0f;
                LOG_WARN("Material '%s': emissive texture not decodable, "
                         "falling back to emissionStrength=%.2f",
                         materialName.c_str(), mat.emissionStrength);
            }
        }

        // Detect legacy Collada Phong materials that should render as pure
        // Lambertian diffuse (matching classic CPU path tracers that ignore the
        // tiny <specular> and high <shininess> from Phong). Only kick in for
        // .dae files and only when there is no PBR metallic factor, no
        // metallic-roughness texture, and the <specular> color is negligible.
        if (ext == ".dae" && !hasPbrMetallic && mat.metallicRoughTexPath.empty()) {
            aiColor3D specularColor(0.0f, 0.0f, 0.0f);
            aiMat->Get(AI_MATKEY_COLOR_SPECULAR, specularColor);
            float specLum = luminance(toFloat3(specularColor));
            if (specLum < 0.01f && mat.transmission <= 0.0f && mat.emissionStrength <= 0.0f) {
                mat.pureDiffuse = true;
                mat.metallic = 0.0f;
                LOG_INFO("Material '%s': treating as pure Lambertian (Collada Phong, specular=%.4f)",
                         materialName.c_str(), specLum);
            }
        }

        // Emissive materials should not be treated as glass/transmissive.
        // FBX often sets opacity < 1 on light bulb materials, but they should
        // glow opaquely, not refract light through them.
        if (mat.emissionStrength > 0.0f && mat.transmission > 0.0f) {
            LOG_INFO("Material '%s': disabling transmission (%.3f) for emissive material",
                     materialName.c_str(), mat.transmission);
            mat.transmission = 0.0f;
        }

        // ── Diagnostic dump ──────────────────────────────────────────
        // Log every channel we read from Assimp plus the resulting PBRMaterial
        // state. Use this to debug "why is emissive yellow instead of green",
        // "why is the floor not reflective", "why is this scene over/under
        // exposed", etc.
        {
            aiColor3D diagDiffuse(0, 0, 0);
            aiColor3D diagBaseColor(0, 0, 0);
            aiColor3D diagSpecular(0, 0, 0);
            aiColor3D diagEmissive(0, 0, 0);
            aiColor3D diagAmbient(0, 0, 0);
            aiColor3D diagReflective(0, 0, 0);
            float diagShininess = 0.0f;
            float diagShininessStrength = 0.0f;
            float diagReflectivity = 0.0f;
            float diagMetallic = -1.0f;
            float diagRoughness = -1.0f;
            float diagSpecFactor = -1.0f;
            float diagEmissiveIntensity = -1.0f;
            float diagOpacity = 1.0f;
            float diagIor = 0.0f;
            aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, diagDiffuse);
            aiMat->Get(AI_MATKEY_BASE_COLOR, diagBaseColor);
            aiMat->Get(AI_MATKEY_COLOR_SPECULAR, diagSpecular);
            aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, diagEmissive);
            aiMat->Get(AI_MATKEY_COLOR_AMBIENT, diagAmbient);
            aiMat->Get(AI_MATKEY_COLOR_REFLECTIVE, diagReflective);
            aiMat->Get(AI_MATKEY_SHININESS, diagShininess);
            aiMat->Get(AI_MATKEY_SHININESS_STRENGTH, diagShininessStrength);
            aiMat->Get(AI_MATKEY_REFLECTIVITY, diagReflectivity);
            aiMat->Get(AI_MATKEY_METALLIC_FACTOR, diagMetallic);
            aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, diagRoughness);
            aiMat->Get(AI_MATKEY_SPECULAR_FACTOR, diagSpecFactor);
            aiMat->Get(AI_MATKEY_EMISSIVE_INTENSITY, diagEmissiveIntensity);
            aiMat->Get(AI_MATKEY_OPACITY, diagOpacity);
            aiMat->Get(AI_MATKEY_REFRACTI, diagIor);

            auto texCount = [&](aiTextureType t) {
                return (unsigned)aiMat->GetTextureCount(t);
            };

            LOG_INFO("─── Material '%s' (idx=%u) ───", materialName.c_str(), i);
            LOG_INFO("  Assimp raw: diffuse=(%.3f,%.3f,%.3f) base=(%.3f,%.3f,%.3f) spec=(%.3f,%.3f,%.3f)",
                     diagDiffuse.r, diagDiffuse.g, diagDiffuse.b,
                     diagBaseColor.r, diagBaseColor.g, diagBaseColor.b,
                     diagSpecular.r, diagSpecular.g, diagSpecular.b);
            LOG_INFO("              emissive=(%.3f,%.3f,%.3f) ambient=(%.3f,%.3f,%.3f) reflective=(%.3f,%.3f,%.3f)",
                     diagEmissive.r, diagEmissive.g, diagEmissive.b,
                     diagAmbient.r, diagAmbient.g, diagAmbient.b,
                     diagReflective.r, diagReflective.g, diagReflective.b);
            LOG_INFO("              shininess=%.3f shin_strength=%.3f reflectivity=%.3f opacity=%.3f ior=%.3f",
                     diagShininess, diagShininessStrength, diagReflectivity, diagOpacity, diagIor);
            LOG_INFO("              metallic_factor=%.3f roughness_factor=%.3f spec_factor=%.3f emissive_intensity=%.3f",
                     diagMetallic, diagRoughness, diagSpecFactor, diagEmissiveIntensity);
            LOG_INFO("  Tex counts: BASE=%u DIFFUSE=%u NORMAL=%u METALNESS=%u DIFF_ROUGH=%u SPECULAR=%u EMISSIVE=%u SHININESS=%u",
                     texCount(aiTextureType_BASE_COLOR),
                     texCount(aiTextureType_DIFFUSE),
                     texCount(aiTextureType_NORMALS),
                     texCount(aiTextureType_METALNESS),
                     texCount(aiTextureType_DIFFUSE_ROUGHNESS),
                     texCount(aiTextureType_SPECULAR),
                     texCount(aiTextureType_EMISSIVE),
                     texCount(aiTextureType_SHININESS));
            LOG_INFO("  Resolved -> albedo=(%.3f,%.3f,%.3f) roughness=%.3f metallic=%.3f transmission=%.3f ior=%.3f pureDiffuse=%d",
                     mat.albedo.x, mat.albedo.y, mat.albedo.z,
                     mat.roughness, mat.metallic, mat.transmission, mat.ior,
                     (int)mat.pureDiffuse);
            LOG_INFO("  Resolved -> emission=(%.3f,%.3f,%.3f) emissionStrength=%.3f",
                     mat.emission.x, mat.emission.y, mat.emission.z, mat.emissionStrength);
            LOG_INFO("  Texture paths: albedo='%s' normal='%s' metRough='%s' emissive='%s'",
                     mat.albedoTexPath.c_str(),
                     mat.normalTexPath.c_str(),
                     mat.metallicRoughTexPath.c_str(),
                     mat.emissiveTexPath.c_str());
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

        // Skip lights that the COLLADA file marks as area lights via the CGL
        // <extra> extension — those are provided by an emissive mesh elsewhere
        // in the scene, and loading them here would double-count the emitter.
        if (!colladaCGLAreaLights.empty() &&
            colladaCGLAreaLights.count(aiL->mName.C_Str()) > 0) {
            LOG_INFO("Skipping point light '%s': COLLADA marks it as CGL area light",
                     aiL->mName.C_Str());
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

    // Emissive triangle area lights.
    //
    // For uniform (no-texture) emitters: the light's emission and weight are a
    // simple product of material emission, albedo, and triangle area.
    //
    // For textured emitters: we additionally CPU-decode the emissive texture
    // once and rasterize each triangle's UV footprint to compute an average
    // texel luminance. The per-triangle weight is area × avgTexLum ×
    // luminance(albedo) × emissionStrength, biasing CDF selection toward
    // brighter regions of the emissive texture. At NEE time the kernel
    // re-samples the texture at the sampled barycentric point to recover the
    // true per-texel emission (see PathTraceKernel.cu).
    const auto& meshes = scene.getMeshes();
    const auto& materials = scene.getMaterials();

    // The per-triangle importance-sampling weights reuse the same CPU-decoded
    // texture cache that adaptive emissionStrength populated during material
    // load (`emissiveTexCache` / `getDecodedEmissive` above). Calling
    // getDecodedEmissive() again here is cheap: the first hit populates the
    // cache, subsequent hits just return the stored pixels.
    for (const auto& mesh : meshes) {
        if (mesh.materialIndex < 0 || (size_t)mesh.materialIndex >= materials.size()) {
            continue;
        }

        const PBRMaterial& mat = materials[(size_t)mesh.materialIndex];
        if (mat.emissionStrength <= 0.0f) {
            continue;
        }

        bool hasTexture = !mat.emissiveTexPath.empty();
        const DecodedEmissiveTex* decoded = nullptr;
        if (hasTexture) {
            const DecodedEmissiveTex& dt = getDecodedEmissive(mat.emissiveTexPath);
            if (dt.valid) decoded = &dt;
            // If the CPU decode fails, we still want to create area lights for
            // this mesh — fall back to a uniform weight based on mat.emission.
        }

        // Baseline emission for uniform emitters: mat.emission × strength.
        // For textured emitters the kernel computes Le = texel × (strength,
        // strength, strength), so the per-light `emission` multiplier is just
        // a scalar replicated across RGB. This matches how the kernel's
        // BSDF-hit emissive path reads texel × strength, so NEE and path-hit
        // MIS contributions stay radiometrically consistent.
        float3 baseEmission = mat.emission * mat.emissionStrength;
        float baseEmissionLum = luminance(baseEmission);

        if (!hasTexture && baseEmissionLum <= 0.0f) continue;

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
            if (area <= 1e-8f) continue;

            TriangleAreaLight light;
            light.v0 = v0;
            light.e1 = e1;
            light.e2 = e2;
            light.normal = normalize(n);
            light.area = area;

            if (decoded && !mesh.uvs.empty() &&
                i0 < mesh.uvs.size() && i1 < mesh.uvs.size() && i2 < mesh.uvs.size())
            {
                float2 uv0 = mesh.uvs[i0];
                float2 uv1 = mesh.uvs[i1];
                float2 uv2 = mesh.uvs[i2];
                float avgTexLum = rasterizeTriangleAvgLuminance(
                    uv0, uv1, uv2,
                    decoded->pixels.data(), decoded->width, decoded->height);
                if (avgTexLum <= 0.0f) continue; // dark region — skip to keep CDF clean

                light.uv0 = uv0;
                light.uv1 = uv1;
                light.uv2 = uv2;
                // For textured emitters, emission is a scalar multiplier
                // applied on top of the texel fetch. Kernel does
                //   Le = tex2D(emissiveTex, uv) * light.emission
                // so storing (strength, strength, strength) gives Le = texel×strength,
                // matching the BSDF-hit branch that computes texel × mat.emissionStrength.
                light.emission = make_float3(mat.emissionStrength,
                                             mat.emissionStrength,
                                             mat.emissionStrength);
                // Weight ~ integrated Le over the triangle:
                //   ∫ texel × strength dA ≈ area × avgTexLum × strength
                light.weight = area * avgTexLum * mat.emissionStrength;
                // Record the material so Application can back-fill the CUDA
                // texture handle after TextureManager loads it.
                light.materialIndex = mesh.materialIndex;
            } else {
                // Uniform emitter (or texture decode failed): fall back to the
                // old behaviour.
                if (baseEmissionLum <= 0.0f) continue;
                light.emission = baseEmission;
                light.weight = area * baseEmissionLum;
            }

            scene.getAreaLights().push_back(light);
        }
    }

    // Store the emissive-texture path → material index mapping through the
    // area light weights; the CUDA texture object is bound later (Application
    // loads textures after SceneLoader). The kernel still needs the handle
    // attached to each light — DeviceScene wires this up via material index.

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
