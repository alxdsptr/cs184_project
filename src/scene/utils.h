#pragma once
#include <assimp/material.h>
#include <assimp/matrix4x4.h>
#include <assimp/scene.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace scene_loader_util {

void decompressBC1Block(const uint8_t* block, uint8_t out[4][4][4]);

bool decompressDDS(const std::string& path,
                          std::vector<unsigned char>& outPixels,
                          int& outWidth, int& outHeight);
                          
float3 toFloat3(const aiVector3D& v);
float3 toFloat3(const aiColor3D& c);
float3 transformDirection(const aiMatrix4x4& m, const aiVector3D& v);
float luminance(const float3& c);

std::unordered_map<std::string, float3> parseColladaRadiance(const std::string& path);

// Names of <light> nodes whose <extra><technique profile="CGL"><area> extension
// is present — these should be treated as area lights (provided by an emissive
// mesh elsewhere in the scene) rather than as point/spot lights.
std::unordered_set<std::string> parseColladaCGLAreaLights(const std::string& path);

std::string lowerString(std::string value);
std::string trimString(std::string value);
std::vector<std::string> extractQuotedStrings(const std::string& line);
std::vector<float> extractBracketFloats(const std::string& line);

std::filesystem::path resolvePbrtMeshPath(
    const std::filesystem::path& sceneDir,
    const std::string& relativeMeshPath);

const aiNode* findNodeByName(const aiNode* node, const aiString& name);
aiMatrix4x4 computeWorldTransform(const aiNode* node);

void applyUnitScaling(aiScene* aiScn, const std::string& ext);
std::string getTexturePath(const aiMaterial* mat, aiTextureType type, const std::string& baseDir);

}