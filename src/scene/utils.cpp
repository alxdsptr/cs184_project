#include <gli/gli.hpp>
#include <gli/convert.hpp>
#include "utils.h"
#include "SceneLoader.h"
#include "core/Math.h"
#include "util/Log.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace scene_loader_util {

// ── BC1 (DXT1) software decompression ─────────────────────────
void decompressBC1Block(const uint8_t* block, uint8_t out[4][4][4]) {
    uint16_t c0 = block[0] | (block[1] << 8);
    uint16_t c1 = block[2] | (block[3] << 8);

    uint8_t colors[4][4]; // [index][rgba]
    // Decode 5-6-5 to 8-8-8
    colors[0][0] = (uint8_t)(((c0 >> 11) & 0x1F) * 255 / 31);
    colors[0][1] = (uint8_t)(((c0 >> 5)  & 0x3F) * 255 / 63);
    colors[0][2] = (uint8_t)(( c0        & 0x1F) * 255 / 31);
    colors[0][3] = 255;

    colors[1][0] = (uint8_t)(((c1 >> 11) & 0x1F) * 255 / 31);
    colors[1][1] = (uint8_t)(((c1 >> 5)  & 0x3F) * 255 / 63);
    colors[1][2] = (uint8_t)(( c1        & 0x1F) * 255 / 31);
    colors[1][3] = 255;

    if (c0 > c1) {
        for (int k = 0; k < 3; k++) {
            colors[2][k] = (uint8_t)((2 * colors[0][k] + colors[1][k] + 1) / 3);
            colors[3][k] = (uint8_t)((colors[0][k] + 2 * colors[1][k] + 1) / 3);
        }
        colors[2][3] = 255;
        colors[3][3] = 255;
    } else {
        for (int k = 0; k < 3; k++) {
            colors[2][k] = (uint8_t)((colors[0][k] + colors[1][k]) / 2);
        }
        colors[2][3] = 255;
        colors[3][0] = colors[3][1] = colors[3][2] = 0;
        colors[3][3] = 0; // transparent black
    }

    uint32_t indices = block[4] | (block[5] << 8) | (block[6] << 16) | (block[7] << 24);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int idx = indices & 0x3;
            indices >>= 2;
            out[row][col][0] = colors[idx][0];
            out[row][col][1] = colors[idx][1];
            out[row][col][2] = colors[idx][2];
            out[row][col][3] = colors[idx][3];
        }
    }
}

// ── BC3 (DXT5) software decompression ─────────────────────────
void decompressBC3Block(const uint8_t* block, uint8_t out[4][4][4]) {
    // First 8 bytes: alpha block
    uint8_t a0 = block[0];
    uint8_t a1 = block[1];
    uint8_t alphas[8];
    alphas[0] = a0;
    alphas[1] = a1;
    if (a0 > a1) {
        for (int i = 1; i <= 6; i++)
            alphas[i + 1] = (uint8_t)(((7 - i) * a0 + i * a1 + 3) / 7);
    } else {
        for (int i = 1; i <= 4; i++)
            alphas[i + 1] = (uint8_t)(((5 - i) * a0 + i * a1 + 2) / 5);
        alphas[6] = 0;
        alphas[7] = 255;
    }
    // 48-bit alpha index table
    uint64_t aBits = 0;
    for (int i = 2; i < 8; i++)
        aBits |= (uint64_t)block[i] << (8 * (i - 2));

    // Last 8 bytes: color block (same as BC1)
    uint8_t colorOut[4][4][4];
    decompressBC1Block(block + 8, colorOut);

    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int pixel = row * 4 + col;
            int aIdx = (int)((aBits >> (3 * pixel)) & 0x7);
            out[row][col][0] = colorOut[row][col][0];
            out[row][col][1] = colorOut[row][col][1];
            out[row][col][2] = colorOut[row][col][2];
            out[row][col][3] = alphas[aIdx];
        }
    }
}

bool decompressDDS(const std::string& path,
                   std::vector<unsigned char>& outPixels,
                   int& outWidth, int& outHeight) {
    gli::texture rawTex = gli::load(path);
    if (rawTex.empty() || rawTex.target() != gli::TARGET_2D) return false;

    gli::texture2d tex2D(rawTex);
    if (tex2D.empty()) return false;

    auto extent = tex2D.extent(0);
    outWidth = extent.x;
    outHeight = extent.y;
    if (outWidth == 0 || outHeight == 0) return false;

    gli::format fmt = tex2D.format();
    bool isBC1 = (fmt == gli::FORMAT_RGB_DXT1_UNORM_BLOCK8 ||
                  fmt == gli::FORMAT_RGB_DXT1_SRGB_BLOCK8 ||
                  fmt == gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8 ||
                  fmt == gli::FORMAT_RGBA_DXT1_SRGB_BLOCK8);
    bool isBC3 = (fmt == gli::FORMAT_RGBA_DXT5_UNORM_BLOCK16 ||
                  fmt == gli::FORMAT_RGBA_DXT5_SRGB_BLOCK16);

    if (!isBC1 && !isBC3) return false; // unsupported, let gli try

    size_t blockSize = isBC1 ? 8 : 16;
    int bw = (outWidth + 3) / 4;
    int bh = (outHeight + 3) / 4;

    const uint8_t* src = static_cast<const uint8_t*>(tex2D.data(0, 0, 0));
    outPixels.resize((size_t)outWidth * outHeight * 4);

    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            const uint8_t* block = src + ((size_t)by * bw + bx) * blockSize;
            uint8_t decoded[4][4][4];
            if (isBC1) decompressBC1Block(block, decoded);
            else       decompressBC3Block(block, decoded);

            for (int row = 0; row < 4; row++) {
                int py = by * 4 + row;
                if (py >= outHeight) break;
                for (int col = 0; col < 4; col++) {
                    int px = bx * 4 + col;
                    if (px >= outWidth) break;
                    size_t off = ((size_t)py * outWidth + px) * 4;
                    outPixels[off + 0] = decoded[row][col][0];
                    outPixels[off + 1] = decoded[row][col][1];
                    outPixels[off + 2] = decoded[row][col][2];
                    outPixels[off + 3] = decoded[row][col][3];
                }
            }
        }
    }
    return true;
}

float3 toFloat3(const aiVector3D& v) { return make_float3(v.x, v.y, v.z); }
float3 toFloat3(const aiColor3D& c) { return make_float3(c.r, c.g, c.b); }

float3 transformDirection(const aiMatrix4x4& m, const aiVector3D& v) {
    return make_float3(
        m.a1 * v.x + m.a2 * v.y + m.a3 * v.z,
        m.b1 * v.x + m.b2 * v.y + m.b3 * v.z,
        m.c1 * v.x + m.c2 * v.y + m.c3 * v.z
    );
}

float luminance(const float3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

std::unordered_map<std::string, float3> parseColladaRadiance(const std::string& path) {
    std::unordered_map<std::string, float3> result;

    std::ifstream in(path);
    if (!in.is_open()) return result;

    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    in.close();

    std::unordered_map<std::string, float3> effectRadiance;
    {
        size_t pos = 0;
        while (true) {
            size_t effectStart = content.find("<effect ", pos);
            if (effectStart == std::string::npos) break;

            size_t idPos = content.find("id=\"", effectStart);
            size_t effectEnd = content.find("</effect>", effectStart);
            if (idPos == std::string::npos || effectEnd == std::string::npos) {
                pos = effectStart + 1;
                continue;
            }
            if (idPos > effectEnd) {
                pos = effectEnd;
                continue;
            }

            size_t idStart = idPos + 4;
            size_t idEnd = content.find('"', idStart);
            if (idEnd == std::string::npos) break;
            std::string effectId = content.substr(idStart, idEnd - idStart);

            size_t radPos = content.find("<radiance>", effectStart);
            if (radPos != std::string::npos && radPos < effectEnd) {
                size_t radStart = radPos + 10;
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

    {
        size_t pos = 0;
        while (true) {
            size_t matStart = content.find("<material ", pos);
            if (matStart == std::string::npos) break;

            size_t matEnd = content.find("</material>", matStart);
            if (matEnd == std::string::npos) matEnd = content.find("/>", matStart);
            if (matEnd == std::string::npos) break;

            size_t namePos = content.find("name=\"", matStart);
            if (namePos != std::string::npos && namePos < matEnd) {
                size_t nameStart = namePos + 6;
                size_t nameEnd = content.find('"', nameStart);
                if (nameEnd != std::string::npos) {
                    std::string matName = content.substr(nameStart, nameEnd - nameStart);

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

std::unordered_set<std::string> parseColladaCGLAreaLights(const std::string& path) {
    std::unordered_set<std::string> result;

    std::ifstream in(path);
    if (!in.is_open()) return result;

    std::string content((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
    in.close();

    size_t pos = 0;
    while (true) {
        size_t lightStart = content.find("<light ", pos);
        if (lightStart == std::string::npos) break;

        size_t lightEnd = content.find("</light>", lightStart);
        if (lightEnd == std::string::npos) break;

        std::string block = content.substr(lightStart, lightEnd - lightStart);

        bool hasCGLArea = false;
        size_t extraPos = block.find("<extra");
        while (extraPos != std::string::npos) {
            size_t extraEnd = block.find("</extra>", extraPos);
            if (extraEnd == std::string::npos) break;
            std::string extraBlock = block.substr(extraPos, extraEnd - extraPos);
            if (extraBlock.find("profile=\"CGL\"") != std::string::npos &&
                extraBlock.find("<area") != std::string::npos) {
                hasCGLArea = true;
                break;
            }
            extraPos = block.find("<extra", extraEnd);
        }

        if (hasCGLArea) {
            auto extractAttr = [&](const std::string& attr) -> std::string {
                std::string needle = attr + "=\"";
                size_t p = block.find(needle);
                if (p == std::string::npos) return "";
                size_t s = p + needle.size();
                size_t e = block.find('"', s);
                if (e == std::string::npos) return "";
                return block.substr(s, e - s);
            };
            std::string id = extractAttr("id");
            std::string name = extractAttr("name");
            if (!id.empty()) result.insert(id);
            if (!name.empty()) result.insert(name);

            // Also record any <node> that instantiates this light — Assimp
            // typically sets aiLight::mName to the instantiating node's name.
            if (!id.empty()) {
                std::string needle = "url=\"#" + id + "\"";
                size_t p = 0;
                while ((p = content.find(needle, p)) != std::string::npos) {
                    size_t instPos = content.rfind("<instance_light", p);
                    if (instPos == std::string::npos) { p += needle.size(); continue; }
                    size_t nodeStart = content.rfind("<node ", instPos);
                    if (nodeStart == std::string::npos) { p += needle.size(); continue; }
                    size_t nodeHeaderEnd = content.find('>', nodeStart);
                    if (nodeHeaderEnd == std::string::npos) { p += needle.size(); continue; }
                    std::string nodeHeader = content.substr(nodeStart, nodeHeaderEnd - nodeStart);

                    auto extractFrom = [&](const std::string& attr) -> std::string {
                        std::string n = attr + "=\"";
                        size_t q = nodeHeader.find(n);
                        if (q == std::string::npos) return "";
                        size_t s = q + n.size();
                        size_t e = nodeHeader.find('"', s);
                        if (e == std::string::npos) return "";
                        return nodeHeader.substr(s, e - s);
                    };
                    std::string nodeId = extractFrom("id");
                    std::string nodeName = extractFrom("name");
                    if (!nodeId.empty()) result.insert(nodeId);
                    if (!nodeName.empty()) result.insert(nodeName);
                    p += needle.size();
                }
            }
        }

        pos = lightEnd + 8;
    }

    return result;
}

std::string lowerString(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return value;
}

std::string trimString(std::string value) {
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!value.empty() && isSpace((unsigned char)value.front())) value.erase(value.begin());
    while (!value.empty() && isSpace((unsigned char)value.back())) value.pop_back();
    return value;
}

std::vector<std::string> extractQuotedStrings(const std::string& line) {
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

std::vector<float> extractBracketFloats(const std::string& line) {
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

std::filesystem::path resolvePbrtMeshPath(
    const std::filesystem::path& sceneDir,
    const std::string& relativeMeshPath) {
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

const aiNode* findNodeByName(const aiNode* node, const aiString& name) {
    if (!node) return nullptr;
    if (node->mName == name) return node;
    for (unsigned i = 0; i < node->mNumChildren; i++) {
        const aiNode* found = findNodeByName(node->mChildren[i], name);
        if (found) return found;
    }
    return nullptr;
}

aiMatrix4x4 computeWorldTransform(const aiNode* node) {
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

void applyUnitScaling(aiScene* aiScn, const std::string& ext) {
    if (!aiScn) return;

    double unitScale = 1.0;
    bool applied = false;

    if (ext == ".fbx") {
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

                if (maxCoord > 100.0f) {
                    unitScale = 0.01;
                    applied = true;
                    LOG_INFO("Detected large FBX coordinates (max: %.1f). Applying 0.01 scale factor.", maxCoord);
                }
            }
        }
    }

    if (applied && std::abs(unitScale - 1.0) > 1e-6) {
        LOG_INFO("Applying unit scale factor: %.6f to %s file", unitScale, ext.c_str());

        for (unsigned m = 0; m < aiScn->mNumMeshes; m++) {
            aiMesh* mesh = aiScn->mMeshes[m];
            for (unsigned v = 0; v < mesh->mNumVertices; v++) {
                mesh->mVertices[v] *= (float)unitScale;
            }
        }

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

// STB image loader — we declare it as extern here because Texture.cpp already
// defines STB_IMAGE_IMPLEMENTATION. Declaring the signatures manually avoids
// pulling in the whole stb_image.h again with a redefinition of the impl.
extern "C" unsigned char* stbi_load(const char*, int*, int*, int*, int);
extern "C" void stbi_image_free(void*);
extern "C" void stbi_set_flip_vertically_on_load(int);

bool loadTexturePixelsRGBA8(const std::string& path,
                            std::vector<unsigned char>& outPixels,
                            int& outWidth, int& outHeight) {
    outPixels.clear();
    outWidth = 0;
    outHeight = 0;
    if (path.empty()) return false;

    // DDS: reuse the existing software decompressor (BC1/BC3). If that fails,
    // give up — we don't pull gli into utils.cpp to keep the CPU path small.
    std::string lower = lowerString(path);
    if (lower.size() >= 4 && lower.substr(lower.size() - 4) == ".dds") {
        if (decompressDDS(path, outPixels, outWidth, outHeight)) return true;
        return false;
    }

    stbi_set_flip_vertically_on_load(0);
    int w = 0, h = 0, c = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &c, 4);
    if (!pixels) return false;

    outWidth = w;
    outHeight = h;
    size_t bytes = (size_t)w * (size_t)h * 4;
    outPixels.resize(bytes);
    std::memcpy(outPixels.data(), pixels, bytes);
    stbi_image_free(pixels);
    return true;
}

namespace {
// Wrap a normalized UV coordinate into [0, 1). Matches cudaAddressModeWrap.
inline float wrap01(float v) {
    float f = v - std::floor(v);
    if (f < 0.0f) f += 1.0f;
    if (f >= 1.0f) f = 0.0f;
    return f;
}

inline float texelLuminanceRGBA8(const unsigned char* px, int width, int height, int x, int y) {
    // Wrap texel coordinates.
    x = x % width;   if (x < 0) x += width;
    y = y % height;  if (y < 0) y += height;
    const unsigned char* p = px + ((size_t)y * width + x) * 4;
    // sRGB -> linear approximation via square (cheap, good enough for weights).
    float r = (p[0] / 255.0f); r *= r;
    float g = (p[1] / 255.0f); g *= g;
    float b = (p[2] / 255.0f); b *= b;
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}
}

float rasterizeTriangleAvgLuminance(
    float2 uv0, float2 uv1, float2 uv2,
    const unsigned char* pixels, int width, int height)
{
    if (!pixels || width <= 0 || height <= 0) return 0.0f;

    // UVs might be outside [0,1]; find the shifted copy whose centroid lies in
    // [0,1]^2, then rasterize in texel space. This is a simplification — for
    // triangles spanning the wrap boundary we accept minor bias.
    float cu = (uv0.x + uv1.x + uv2.x) / 3.0f;
    float cv = (uv0.y + uv1.y + uv2.y) / 3.0f;
    float shiftU = std::floor(cu);
    float shiftV = std::floor(cv);

    float2 p0 = make_float2((uv0.x - shiftU) * width,  (uv0.y - shiftV) * height);
    float2 p1 = make_float2((uv1.x - shiftU) * width,  (uv1.y - shiftV) * height);
    float2 p2 = make_float2((uv2.x - shiftU) * width,  (uv2.y - shiftV) * height);

    float minX = std::min(std::min(p0.x, p1.x), p2.x);
    float maxX = std::max(std::max(p0.x, p1.x), p2.x);
    float minY = std::min(std::min(p0.y, p1.y), p2.y);
    float maxY = std::max(std::max(p0.y, p1.y), p2.y);

    // Add one-texel padding so we don't miss thin triangles and clamp to a
    // reasonable range. Negative indices are wrapped by texelLuminanceRGBA8.
    int ix0 = (int)std::floor(minX) - 1;
    int ix1 = (int)std::ceil(maxX)  + 1;
    int iy0 = (int)std::floor(minY) - 1;
    int iy1 = (int)std::ceil(maxY)  + 1;

    // Edge function (returns 2x signed area of the sub-triangle).
    auto edge = [](float2 a, float2 b, float2 c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    };
    float triArea2 = edge(p0, p1, p2);
    float sign = triArea2 >= 0.0f ? 1.0f : -1.0f;
    float absArea2 = std::abs(triArea2);
    if (absArea2 < 1e-8f) {
        // Degenerate UV triangle — fall back to sampling the centroid texel.
        float u = wrap01(cu - shiftU) * width;
        float v = wrap01(cv - shiftV) * height;
        return texelLuminanceRGBA8(pixels, width, height, (int)u, (int)v);
    }

    double lumSum = 0.0;
    uint64_t count = 0;
    for (int y = iy0; y <= iy1; y++) {
        for (int x = ix0; x <= ix1; x++) {
            float2 p = make_float2((float)x + 0.5f, (float)y + 0.5f);
            float w0 = edge(p1, p2, p) * sign;
            float w1 = edge(p2, p0, p) * sign;
            float w2 = edge(p0, p1, p) * sign;
            if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
                lumSum += texelLuminanceRGBA8(pixels, width, height, x, y);
                count++;
            }
        }
    }

    if (count == 0) {
        // Triangle is smaller than a texel — sample the centroid.
        float u = wrap01(cu - shiftU) * width;
        float v = wrap01(cv - shiftV) * height;
        return texelLuminanceRGBA8(pixels, width, height, (int)u, (int)v);
    }
    return (float)(lumSum / (double)count);
}

std::string getTexturePath(const aiMaterial* mat, aiTextureType type, const std::string& baseDir) {
    if (mat->GetTextureCount(type) > 0) {
        aiString str;
        mat->GetTexture(type, 0, &str);
        std::string texturePath = str.C_Str();

        std::string lowerPath = texturePath;
        std::transform(lowerPath.begin(), lowerPath.end(), lowerPath.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });

        if (lowerPath.find(".exr") != std::string::npos) {
            LOG_WARN("Texture format .exr not supported. Skipping: %s", texturePath.c_str());
            return "";
        }

        std::vector<std::filesystem::path> candidates;

        candidates.push_back(texturePath);
        candidates.push_back(std::filesystem::path(baseDir) / texturePath);

        std::filesystem::path texFile = texturePath;
        std::string filename = texFile.filename().string();
        candidates.push_back(std::filesystem::path(baseDir) / filename);

        if (!baseDir.empty()) {
            auto basePathObj = std::filesystem::path(baseDir);

            for (int i = 0; i < 2 && basePathObj.has_parent_path(); i++) {
                basePathObj = basePathObj.parent_path();
                candidates.push_back(basePathObj / filename);
                candidates.push_back(basePathObj / texturePath);
            }
        }

        for (const auto& candidate : candidates) {
            if (std::filesystem::exists(candidate)) {
                std::string result = candidate.string();
                LOG_INFO("Resolved texture: %s -> %s", texturePath.c_str(), result.c_str());
                return result;
            }
        }

        LOG_WARN("Failed to locate texture: %s (tried %zu paths)", texturePath.c_str(), candidates.size());
        return (std::filesystem::path(baseDir) / texturePath).string();
    }
    return "";
}
}