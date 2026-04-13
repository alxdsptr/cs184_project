#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // from assimp/contrib/stb

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <gli/gli.hpp>
#include <gli/convert.hpp>

#include "scene/Texture.h"
#include "util/CudaCheck.h"
#include "util/Log.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
bool hasDdsExtension(const std::string& path) {
    auto dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }

    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".dds";
}

bool loadDDSWithGliRGBA8(
    const std::string& path,
    std::vector<unsigned char>& outPixels,
    int& outWidth,
    int& outHeight)
{
    gli::texture rawTex = gli::load(path);
    if (rawTex.empty()) {
        return false;
    }

    if (rawTex.target() != gli::TARGET_2D) {
        LOG_WARN("Unsupported DDS texture target (only 2D supported): %s", path.c_str());
        return false;
    }

    gli::texture2d rawTex2D(rawTex);
    if (rawTex2D.empty()) {
        return false;
    }

    gli::texture2d rgbaTex = gli::convert(rawTex2D, gli::FORMAT_RGBA8_UNORM_PACK8);
    if (rgbaTex.empty()) {
        LOG_WARN("Failed to convert DDS to RGBA8: %s", path.c_str());
        return false;
    }

    auto extent = rgbaTex.extent(0);
    if (extent.x == 0 || extent.y == 0) {
        return false;
    }

    outWidth = static_cast<int>(extent.x);
    outHeight = static_cast<int>(extent.y);
    size_t expectedBytes = static_cast<size_t>(outWidth) * static_cast<size_t>(outHeight) * 4;
    if (rgbaTex.size() < expectedBytes) {
        LOG_WARN("DDS conversion produced insufficient pixel data: %s", path.c_str());
        return false;
    }

    const auto* src = static_cast<const unsigned char*>(rgbaTex.data());
    if (!src) {
        return false;
    }

    outPixels.resize(expectedBytes);
    std::memcpy(outPixels.data(), src, expectedBytes);
    return true;
}
}

cudaTextureObject_t TextureManager::loadTexture(const std::string& path) {
    if (path.empty()) return 0;

    int w = 0, h = 0, c = 0;
    std::vector<unsigned char> ddsPixels;
    unsigned char* pixels = nullptr;
    bool pixelsOwnedByStb = false;

    if (hasDdsExtension(path)) {
        if (!loadDDSWithGliRGBA8(path, ddsPixels, w, h)) {
            LOG_WARN("Failed to load DDS texture with gli: %s", path.c_str());
            return 0;
        }
        pixels = ddsPixels.data();
    } else {
        stbi_set_flip_vertically_on_load(0);
        pixels = stbi_load(path.c_str(), &w, &h, &c, 4); // force RGBA
        if (!pixels) {
            LOG_WARN("Failed to load texture: %s", path.c_str());
            return 0;
        }
        pixelsOwnedByStb = true;
    }

    // Create CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray = nullptr;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, pixels, w * 4, w * 4, h, cudaMemcpyHostToDevice));
    if (pixelsOwnedByStb) {
        stbi_image_free(pixels);
    }

    // Create texture object
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    m_textures.push_back({cuArray, texObj});
    LOG_INFO("Loaded texture: %s (%dx%d)", path.c_str(), w, h);
    return texObj;
}

void TextureManager::freeAll() {
    for (auto& t : m_textures) {
        if (t.obj)   cudaDestroyTextureObject(t.obj);
        if (t.array) cudaFreeArray(t.array);
    }
    m_textures.clear();
}
