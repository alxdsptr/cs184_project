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
#include "utils.h"

#include <algorithm>
#include <cmath>
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

cudaTextureObject_t TextureManager::loadTexture(const std::string& path, bool sRGB) {
    if (path.empty()) return 0;

    int w = 0, h = 0, c = 0;
    std::vector<unsigned char> ddsPixels;
    unsigned char* pixels = nullptr;
    bool pixelsOwnedByStb = false;

    if (hasDdsExtension(path)) {
        // Try BC1/BC3 software decompression first (gli::convert can't decompress these)
        if (!scene_loader_util::decompressDDS(path, ddsPixels, w, h)) {
            // Fall back to gli convert for uncompressed DDS formats
            if (!loadDDSWithGliRGBA8(path, ddsPixels, w, h)) {
                LOG_WARN("Failed to load DDS texture: %s", path.c_str());
                return 0;
            }
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

    cudaArray_t cuArray = nullptr;

    // Both the sRGB and linear paths upload uchar4 and let the texture unit
    // normalise to [0,1] float on fetch. For sRGB textures we additionally
    // set `texDesc.sRGB = 1`, which makes the hardware apply the sRGB-to-
    // linear transfer function during sampling (IEC 61966-2-1). This works
    // because the channel descriptor is uchar4 and the read mode is
    // NormalizedFloat — the two conditions CUDA requires for HW sRGB decode
    // to take effect. The kernel side still calls tex2D<float4>() and gets
    // linear-space radiance values back, so no shader changes are needed.
    // Memory savings: 4× (was float4 = 16 B/pixel, now uchar4 = 4 B/pixel).
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(
        cuArray, 0, 0, pixels, w * 4, w * 4, h, cudaMemcpyHostToDevice));
    if (pixelsOwnedByStb) {
        stbi_image_free(pixels);
    }
    cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;

    // Create texture object
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = readMode;
    texDesc.normalizedCoords = 1;
    texDesc.sRGB = sRGB ? 1 : 0;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    m_textures.push_back({cuArray, texObj});
    LOG_DEBUG("Loaded texture: %s (%dx%d, sRGB=%d)", path.c_str(), w, h, (int)sRGB);
    return texObj;
}

cudaTextureObject_t TextureManager::loadHDRTexture(const std::string& path, int& outWidth, int& outHeight) {
    if (path.empty()) return 0;

    outWidth = 0;
    outHeight = 0;

    // stb_image can load .hdr (Radiance RGBE) files as float
    stbi_set_flip_vertically_on_load(0);
    int w = 0, h = 0, c = 0;
    float* hdrPixels = stbi_loadf(path.c_str(), &w, &h, &c, 4); // force RGBA float
    if (!hdrPixels) {
        LOG_WARN("Failed to load HDR texture: %s", path.c_str());
        return 0;
    }

    outWidth = w;
    outHeight = h;

    // Create CUDA array with float4 channel format
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaArray_t cuArray = nullptr;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, hdrPixels,
                                    w * sizeof(float4), w * sizeof(float4), h,
                                    cudaMemcpyHostToDevice));
    stbi_image_free(hdrPixels);

    // Create texture object with clamp addressing and linear filtering
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType; // read as float directly
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    m_textures.push_back({cuArray, texObj});
    LOG_DEBUG("Loaded HDR texture: %s (%dx%d)", path.c_str(), w, h);
    return texObj;
}

bool TextureManager::projectEnvToSH(const std::string& path, float shCoeffsRGB[9][3]) {
    for (int i = 0; i < 9; i++) {
        shCoeffsRGB[i][0] = 0.0f;
        shCoeffsRGB[i][1] = 0.0f;
        shCoeffsRGB[i][2] = 0.0f;
    }
    if (path.empty()) return false;

    stbi_set_flip_vertically_on_load(0);
    int w = 0, h = 0, c = 0;
    float* pixels = stbi_loadf(path.c_str(), &w, &h, &c, 4);
    if (!pixels) {
        LOG_WARN("SH projection: failed to open HDR %s", path.c_str());
        return false;
    }

    // Real SH basis in the same order as gpu/SHEnv.cuh::sh_basis9.
    auto shBasis = [](float x, float y, float z, float out[9]) {
        out[0] = 0.282094792f;
        out[1] = 0.488602512f * y;
        out[2] = 0.488602512f * z;
        out[3] = 0.488602512f * x;
        out[4] = 1.092548431f * x * y;
        out[5] = 1.092548431f * y * z;
        out[6] = 0.315391565f * (3.0f * z * z - 1.0f);
        out[7] = 1.092548431f * x * z;
        out[8] = 0.546274215f * (x * x - y * y);
    };

    const float twoPi = 2.0f * 3.14159265358979323846f;
    const float piF   = 3.14159265358979323846f;

    // Riemann integration over the equirectangular grid. The sin(theta)
    // weight is the correct area element; each row contributes differentially,
    // so we must include it to avoid pole overweighting.
    double acc[9][3] = {{0}};
    for (int py = 0; py < h; py++) {
        float v = ((float)py + 0.5f) / (float)h;
        float theta = v * piF;
        float sinT = sinf(theta);
        float cosT = cosf(theta);
        for (int px = 0; px < w; px++) {
            float u = ((float)px + 0.5f) / (float)w;
            float phi = u * twoPi - piF;
            float x = sinT * cosf(phi);
            float y = cosT;
            float z = sinT * sinf(phi);

            const float* texel = &pixels[(py * w + px) * 4];
            float r = texel[0], g = texel[1], b = texel[2];

            // Clamp extreme HDR values (sun discs) to keep SH projection
            // stable — a single 10000-nit pixel otherwise dominates the fit.
            const float clamp = 100.0f;
            float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            if (lum > clamp) {
                float s = clamp / lum;
                r *= s; g *= s; b *= s;
            }

            float basis[9];
            shBasis(x, y, z, basis);

            // dΩ = sin(theta) * (2π/w) * (π/h)
            float dOmega = sinT * (twoPi / (float)w) * (piF / (float)h);
            for (int i = 0; i < 9; i++) {
                acc[i][0] += (double)(r * basis[i] * dOmega);
                acc[i][1] += (double)(g * basis[i] * dOmega);
                acc[i][2] += (double)(b * basis[i] * dOmega);
            }
        }
    }
    stbi_image_free(pixels);

    for (int i = 0; i < 9; i++) {
        shCoeffsRGB[i][0] = (float)acc[i][0];
        shCoeffsRGB[i][1] = (float)acc[i][1];
        shCoeffsRGB[i][2] = (float)acc[i][2];
    }
    LOG_INFO("Projected env map to SH (L2, 9 coeffs): %s", path.c_str());
    return true;
}

void TextureManager::freeAll() {
    for (auto& t : m_textures) {
        if (t.obj)   cudaDestroyTextureObject(t.obj);
        if (t.array) cudaFreeArray(t.array);
    }
    m_textures.clear();
}
