#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // from assimp/contrib/stb

#include "scene/Texture.h"
#include "util/CudaCheck.h"
#include "util/Log.h"

cudaTextureObject_t TextureManager::loadTexture(const std::string& path) {
    if (path.empty()) return 0;

    int w, h, c;
    stbi_set_flip_vertically_on_load(0);
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &c, 4); // force RGBA
    if (!pixels) {
        LOG_WARN("Failed to load texture: %s", path.c_str());
        return 0;
    }

    // Create CUDA array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray = nullptr;
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, pixels, w * 4, w * 4, h, cudaMemcpyHostToDevice));
    stbi_image_free(pixels);

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
