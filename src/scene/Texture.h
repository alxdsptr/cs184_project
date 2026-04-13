#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>

struct TextureData {
    std::vector<unsigned char> pixels;
    int width  = 0;
    int height = 0;
    int channels = 0;
};

class TextureManager {
public:
    // Load image from disk, create CUDA texture object
    cudaTextureObject_t loadTexture(const std::string& path);
    void freeAll();

private:
    struct GPUTexture {
        cudaArray_t         array = nullptr;
        cudaTextureObject_t obj   = 0;
    };
    std::vector<GPUTexture> m_textures;
};
