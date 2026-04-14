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
    // Load LDR image (PNG/JPG/DDS) as RGBA8, create CUDA texture object
    cudaTextureObject_t loadTexture(const std::string& path);

    // Load HDR image (.hdr/.exr) as float4, create CUDA texture object
    // Returns 0 on failure. Also outputs width/height for importance sampling.
    cudaTextureObject_t loadHDRTexture(const std::string& path, int& outWidth, int& outHeight);

    void freeAll();

private:
    struct GPUTexture {
        cudaArray_t         array = nullptr;
        cudaTextureObject_t obj   = 0;
    };
    std::vector<GPUTexture> m_textures;
};
