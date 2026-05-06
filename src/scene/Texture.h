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
    // Load LDR image (PNG/JPG/DDS) as RGBA8, create CUDA texture object.
    // When sRGB = true, creates an sRGB-format CUDA array so texel fetches
    // are automatically gamma-decoded (sRGB → linear) by hardware. Use this
    // for colour textures (albedo, emissive). Set sRGB = false for data
    // textures (normal maps, metallic-roughness) where the values are already
    // linear and must not be gamma-corrected.
    cudaTextureObject_t loadTexture(const std::string& path, bool sRGB = false);

    // Decode an LDR texture file into RGBA8 pixels on the CPU (no GPU work).
    // Thread-safe — safe to call concurrently from worker threads to overlap
    // I/O and software decompression on asset-heavy scenes. Returns false on
    // failure (logs a warning).
    static bool decodeTextureRGBA8(const std::string& path, TextureData& out);

    // Upload an already-decoded RGBA8 image and create a CUDA texture object.
    // Must be called from the main thread because it pushes into m_textures.
    // `debugPath` is used only for logging.
    cudaTextureObject_t uploadTexture(const TextureData& data, bool sRGB,
                                      const std::string& debugPath);

    // Load HDR image (.hdr/.exr) as float4, create CUDA texture object
    // Returns 0 on failure. Also outputs width/height for importance sampling.
    cudaTextureObject_t loadHDRTexture(const std::string& path, int& outWidth, int& outHeight);

    // Project an HDR equirectangular environment map into L2 (9 coefficient)
    // Spherical Harmonics radiance coefficients. Coefficient order matches
    // `sh_basis9` in gpu/SHEnv.cuh. Returns false on load failure.
    // Re-reads the HDR file via stb_image; the CUDA texture handle itself is
    // opaque so we can't read back from the GPU-side array.
    static bool projectEnvToSH(const std::string& path, float shCoeffsRGB[9][3]);

    void freeAll();

private:
    struct GPUTexture {
        cudaArray_t         array = nullptr;
        cudaTextureObject_t obj   = 0;
    };
    std::vector<GPUTexture> m_textures;
};
