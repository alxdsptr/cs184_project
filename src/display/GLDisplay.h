#pragma once
#include "display/DisplayBackend.h"
#include <cstdint>
#include <string>

struct cudaGraphicsResource;

class GLDisplay : public DisplayBackend {
public:
    void init(uint32_t width, uint32_t height) override;
    void resize(uint32_t width, uint32_t height) override;
    void* mapForCUDA() override;
    void unmapFromCUDA() override;
    void present() override;
    bool saveToPNG(const std::string& path) const;
    void shutdown() override;

private:
    void createPBO();
    void destroyPBO();
    void createShaderProgram();

    uint32_t m_width  = 0;
    uint32_t m_height = 0;

    // OpenGL objects
    unsigned int m_pbo        = 0;
    unsigned int m_texture    = 0;
    unsigned int m_vao        = 0;
    unsigned int m_shaderProg = 0;

    // CUDA interop
    cudaGraphicsResource* m_cudaResource = nullptr;
};
