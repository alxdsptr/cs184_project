#include "display/GLDisplay.h"
#include "display/GLShaders.h"
#include "util/CudaCheck.h"
#include "util/Log.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <cstdio>
#include <cstdlib>

// ── Shader compilation helper ────────────────────────────────
static unsigned int compileShader(unsigned int type, const char* src) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        LOG_ERROR("Shader compile: %s", log);
    }
    return shader;
}

// ── GLDisplay implementation ─────────────────────────────────
void GLDisplay::init(uint32_t width, uint32_t height) {
    m_width  = width;
    m_height = height;

    createShaderProgram();
    glGenVertexArrays(1, &m_vao);

    // Create texture for display
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    createPBO();
}

void GLDisplay::createPBO() {
    glGenBuffers(1, &m_pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_pbo, cudaGraphicsRegisterFlagsWriteDiscard));
}

void GLDisplay::destroyPBO() {
    if (m_cudaResource) {
        cudaGraphicsUnregisterResource(m_cudaResource);
        m_cudaResource = nullptr;
    }
    if (m_pbo) {
        glDeleteBuffers(1, &m_pbo);
        m_pbo = 0;
    }
}

void GLDisplay::resize(uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height) return;
    m_width  = width;
    m_height = height;

    destroyPBO();

    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    createPBO();
}

void* GLDisplay::mapForCUDA() {
    void* devPtr = nullptr;
    size_t size  = 0;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaResource, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&devPtr, &size, m_cudaResource));
    return devPtr;
}

void GLDisplay::unmapFromCUDA() {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaResource, 0));
}

void GLDisplay::present() {
    // Copy PBO data into texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Draw fullscreen triangle
    glDisable(GL_DEPTH_TEST);
    glUseProgram(m_shaderProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void GLDisplay::createShaderProgram() {
    unsigned int vs = compileShader(GL_VERTEX_SHADER, g_quadVertSrc);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, g_quadFragSrc);
    m_shaderProg = glCreateProgram();
    glAttachShader(m_shaderProg, vs);
    glAttachShader(m_shaderProg, fs);
    glLinkProgram(m_shaderProg);
    int success;
    glGetProgramiv(m_shaderProg, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(m_shaderProg, 512, nullptr, log);
        LOG_ERROR("Shader link: %s", log);
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
}

void GLDisplay::shutdown() {
    destroyPBO();
    if (m_texture)    { glDeleteTextures(1, &m_texture); m_texture = 0; }
    if (m_vao)        { glDeleteVertexArrays(1, &m_vao); m_vao = 0; }
    if (m_shaderProg) { glDeleteProgram(m_shaderProg); m_shaderProg = 0; }
}
