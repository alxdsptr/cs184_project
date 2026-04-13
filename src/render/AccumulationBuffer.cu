#include "render/AccumulationBuffer.h"
#include "util/CudaCheck.h"

void AccumulationBuffer::init(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;
    size_t pixels = (size_t)width * height;
    CUDA_CHECK(cudaMalloc(&m_accumBuffer,  pixels * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&m_outputBuffer, pixels * sizeof(float4)));
    reset();
}

void AccumulationBuffer::resize(uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height) return;
    free();
    init(width, height);
}

void AccumulationBuffer::reset() {
    if (m_accumBuffer) {
        size_t pixels = (size_t)m_width * m_height;
        CUDA_CHECK(cudaMemset(m_accumBuffer, 0, pixels * sizeof(float4)));
    }
    m_sampleCount = 0;
}

void AccumulationBuffer::free() {
    if (m_accumBuffer)  { cudaFree(m_accumBuffer); m_accumBuffer = nullptr; }
    if (m_outputBuffer) { cudaFree(m_outputBuffer); m_outputBuffer = nullptr; }
    m_width = m_height = 0;
    m_sampleCount = 0;
}
