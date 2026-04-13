#include "render/AuxBuffers.h"
#include "util/CudaCheck.h"

void AuxBuffers::init(uint32_t width, uint32_t height) {
    m_width = width;
    m_height = height;
    size_t pixels = (size_t)width * height;
    CUDA_CHECK(cudaMalloc(&m_ptrs.d_motionVectors, pixels * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&m_ptrs.d_linearDepth,   pixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_ptrs.d_albedo,         pixels * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&m_ptrs.d_normal,         pixels * sizeof(float3)));
}

void AuxBuffers::resize(uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height) return;
    free();
    init(width, height);
}

void AuxBuffers::free() {
    if (m_ptrs.d_motionVectors) { cudaFree(m_ptrs.d_motionVectors); }
    if (m_ptrs.d_linearDepth)   { cudaFree(m_ptrs.d_linearDepth); }
    if (m_ptrs.d_albedo)        { cudaFree(m_ptrs.d_albedo); }
    if (m_ptrs.d_normal)        { cudaFree(m_ptrs.d_normal); }
    m_ptrs = AuxBufferPtrs{};
    m_width = m_height = 0;
}
