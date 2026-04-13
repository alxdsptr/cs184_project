#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct AuxBufferPtrs {
    float2* d_motionVectors = nullptr;
    float*  d_linearDepth   = nullptr;
    float3* d_albedo        = nullptr;
    float3* d_normal        = nullptr;
};

class AuxBuffers {
public:
    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void free();
    AuxBufferPtrs getPtrs() const { return m_ptrs; }

private:
    AuxBufferPtrs m_ptrs;
    uint32_t m_width = 0, m_height = 0;
};
