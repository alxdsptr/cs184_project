#pragma once
#include <cuda_runtime.h>
#include <cstdint>

struct AuxBufferPtrs {
    float2* d_motionVectors = nullptr;
    float*  d_linearDepth   = nullptr;
    float3* d_albedo        = nullptr;
    float3* d_normal        = nullptr;
};

// Optional CUDA-Vulkan interop surfaces written at the primary hit. Used in
// DLSSOnly mode (and NRD modes via the split kernel) so DLSS / NRD can read
// motion / viewZ as VkImages without an extra copy. If a field is 0, the
// corresponding write is skipped — Native mode passes all zeros.
struct PrimaryHitSurfaces {
    cudaSurfaceObject_t motionVectors = 0;  // RG16F, pixel-space delta
    cudaSurfaceObject_t viewZ         = 0;  // R32F,  linear meters
    cudaSurfaceObject_t hdrColor      = 0;  // RGBA16F, accumulated HDR (replaces d_outputBuffer when set)
    cudaSurfaceObject_t ndcDepth      = 0;  // R32F,  clip.z / clip.w (DLSS depth input)
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
