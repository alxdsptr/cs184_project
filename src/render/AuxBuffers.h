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

// Split-output surfaces consumed by NRD / DLSS-RR. Each handle is a
// cudaSurfaceObject_t wrapping a Vulkan-shared VkImage (see
// VulkanSharedAuxBuffers). Used by PathTraceKernelSplit and OptiX's
// __raygen__path_trace_split. Defined unconditionally so OptiXLaunchParams
// can embed it without forcing PATHTRACER_NRD_DLSS_ENABLED on every TU.
//
// Formats:
//   diffuseRadianceHitDist  : RGBA16F  (RGB = demodulated diffuse, A = hitT)
//   specularRadianceHitDist : RGBA16F  (RGB = specular, A = hitT)
//   normalRoughness         : RGBA8    (oct-encoded normal in RG, roughness in A)
//   viewZ                   : R32F     (linear view-space Z, positive in front)
//   motionVectors           : RG16F    (screen-space pixel delta, prev-curr)
//   albedo                  : RGBA8    (diffuse reflectance for composite remod)
//   emissive                : RGBA16F  (linear HDR emissive radiance, not denoised)
//   ndcDepth                : R32F     (clip.z / clip.w, DLSS depth input)
//   ── DLSS-RR-specific guides (zero in NRD-only mode):
//   hdrColor                : RGBA16F  (un-demodulated noisy combined color)
//   worldNormalRoughness    : RGBA16F  (world normal.xyz, roughness.w)
//   specAlbedo              : RGBA16F  (EnvBRDFApprox2, DLSS-RR §3.4.2)
//   specHitT                : R32F     (world distance from primary to spec hit)
struct SplitSurfaceOutputs {
    cudaSurfaceObject_t diffuseRadianceHitDist  = 0;
    cudaSurfaceObject_t specularRadianceHitDist = 0;
    cudaSurfaceObject_t normalRoughness         = 0;
    cudaSurfaceObject_t viewZ                   = 0;
    cudaSurfaceObject_t motionVectors           = 0;
    cudaSurfaceObject_t albedo                  = 0;
    cudaSurfaceObject_t emissive                = 0;
    cudaSurfaceObject_t ndcDepth                = 0;
    cudaSurfaceObject_t hdrColor                = 0;
    cudaSurfaceObject_t worldNormalRoughness    = 0;
    cudaSurfaceObject_t specAlbedo              = 0;
    cudaSurfaceObject_t specHitT                = 0;
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
