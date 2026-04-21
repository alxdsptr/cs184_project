#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include <cuda_runtime.h>

void launchPathTraceKernel(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    float4* d_accumBuffer,
    float4* d_outputBuffer,
    AuxBufferPtrs auxBuffers,
    uint32_t width,
    uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerPixel = 1,  // spp per kernel invocation
    PrimaryHitSurfaces gbufferSurfaces = {}  // optional Vulkan-shared writes for DLSSOnly
);

#ifdef PATHTRACER_NRD_DLSS_ENABLED
// Split-output surfaces consumed by NRD. Each handle is a cudaSurfaceObject_t
// wrapping a Vulkan-shared VkImage (see VulkanSharedAuxBuffers). Formats:
//   diffuseRadianceHitDist  : RGBA16F   (RGB = demodulated diffuse radiance, A = hitT)
//   specularRadianceHitDist : RGBA16F   (RGB = specular radiance, A = hitT)
//   normalRoughness         : RGBA8     (oct-encoded normal in RG, roughness in A)
//   viewZ                   : R32F      (linear view-space Z, positive in front)
//   motionVectors           : RG16F     (screen-space pixel delta, prev-curr)
//   albedo                  : RGBA8     (diffuse reflectance for composite remodulation)
//   emissive                : RGBA16F   (linear HDR emissive radiance, not denoised)
struct SplitSurfaceOutputs {
    cudaSurfaceObject_t diffuseRadianceHitDist = 0;
    cudaSurfaceObject_t specularRadianceHitDist = 0;
    cudaSurfaceObject_t normalRoughness         = 0;
    cudaSurfaceObject_t viewZ                   = 0;
    cudaSurfaceObject_t motionVectors           = 0;
    cudaSurfaceObject_t albedo                  = 0;
    cudaSurfaceObject_t emissive                = 0;
};

// Split-output variant of the path tracer. Writes demodulated diffuse / specular
// radiance plus the NRD g-buffer directly into Vulkan-shared images.
//
// Path classification: the lobe chosen at the first visible bounce determines
// whether the entire path's light contribution is accumulated into the diffuse
// or specular bucket. NEE direct-lighting contributions at the first hit are
// split by Fresnel weight. Emissive is written to its own (non-denoised) image.
//
// Outputs are demodulated by albedo — the composite pass must remultiply.
void launchPathTraceKernelSplit(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    SplitSurfaceOutputs surfaces,
    uint32_t width,
    uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerPixel = 1  // spp per kernel invocation; averages inside the kernel
);
#endif // PATHTRACER_NRD_DLSS_ENABLED
