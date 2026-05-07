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
