#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include <cuda_runtime.h>

#ifdef __CUDACC__
#include <optix.h>
#else
// Host side: we only need OptixTraversableHandle as an opaque 64-bit handle.
typedef unsigned long long OptixTraversableHandle;
#endif

struct LaunchParams {
    DeviceSceneData scene;
    CameraParams    camera;

    float4*         accum;
    float4*         output;
    AuxBufferPtrs   aux;

    // Optional Vulkan-shared surfaces written at the primary hit. Used in
    // DLSSOnly / NRD modes so DLSS / NRD can read motion / viewZ as VkImages
    // without an extra copy. When `gbuffer.hdrColor != 0`, the raygen writes
    // the per-pixel HDR result there in addition to (or instead of) `output`.
    PrimaryHitSurfaces gbuffer;

    unsigned int    width;
    unsigned int    height;
    unsigned int    sampleIndex;
    unsigned int    maxBounces;
    unsigned int    spp;
    unsigned int    enableEnvironment;

    OptixTraversableHandle handle;
};
