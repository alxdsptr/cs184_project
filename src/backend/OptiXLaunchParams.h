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

    // Used only by the dedicated `__raygen__path_trace_split` raygen (NRD).
    // All-zero in the regular raygen launch. Layout/semantics match
    // SplitSurfaceOutputs in render/PathTraceKernel.h. We don't include that
    // header here to avoid forcing PATHTRACER_NRD_DLSS_ENABLED on every TU.
    cudaSurfaceObject_t splitDiffuseRadianceHitDist;   // RGBA16F
    cudaSurfaceObject_t splitSpecularRadianceHitDist;  // RGBA16F
    cudaSurfaceObject_t splitNormalRoughness;          // RGBA8_UNORM (NRD packed)
    cudaSurfaceObject_t splitViewZ;                    // R32F (linear, positive in front)
    cudaSurfaceObject_t splitMotionVectors;            // RG16F (pixel-space prev−curr)
    cudaSurfaceObject_t splitAlbedo;                   // RGBA8_UNORM (demodulation factor)
    cudaSurfaceObject_t splitEmissive;                 // RGBA16F (linear HDR)
    cudaSurfaceObject_t splitNdcDepth;                 // R32F (DLSS depth in [0,1])

    unsigned int    width;
    unsigned int    height;
    unsigned int    sampleIndex;
    unsigned int    maxBounces;
    unsigned int    spp;
    unsigned int    enableEnvironment;
    unsigned int    skipEmissiveInNEE;  // 1 = skip area-light NEE entirely

    OptixTraversableHandle handle;
};
