#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include <cuda_runtime.h>

class Scene;

class RayTracingBackend {
public:
    virtual ~RayTracingBackend() = default;
    virtual void buildAccelerationStructure(const Scene& scene) = 0;
    virtual void launchPathTrace(
        const DeviceSceneData& scene,
        const CameraParams& camera,
        float4* d_accumBuffer,
        float4* d_outputBuffer,
        AuxBufferPtrs auxBuffers,
        uint32_t width, uint32_t height,
        uint32_t sampleIndex,
        bool enableEnvironment,
        uint32_t maxBounces,
        uint32_t samplesPerPixel = 1,
        // DLSSOnly: also write motion/viewZ/HDR into Vulkan-shared surfaces.
        // All-zero (default) → Native behaviour: only the CUDA buffers are
        // written. When `gbufferSurfaces.hdrColor` is non-zero, the kernel
        // bypasses `d_outputBuffer` and writes the HDR result there instead.
        PrimaryHitSurfaces gbufferSurfaces = {}
    ) = 0;

    // BDPT-ready: visibility test for connection strategies
    virtual void traceOcclusionRays(
        const float3* d_origins,
        const float3* d_targets,
        bool* d_visible,
        uint32_t rayCount
    ) = 0;

    virtual DeviceSceneData getSceneData() const = 0;
};
