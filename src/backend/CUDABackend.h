#pragma once
#include "backend/RayTracingBackend.h"
#include "gpu/DeviceScene.h"
#include "accel/BVHNode.h"

class CUDABackend : public RayTracingBackend {
public:
    void buildAccelerationStructure(const Scene& scene) override;
    void launchPathTrace(
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
        PrimaryHitSurfaces gbufferSurfaces = {},
        bool skipEmissiveInNEE = false
    ) override;

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    void launchPathTraceSplit(
        const DeviceSceneData& scene,
        const CameraParams& camera,
        SplitSurfaceOutputs surfaces,
        uint32_t width, uint32_t height,
        uint32_t sampleIndex,
        bool enableEnvironment,
        uint32_t maxBounces,
        uint32_t samplesPerPixel = 1,
        bool skipEmissiveInNEE = false) override;
#endif
    void traceOcclusionRays(
        const float3* d_origins,
        const float3* d_targets,
        bool* d_visible,
        uint32_t rayCount
    ) override;
    DeviceSceneData getSceneData() const override {
        auto data = m_deviceScene.getData();
        data.d_bvhNodes   = m_bvhNodes;
        data.bvhRootIndex = m_bvhRoot;
        return data;
    }
    void updatePointLightsEnabled(const bool* enabledFlags, uint32_t count) override {
        m_deviceScene.updatePointLightsEnabled(enabledFlags, count);
    }

private:
    DeviceScene m_deviceScene;
    BVHNode*    m_bvhNodes = nullptr;
    uint32_t    m_bvhRoot  = 0;
};
