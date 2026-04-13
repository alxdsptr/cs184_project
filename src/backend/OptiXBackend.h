#pragma once
#include "backend/RayTracingBackend.h"

// Stub for future OptiX backend.
// OptiX replaces: BVH build/traversal, ray generation, closest-hit dispatch.
// Reuses: BRDF.h, MaterialGPU.h, RayTypes.h, tonemapping, accumulation.
class OptiXBackend : public RayTracingBackend {
public:
    void buildAccelerationStructure(const Scene&) override {}
    void launchPathTrace(
        const DeviceSceneData&, const CameraParams&,
        float4*, float4*, AuxBufferPtrs,
        uint32_t, uint32_t, uint32_t) override {}
    void traceOcclusionRays(
        const float3*, const float3*, bool*, uint32_t) override {}
    DeviceSceneData getSceneData() const override { return DeviceSceneData{}; }
};
