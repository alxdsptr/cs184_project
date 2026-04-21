#pragma once
#include "backend/RayTracingBackend.h"
#include "gpu/DeviceScene.h"

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <string>

class OptiXBackend : public RayTracingBackend {
public:
    OptiXBackend();
    ~OptiXBackend() override;

    // Loads the OptiX function table, creates the context, module, program
    // groups, pipeline, and SBT. Returns false on failure (caller can fall
    // back to CUDA backend).
    bool init(const std::string& optixirPath);

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
        PrimaryHitSurfaces gbufferSurfaces = {}
    ) override;

    void traceOcclusionRays(
        const float3* d_origins,
        const float3* d_targets,
        bool* d_visible,
        uint32_t rayCount
    ) override;

    DeviceSceneData getSceneData() const override {
        return m_deviceScene.getData();
    }

private:
    bool loadModule(const std::string& optixirPath);
    bool buildPipeline();
    bool buildSBT();
    bool buildGAS(const DeviceSceneData& data);
    void freeGAS();
    void destroyAll();

    OptixDeviceContext      m_ctx            = nullptr;
    OptixModule             m_module         = nullptr;
    OptixProgramGroup       m_pgRaygen       = nullptr;
    OptixProgramGroup       m_pgRaygenSplit  = nullptr;  // NRD split-output raygen
    OptixProgramGroup       m_pgMissRadiance = nullptr;
    OptixProgramGroup       m_pgMissShadow   = nullptr;
    OptixProgramGroup       m_pgHitRadiance  = nullptr;
    OptixProgramGroup       m_pgHitShadow    = nullptr;
    OptixPipeline           m_pipeline       = nullptr;

    CUdeviceptr             m_sbtRecordsBuf  = 0;
    OptixShaderBindingTable m_sbt{};
    // Device pointers for the two raygen records inside m_sbtRecordsBuf —
    // launchPathTrace / launchPathTraceSplit swap `m_sbt.raygenRecord`
    // between them depending on which raygen they want to launch.
    CUdeviceptr             m_dRaygenRecord       = 0;
    CUdeviceptr             m_dRaygenSplitRecord  = 0;

    CUdeviceptr             m_gasOutput      = 0;
    OptixTraversableHandle  m_gasHandle      = 0;

    CUdeviceptr             m_dLaunchParams  = 0;
    CUstream                m_stream         = nullptr;

    OptixPipelineCompileOptions m_pipelineCompileOptions{};

    DeviceScene             m_deviceScene;
    bool                    m_initialized    = false;
};
