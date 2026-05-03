#pragma once
#include "backend/RayTracingBackend.h"
#include "render/ReSTIRGI.h"  // for GIReservoir
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

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    void launchPathTraceSplit(
        const DeviceSceneData& scene,
        const CameraParams& camera,
        SplitSurfaceOutputs surfaces,
        uint32_t width, uint32_t height,
        uint32_t sampleIndex,
        bool enableEnvironment,
        uint32_t maxBounces,
        uint32_t samplesPerPixel = 1) override;
#endif

    void traceOcclusionRays(
        const float3* d_origins,
        const float3* d_targets,
        bool* d_visible,
        uint32_t rayCount
    ) override;

    DeviceSceneData getSceneData() const override {
        return m_deviceScene.getData();
    }

    // Used by the Renderer's ReSTIR prepass — exposes the OptiX GAS handle
    // via the scene struct so ReSTIR kernels / raygens can trace primary
    // rays against the same acceleration structure launchPathTrace uses.
    // The CUDA backend fills d_bvhNodes; we leave those null (the OptiX
    // raygen uses `handle` instead) and stash the handle in the scene's
    // reserved slot.
    void patchScene(DeviceSceneData& scene) const override {
        // Nothing to patch on scene itself — OptiX uses params.handle, not
        // scene.d_bvhNodes. Keeping the default no-op would also work; this
        // override exists to document the contract and give a hook point
        // when we eventually teach the CUDA spatial/temporal passes to use
        // an OptiX-traced occlusion query.
        (void)scene;
    }

    // Launch the ReSTIR DI initial-candidates raygen. Returns false if the
    // backend isn't initialized. The temporal / spatial passes still run on
    // CUDA (they don't trace rays — they read the reservoir + surface
    // buffers written here).
    bool launchReSTIRInitCandidatesOptiX(
        const DeviceSceneData& scene,
        const CameraParams&    camera,
        void*                  d_reservoirsCurr,  // ReSTIRReservoir*
        void*                  d_surfacesCurr,    // ReSTIRSurface*
        uint32_t               width,
        uint32_t               height,
        uint32_t               sampleIndex,
        uint32_t               numCandidates);

    bool runReSTIRInitCandidates(
        const DeviceSceneData& scene,
        const CameraParams&    camera,
        void*                  d_reservoirsCurr,
        void*                  d_surfacesCurr,
        uint32_t               width,
        uint32_t               height,
        uint32_t               sampleIndex,
        uint32_t               numCandidates) override
    {
        return launchReSTIRInitCandidatesOptiX(
            scene, camera, d_reservoirsCurr, d_surfacesCurr,
            width, height, sampleIndex, numCandidates);
    }

    // Launch the ReSTIR DI visibility-reuse raygen against the GAS. One
    // shadow ray per pixel, zeroes W on occlusion. Returns false if the
    // backend isn't initialized.
    bool launchReSTIRVisibilityReuseOptiX(
        const DeviceSceneData& scene,
        void*                  d_reservoirsCurr,
        const void*            d_surfacesCurr,
        uint32_t               width,
        uint32_t               height);

    bool runReSTIRVisibilityReuse(
        const DeviceSceneData& scene,
        void*                  d_reservoirsCurr,
        const void*            d_surfacesCurr,
        uint32_t               width,
        uint32_t               height) override
    {
        return launchReSTIRVisibilityReuseOptiX(
            scene, d_reservoirsCurr, d_surfacesCurr, width, height);
    }

    // ReSTIR GI initial-candidates raygen. Casts the primary ray, builds the
    // visible-point surface, samples one BSDF direction, traces the indirect
    // ray, samples one NEE at the indirect hit (via lightBVH+shadow ray),
    // packs everything into a GIReservoir written to d_giReservoirsCurr.
    bool launchReSTIRGIInitCandidatesOptiX(
        const DeviceSceneData& scene,
        const CameraParams&    camera,
        void*                  d_giReservoirsCurr,
        void*                  d_giSurfacesCurr,
        uint32_t               width,
        uint32_t               height,
        uint32_t               sampleIndex,
        bool                   enableEnvironment);

    bool runReSTIRGIInitCandidates(
        const DeviceSceneData& scene,
        const CameraParams&    camera,
        void*                  d_giReservoirsCurr,
        void*                  d_giSurfacesCurr,
        uint32_t               width,
        uint32_t               height,
        uint32_t               sampleIndex,
        bool                   enableEnvironment) override
    {
        return launchReSTIRGIInitCandidatesOptiX(
            scene, camera, d_giReservoirsCurr, d_giSurfacesCurr,
            width, height, sampleIndex, enableEnvironment);
    }

    // ReSTIR PT initial-candidates raygen. Casts the primary ray, the first
    // BSDF-sampled bounce to the reconnection vertex, then walks `pathLength`
    // more BSDF→NEE bounces to gather Lo at x_r. Output reservoir layout
    // matches the CUDA kernel (and ReSTIR GI's) so the temporal/spatial/shade
    // passes consume either backend's output unchanged.
    bool launchReSTIRPTInitCandidatesOptiX(
        const DeviceSceneData& scene,
        const CameraParams&    camera,
        void*                  d_ptReservoirsCurr,
        void*                  d_ptSurfacesCurr,
        uint32_t               width,
        uint32_t               height,
        uint32_t               sampleIndex,
        bool                   enableEnvironment,
        uint32_t               pathLength);

    bool runReSTIRPTInitCandidates(
        const DeviceSceneData& scene,
        const CameraParams&    camera,
        void*                  d_ptReservoirsCurr,
        void*                  d_ptSurfacesCurr,
        uint32_t               width,
        uint32_t               height,
        uint32_t               sampleIndex,
        bool                   enableEnvironment,
        uint32_t               pathLength) override
    {
        return launchReSTIRPTInitCandidatesOptiX(
            scene, camera, d_ptReservoirsCurr, d_ptSurfacesCurr,
            width, height, sampleIndex, enableEnvironment, pathLength);
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
    OptixProgramGroup       m_pgRaygenReSTIR = nullptr;  // ReSTIR DI init-candidates raygen
    OptixProgramGroup       m_pgRaygenReSTIRVis = nullptr; // ReSTIR DI visibility-reuse raygen
    OptixProgramGroup       m_pgRaygenReSTIRGI = nullptr;  // ReSTIR GI init-candidates raygen
    OptixProgramGroup       m_pgRaygenReSTIRPT = nullptr;  // ReSTIR PT init-candidates raygen
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
    CUdeviceptr             m_dRaygenReSTIRRecord = 0;
    CUdeviceptr             m_dRaygenReSTIRVisRecord = 0;
    CUdeviceptr             m_dRaygenReSTIRGIRecord = 0;
    CUdeviceptr             m_dRaygenReSTIRPTRecord = 0;

    CUdeviceptr             m_gasOutput      = 0;
    OptixTraversableHandle  m_gasHandle      = 0;

    CUdeviceptr             m_dLaunchParams  = 0;
    CUstream                m_stream         = nullptr;

    OptixPipelineCompileOptions m_pipelineCompileOptions{};

    DeviceScene             m_deviceScene;
    bool                    m_initialized    = false;
};
