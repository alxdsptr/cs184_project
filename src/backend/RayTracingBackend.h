#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include <cuda_runtime.h>

#ifdef PATHTRACER_NRD_DLSS_ENABLED
#include "render/PathTraceKernel.h"  // for SplitSurfaceOutputs
#endif

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

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    // NRD modes: render demodulated diffuse + specular + g-buffer into the
    // Vulkan-shared aux images. CUDA backend uses a SAH-BVH kernel; OptiX
    // backend uses a dedicated split raygen with the same algorithm. Separate
    // entry point so each backend can use its own acceleration structure
    // (CUDA BVH vs OptiX GAS) without the caller knowing or caring.
    virtual void launchPathTraceSplit(
        const DeviceSceneData& scene,
        const CameraParams& camera,
        SplitSurfaceOutputs surfaces,
        uint32_t width, uint32_t height,
        uint32_t sampleIndex,
        bool enableEnvironment,
        uint32_t maxBounces,
        uint32_t samplesPerPixel = 1) = 0;
#endif

    // BDPT-ready: visibility test for connection strategies
    virtual void traceOcclusionRays(
        const float3* d_origins,
        const float3* d_targets,
        bool* d_visible,
        uint32_t rayCount
    ) = 0;

    virtual DeviceSceneData getSceneData() const = 0;

    // Fill in backend-private pointers (triangle BVH nodes, etc.) on `scene`
    // so a caller can launch their own kernels against the same acceleration
    // structure the backend's launchPathTrace uses. The default impl leaves
    // `scene` untouched; CUDABackend overrides it to patch d_bvhNodes /
    // bvhRootIndex. Used by the renderer's ReSTIR prepass, which needs a
    // primary-ray BVH before the main kernel gets a chance to patch one in.
    virtual void patchScene(DeviceSceneData& scene) const { (void)scene; }

    // Run the ReSTIR DI initial-candidates pass for one frame: cast primary
    // rays against this backend's acceleration structure, resolve material
    // at the hit, and stream M light candidates into `d_reservoirsCurr`.
    // `d_surfacesCurr` receives the cached surface record the temporal /
    // spatial CUDA passes will read from.
    //
    // Returns true if the pass ran. Returns false when the backend has no
    // override (and the renderer should fall back to the CUDA-kernel path)
    // or when required scene data (light BVH etc.) is missing.
    virtual bool runReSTIRInitCandidates(
        const DeviceSceneData& /*scene*/,
        const CameraParams&    /*camera*/,
        void*                  /*d_reservoirsCurr*/,
        void*                  /*d_surfacesCurr*/,
        uint32_t               /*width*/,
        uint32_t               /*height*/,
        uint32_t               /*sampleIndex*/,
        uint32_t               /*numCandidates*/) {
        return false;  // backend has no native ReSTIR — caller falls back.
    }

    // Visibility reuse pass: trace one shadow ray per pixel toward the light
    // sample held in `d_reservoirsCurr[i]`, zero W on occlusion. Used to keep
    // occluded samples from poisoning subsequent spatial / temporal reuse
    // (Bitterli 2020 Alg. 5 lines 6-9).
    //
    // Returns true if the pass ran. Returns false when the backend has no
    // native implementation (caller may fall back to a CUDA `bvh_anyHit`
    // kernel if `scene.d_bvhNodes` is populated).
    virtual bool runReSTIRVisibilityReuse(
        const DeviceSceneData& /*scene*/,
        void*                  /*d_reservoirsCurr*/,
        const void*            /*d_surfacesCurr*/,
        uint32_t               /*width*/,
        uint32_t               /*height*/) {
        return false;
    }

    // ReSTIR GI initial-candidates pass. Casts one primary ray per pixel,
    // builds the visible-point surface, samples one BSDF direction, traces
    // the indirect ray to a sample point, captures Lo (emission + 1-bounce
    // NEE), and writes a per-pixel reservoir + surface buffer in the layout
    // CUDA temporal/spatial passes consume. OptiX backend does this with
    // hardware-traced rays against the GAS; CUDA falls back to its SAH-BVH
    // kernel when the backend has no native implementation. Returns true on
    // success.
    virtual bool runReSTIRGIInitCandidates(
        const DeviceSceneData& /*scene*/,
        const CameraParams&    /*camera*/,
        void*                  /*d_giReservoirsCurr*/,
        void*                  /*d_giSurfacesCurr*/,
        uint32_t               /*width*/,
        uint32_t               /*height*/,
        uint32_t               /*sampleIndex*/,
        bool                   /*enableEnvironment*/) {
        return false;
    }
};
