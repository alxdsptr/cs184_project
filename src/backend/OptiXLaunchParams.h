#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include "render/ReSTIR.h"
#include "render/ReSTIRGI.h"
#include "render/ReSTIRPT.h"
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
    // DLSS-RR specific guides (zero in NRD modes; populated in Mode::DLSSRR).
    cudaSurfaceObject_t splitHdrColor;                 // RGBA16F (un-demod noisy color)
    cudaSurfaceObject_t splitWorldNormalRoughness;     // RGBA16F (worldN.xyz, roughness in .w)
    cudaSurfaceObject_t splitSpecAlbedo;               // RGBA16F (EnvBRDFApprox2 per-pixel)
    cudaSurfaceObject_t splitSpecHitT;                 // R32F    (world-space scalar)

    unsigned int    width;
    unsigned int    height;
    unsigned int    sampleIndex;
    unsigned int    maxBounces;
    unsigned int    spp;
    unsigned int    enableEnvironment;

    // Used only by `__raygen__restir_init_candidates`; all-zero in the other
    // raygens. The reservoir + surface buffers mirror the CUDA path's layout
    // (render/ReSTIR.h), so the CUDA temporal / spatial kernels can consume
    // the OptiX raygen's output directly.
    ReSTIRReservoir* restirReservoirsCurr;
    ReSTIRSurface*   restirSurfacesCurr;
    unsigned int     restirNumCandidates;

    // ReSTIR GI init-candidates raygen output. Layout matches the CUDA
    // kernel's so the temporal/spatial passes (still CUDA) can consume
    // either backend's output transparently.
    GIReservoir*     giReservoirsCurr;
    ReSTIRSurface*   giSurfacesCurr;
    unsigned int     giEnableEnvironment;
    unsigned int     giNumCandidates;     // M for the per-pixel RIS (paper §4.1)

    // ReSTIR PT init-candidates raygen output. Same reservoir layout as GI
    // — the difference is in *what's stored*: sampleRadiance is the result
    // of a multi-bounce random walk past the reconnection vertex (path
    // length controlled by `ptPathLength`), not just a 1-bounce NEE.
    GIReservoir*     ptReservoirsCurr;
    ReSTIRSurface*   ptSurfacesCurr;
    unsigned int     ptPathLength;        // bounces past x_r
    unsigned int     ptNumCandidates;     // M for the per-pixel RIS (paper §4.1)

    OptixTraversableHandle handle;
};
