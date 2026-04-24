#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include <cuda_runtime.h>
#include <cstdint>

class RayTracingBackend;

// Per-pixel reservoir produced by ReSTIR DI.
//
// Notation follows Bitterli et al. 2020 (Spatiotemporal Reservoir Resampling
// for Real-Time Ray Tracing):
//   lightIndex   — global index into DeviceSceneData::d_areaLights of the
//                  currently-held sample. UINT32_MAX means "no sample".
//   baryB1/B2    — barycentric coords of the sample point on that triangle
//                  light (b0 = 1 - b1 - b2), encoding both position and uv.
//   pHat         — target-pdf of the held sample at the pixel's surface,
//                  in the "unshadowed integrand" sense:
//                     pHat = luminance(Le) * |f_r| * G
//                  (no visibility — visibility is tested once at shading).
//   W            — unbiased contribution weight: W * pHat ≈ integrand
//                  estimate, so the final estimator is
//                     integrand_at_sample / pHat * W.
//   M            — effective sample count the reservoir represents
//                  (capped during temporal / spatial reuse to bound bias).
struct ReSTIRReservoir {
    uint32_t lightIndex;
    float    baryB1;
    float    baryB2;
    float    pHat;
    float    W;
    float    M;
    // Two padding floats so sizeof(ReSTIRReservoir) is a multiple of 16 and
    // adjacent-pixel loads in the spatial pass are well-aligned.
    float    _pad0;
    float    _pad1;
};

// Per-pixel surface cached at the primary hit. ReSTIR passes read from this
// to evaluate target pdfs and generate/shade samples without re-tracing the
// camera ray. Kept as Structure-of-Arrays-in-a-Structure to keep coalesced
// reads simple.
struct ReSTIRSurface {
    float3 position;       // world-space primary hit
    float  roughness;
    float3 normal;         // shading normal
    float  metallic;
    float3 albedo;
    float  specProb;       // cached materialSpecProb (for pHat evaluation)
    float3 viewDir;        // V = -rayDir
    float  valid;          // 1.0 = valid surface; 0.0 = miss / skip
    // Screen-space pixel position in the PREVIOUS frame, used by temporal
    // reuse to look up a reservoir from the history buffer. Matches the
    // motion-vector convention written elsewhere: prev - curr in pixels.
    float2 prevPixel;
    uint32_t pureDiffuse;
    uint32_t _pad;
};

struct ReSTIRBuffers {
    // Ping-pong pair for temporal reuse (curr ↔ prev).
    ReSTIRReservoir* d_reservoirsCurr = nullptr;
    ReSTIRReservoir* d_reservoirsPrev = nullptr;
    // Scratch buffer for the spatial-reuse pass output (avoids read/write
    // races on d_reservoirsCurr).
    ReSTIRReservoir* d_reservoirsSpatial = nullptr;
    ReSTIRSurface*   d_surfaceCurr = nullptr;
    ReSTIRSurface*   d_surfacePrev = nullptr;
    uint32_t         width  = 0;
    uint32_t         height = 0;
    uint32_t         prevWidth  = 0;
    uint32_t         prevHeight = 0;
    bool             historyValid = false; // false on first frame / camera reset
};

// ── Kernel launch entry points ──────────────────────────────────
// Cast primary rays, resolve material, run RIS with M candidates from the
// light BVH, and write a fresh reservoir + surface record per pixel.
void launchReSTIRInitialCandidates(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    ReSTIRBuffers          buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               numCandidates);

// Combine each pixel's reservoir with the one at its reprojected position
// in the previous frame.
void launchReSTIRTemporalReuse(
    const DeviceSceneData& scene,
    ReSTIRBuffers          buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               temporalMCap);

// Combine each pixel's reservoir with k neighboring reservoirs within a
// screen-space radius (biased formulation — fast, near-ground-truth in
// practice; no Jacobian).
void launchReSTIRSpatialReuse(
    const DeviceSceneData& scene,
    ReSTIRBuffers          buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               numNeighbors,
    float                  radiusPixels);

// Host-side management of the ReSTIR buffer set.
class ReSTIRContext {
public:
    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void free();
    void swapHistory();       // curr → prev for next frame
    void invalidateHistory(); // call when camera resets / mode changes
    ReSTIRBuffers getBuffers() const { return m_buffers; }

    // Drive the full per-frame ReSTIR pipeline (init → temporal → spatial)
    // and leave the final reservoir in m_buffers.d_reservoirsCurr so the
    // caller can hand that pointer to the main path tracer.
    //
    // If `backend` provides a native ReSTIR init implementation (OptiX
    // backend: raygen against the GAS), it's used for the initial-
    // candidates pass. Otherwise the CUDA kernel runs, which requires
    // scene.d_bvhNodes to be populated by the caller (via patchScene).
    // The temporal + spatial passes always run on CUDA — they read buffers
    // only, no ray tracing.
    //
    // Returns true if the reservoir buffer was freshly populated this
    // frame (main kernel may consume it); false if the pass was skipped
    // (caller must disable ReSTIR consumption this frame to avoid reading
    // last frame's reservoirs as if they were current).
    bool runFrame(const DeviceSceneData& scene, const CameraParams& camera,
                  uint32_t width, uint32_t height, uint32_t sampleIndex,
                  RayTracingBackend* backend = nullptr);

    // Runtime tuning knobs.
    void setNumCandidates(uint32_t n) { m_numCandidates = n; }
    void setTemporalMCap(uint32_t n)  { m_temporalMCap = n; }
    void setNumNeighbors(uint32_t n)  { m_numNeighbors = n; }
    void setSpatialRadius(float r)    { m_spatialRadius = r; }
    void setEnabled(bool on)          { m_enabled = on; }

    uint32_t numCandidates() const { return m_numCandidates; }
    uint32_t temporalMCap()  const { return m_temporalMCap; }
    uint32_t numNeighbors()  const { return m_numNeighbors; }
    float    spatialRadius() const { return m_spatialRadius; }
    bool     enabled()       const { return m_enabled; }

private:
    ReSTIRBuffers m_buffers;
    uint32_t m_numCandidates = 8;     // M initial candidates per pixel
    uint32_t m_temporalMCap  = 20;    // Bitterli's cap = 20 * M_initial
    uint32_t m_numNeighbors  = 3;     // neighbors per spatial pass
    float    m_spatialRadius = 15.0f; // pixels
    bool     m_enabled       = true;
};
