#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include "render/ReSTIR.h"   // reuse ReSTIRSurface for visible-point caching
#include <cuda_runtime.h>
#include <cstdint>

class RayTracingBackend;

// ── ReSTIR GI (Ouyang et al. 2021) ─────────────────────────────────────
// Per-pixel reservoir of an indirect path-vertex sample. The "sample" is a
// (samplePoint, sampleNormal, outgoingRadiance) triplet sitting at the hit of
// a single BSDF-sampled bounce out of the primary visible point. Across
// frames and neighbors we resample these samples to drive a low-noise indirect
// estimate at the visible point.
//
// Notation (visible point = q, sample point = x_s):
//   visiblePos / visibleNormal — q, n_q. Cached per-reservoir so spatial
//                                reuse can apply the proper Jacobian when
//                                reusing a sample produced for q at q' ≠ q.
//   samplePos / sampleNormal   — x_s, n_s. For environment hits, samplePos
//                                holds the sampled direction and isEnv = 1.
//   sampleRadiance             — Lo at x_s in direction (q - x_s) (for an
//                                env hit, Lo is the env color along the
//                                direction).
//   pHat                       — luminance(f_r * Lo) * cos(θ_q), evaluated at
//                                the visible point in solid-angle measure.
//   W                          — unbiased contribution weight; estimator at
//                                q is integrand(q, sample) * W.
//   M                          — effective sample count.
struct GIReservoir {
    // Visible-point context (so spatial reuse can re-evaluate properly).
    float3   visiblePos;
    float3   visibleNormal;
    // Sample-point info.
    float3   samplePos;       // for env hits, this holds the world-space dir
    float3   sampleNormal;    // unused for env hits
    float3   sampleRadiance;  // outgoing radiance at the sample point
    // RIS state.
    float    pHat;
    float    W;
    float    M;
    // Flags + auxiliary fields (struct stays 16-byte aligned).
    uint32_t isEnv;           // 1 = environment hit, samplePos is direction
    uint32_t valid;           // 1 = reservoir holds a usable sample
    // _pad0 caches `c_i = p̂_i^src` evaluated at the source surface where the
    //   sample was born — needed by the generalised pairwise MIS denominator
    //   (Lin et al. 2022 Eq. 38). Accessed via gris_cHat() in the device
    //   header. Repurposing the pad keeps the struct binary-compatible with
    //   prior cudaMalloc/Memset blocks and OptiX launch params.
    float    _pad0;
    // xrRoughness — reconnection-vertex GGX roughness. Used by the paper §7.5
    //   roughness-based connectability gate during shift evaluation. 0 means
    //   "unknown / pure-diffuse" (always reconnectable); env samples set 0.
    float    xrRoughness;
    float    _pad2;
};

struct GIBuffers {
    // Ping-pong pair for temporal reuse.
    GIReservoir*    d_reservoirsCurr    = nullptr;
    GIReservoir*    d_reservoirsPrev    = nullptr;
    // Scratch buffer for spatial reuse (avoids RW races on curr).
    GIReservoir*    d_reservoirsSpatial = nullptr;
    // Per-pixel visible surface (reusing the DI surface struct verbatim).
    ReSTIRSurface*  d_surfaceCurr       = nullptr;
    ReSTIRSurface*  d_surfacePrev       = nullptr;
    // Per-pixel float3 indirect-radiance output, consumed by the path tracer
    // as the "indirect contribution from this pixel's primary hit".
    float3*         d_indirectOut       = nullptr;

    uint32_t width      = 0;
    uint32_t height     = 0;
    uint32_t prevWidth  = 0;
    uint32_t prevHeight = 0;
    bool     historyValid = false;
};

// Kernel launch entry points -----------------------------------------------
void launchReSTIRGIInitialCandidates(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    GIBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    bool                   enableEnvironment,
    uint32_t               temporalMCap);

// `frameIndex` is the monotonic per-display-frame counter (camera.frameIndex).
// Mixed into the kernel RNG seed so ReSTIR keeps exploring path space even
// when sampleIndex is pinned to 0 by continuous camera motion.
void launchReSTIRGITemporalReuse(
    const DeviceSceneData& scene,
    GIBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               frameIndex,
    uint32_t               temporalMCap);

void launchReSTIRGISpatialReuse(
    const DeviceSceneData& scene,
    GIBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               frameIndex,
    uint32_t               numNeighbors,
    float                  radiusPixels,
    uint32_t               spatialMCap);

// Materializes the reservoir into a per-pixel float3 indirect-radiance
// buffer (`d_indirectOut`). The path tracer reads this in lieu of doing its
// own indirect bounces.
void launchReSTIRGIShade(
    const DeviceSceneData& scene,
    GIBuffers              buffers,
    uint32_t               width,
    uint32_t               height);

// Host-side management of the ReSTIR GI buffer set.
class ReSTIRGIContext {
public:
    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void free();
    void swapHistory();
    void invalidateHistory();
    GIBuffers getBuffers() const { return m_buffers; }

    // Drives the per-frame ReSTIR GI pipeline:
    //   init candidates → temporal reuse → spatial reuse → shade.
    // Returns true when `d_indirectOut` was freshly populated this frame and
    // is safe for the path tracer to consume; false when the pass was
    // skipped (e.g., scene has no BVH built CPU-side).
    // `cameraMoved` clamps temporal M to m_motionMCap for this frame —
    // pass m_camera.hasMoved() from the renderer.
    bool runFrame(const DeviceSceneData& scene, const CameraParams& camera,
                  uint32_t width, uint32_t height, uint32_t sampleIndex,
                  bool enableEnvironment,
                  class RayTracingBackend* backend = nullptr,
                  bool cameraMoved = false);

    // Runtime tuning knobs.
    void setTemporalMCap(uint32_t n) { m_temporalMCap = n; }
    void setSpatialMCap(uint32_t n)  { m_spatialMCap  = n; }
    void setNumNeighbors(uint32_t n) { m_numNeighbors = n; }
    void setSpatialRadius(float r)   { m_spatialRadius = r; }
    void setEnabled(bool on)         { m_enabled = on; }

    uint32_t temporalMCap()  const { return m_temporalMCap; }
    uint32_t spatialMCap()   const { return m_spatialMCap; }
    uint32_t numNeighbors()  const { return m_numNeighbors; }
    float    spatialRadius() const { return m_spatialRadius; }
    bool     enabled()       const { return m_enabled; }

private:
    GIBuffers m_buffers;
    uint32_t m_temporalMCap  = 20;    // Bitterli/Lin-style cap on history.
                                       // Lower (~5) is auto-applied while the
                                       // camera is moving; see runFrame().
    uint32_t m_motionMCap    = 10;    // applied while camera is moving
    uint32_t m_spatialMCap   = 500;   // higher cap once spatially fused
    uint32_t m_numNeighbors  = 0;     // spatial reuse off by default
                                       // (combined with temporal reuse it
                                       // produces a stable but biased
                                       // overexposure on glossy/reflective
                                       // surfaces; revisit once we add
                                       // visibility re-test in spatial)
    float    m_spatialRadius = 30.0f; // pixels
    bool     m_enabled       = false; // off by default — opt-in
};
