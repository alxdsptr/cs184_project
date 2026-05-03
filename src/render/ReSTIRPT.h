#pragma once
#include "gpu/DeviceScene.h"
#include "core/Camera.h"
#include "render/AuxBuffers.h"
#include "render/ReSTIR.h"     // ReSTIRSurface
#include "render/ReSTIRGI.h"   // GIReservoir (we reuse the layout)
#include <cuda_runtime.h>
#include <cstdint>

class RayTracingBackend;

// ── ReSTIR PT (Lin et al. 2022, "Generalized Resampled Importance Sampling
//   Foundations of ReSTIR") ───────────────────────────────────────────────
// Per-pixel reservoir of a *path* sample drawn at the primary visible point.
//
// Implementation uses the **reconnection shift** of Lin et al. exclusively:
// each path is parameterised by (visible point q, reconnection vertex x_r,
// outgoing radiance Lo at x_r toward q). Lo is the result of running a
// short BSDF random-walk with NEE at every vertex from x_r onward, so the
// stored sample carries a multi-bounce path postfix — that is the "PT" in
// ReSTIR PT, distinguishing it from ReSTIR GI which restricts the postfix
// to a single NEE shadow ray (k=1 bounce after x_r).
//
// We deliberately reuse `GIReservoir` and `ReSTIRSurface` byte-for-byte
// from ReSTIRGI:
//   • visiblePos, visibleNormal — q, n_q  (primary hit cache for re-eval)
//   • samplePos, sampleNormal   — x_r, n_r (reconnection vertex)
//   • sampleRadiance            — Lo at x_r toward q (multi-bounce result)
//   • pHat, W, M                — RIS bookkeeping (see ReSTIR.h)
//   • isEnv                     — when set, samplePos holds an env direction
// The Jacobian formula is identical to GI's reconnection-shift Jacobian
// (cos_r' / r'^2) / (cos_r / r^2), so the CUDA helpers in
// ReSTIRGIDevice.cuh apply unchanged. The pipeline differs from GI only at
// the *initial-candidates* step (path tracer instead of one NEE).

struct PTBuffers {
    // Ping-pong pair for temporal reuse.
    GIReservoir*    d_reservoirsCurr    = nullptr;
    GIReservoir*    d_reservoirsPrev    = nullptr;
    // Scratch for spatial reuse (avoid RW races on curr).
    GIReservoir*    d_reservoirsSpatial = nullptr;
    // Cached visible-surface (same struct DI/GI use).
    ReSTIRSurface*  d_surfaceCurr       = nullptr;
    ReSTIRSurface*  d_surfacePrev       = nullptr;
    // Per-pixel float3 indirect-radiance output, consumed by the path tracer
    // as the primary-hit indirect contribution (replaces continuation
    // bounces). Same layout as GI's d_indirectOut.
    float3*         d_indirectOut       = nullptr;

    uint32_t width      = 0;
    uint32_t height     = 0;
    uint32_t prevWidth  = 0;
    uint32_t prevHeight = 0;
    bool     historyValid = false;
};

// ── Kernel launch entry points ────────────────────────────────────────────
// CUDA path. The OptiX backend exposes its own raygen for the initial-
// candidates step; the temporal/spatial/shade passes run on CUDA in either
// backend (they read buffers only — no rays).

void launchReSTIRPTInitialCandidates(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    bool                   enableEnvironment,
    uint32_t               pathLength); // bounces past x_r (k+1 vertices total)

void launchReSTIRPTTemporalReuse(
    const DeviceSceneData& scene,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               temporalMCap);

void launchReSTIRPTSpatialReuse(
    const DeviceSceneData& scene,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               numNeighbors,
    float                  radiusPixels,
    uint32_t               spatialMCap);

void launchReSTIRPTShade(
    const DeviceSceneData& scene,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height);

// Host-side management of the ReSTIR PT buffer set.
class ReSTIRPTContext {
public:
    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void free();
    void swapHistory();
    void invalidateHistory();
    PTBuffers getBuffers() const { return m_buffers; }

    // Drives the per-frame ReSTIR PT pipeline:
    //   init candidates (path random-walk) → temporal → spatial → shade.
    // Returns true when d_indirectOut is freshly populated and the path
    // tracer may consume it; false when the pass was skipped (no scene /
    // CUDA BVH for the fall-back path).
    bool runFrame(const DeviceSceneData& scene, const CameraParams& camera,
                  uint32_t width, uint32_t height, uint32_t sampleIndex,
                  bool enableEnvironment,
                  RayTracingBackend* backend = nullptr);

    // Runtime knobs.
    void setTemporalMCap(uint32_t n) { m_temporalMCap = n; }
    void setSpatialMCap(uint32_t n)  { m_spatialMCap  = n; }
    void setNumNeighbors(uint32_t n) { m_numNeighbors = n; }
    void setSpatialRadius(float r)   { m_spatialRadius = r; }
    void setPathLength(uint32_t n)   { m_pathLength = n; }
    void setEnabled(bool on)         { m_enabled = on; }

    uint32_t temporalMCap()  const { return m_temporalMCap; }
    uint32_t spatialMCap()   const { return m_spatialMCap; }
    uint32_t numNeighbors()  const { return m_numNeighbors; }
    float    spatialRadius() const { return m_spatialRadius; }
    uint32_t pathLength()    const { return m_pathLength; }
    bool     enabled()       const { return m_enabled; }

private:
    PTBuffers m_buffers;
    // Knob tuning notes (ground out empirically vs. ground-truth path traced
    // reference on the bundled assets; same defaults as ReSTIR GI plus an
    // extra pathLength=4 — beyond ~4 bounces from x_r the additional variance
    // dominates anything reservoir reuse can recover). Russian roulette inside
    // the random-walk further bounds path length.
    uint32_t m_temporalMCap  = 30;
    uint32_t m_spatialMCap   = 500;
    uint32_t m_numNeighbors  = 3;
    float    m_spatialRadius = 30.0f;
    uint32_t m_pathLength    = 4;     // bounces past the reconnection vertex
    bool     m_enabled       = false; // off by default — opt-in
};
