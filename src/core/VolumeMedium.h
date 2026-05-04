#pragma once
#include "core/Types.h"
#include "core/Math.h"

// Density-field selector. The volume integrator reads `densityKind` from
// VolumeMedium and switches on it inside `mediumDensityAt` (see
// core/VolumeDevice.cuh). All evaluators return a scalar in [0, 1] which
// scales the base extinction coefficient at that point.
enum VolumeDensityKind : uint32_t {
    VolumeDensity_Constant      = 0,
    VolumeDensity_HeightFalloff = 1,  // exp(-(y - yBase)/falloffHeight) ground fog
    VolumeDensity_FBM           = 2,  // 4-octave FBM clamped to [0, 1]
    VolumeDensity_HeightFBM     = 3,  // height falloff modulated by FBM (smoke plume look)
};

// World-axis-aligned participating-medium volume. When `bounded` is true
// the integrator only samples scatter events inside [bmin, bmax]; outside
// the box transmittance is one. With `bounded == false` the volume is
// treated as infinite — useful only with `densityKind != Constant`, since
// constant density over an infinite medium has zero radiance reaching any
// point from infinitely far light.
//
// The base extinction is sigma_a + sigma_s; `density` is a global scale to
// make lookdev easier. `majorantSigmaT` caches the per-channel max
// extinction and is recomputed by the host whenever any of the relevant
// fields change — it is the bound delta/ratio tracking uses.
struct VolumeMedium {
    bool   enabled    = false;
    float3 sigmaA     = make_float3(0.0f, 0.0f, 0.0f);
    float3 sigmaS     = make_float3(0.0f, 0.0f, 0.0f);
    float  density    = 1.0f;
    float  anisotropy = 0.0f;          // Henyey-Greenstein g in [-0.99, 0.99]

    // Bounding box. Must be set when `bounded == true`.
    bool   bounded = false;
    float3 bmin    = make_float3(-1e30f, -1e30f, -1e30f);
    float3 bmax    = make_float3( 1e30f,  1e30f,  1e30f);

    uint32_t densityKind = VolumeDensity_Constant;

    // Height-falloff parameters (used when densityKind selects them).
    // density(p) = exp(-max(0, p.y - yBase) / falloffHeight)
    float yBase         = 0.0f;
    float falloffHeight = 10.0f;

    // FBM parameters (used by FBM and HeightFBM kinds).
    float fbmFrequency  = 0.05f;
    int   fbmOctaves    = 4;
    float fbmLacunarity = 2.0f;
    float fbmGain       = 0.5f;
    float heightFBMMix  = 0.6f;        // HeightFBM: 0 = pure height, 1 = full FBM modulation

    // Cached per-channel majorant of sigma_t over the volume. The host sets
    // this via `recomputeMajorant()` on every parameter change; the device
    // never recomputes. Equals max-channel-σ_t * density * peakDensity,
    // where peakDensity = 1 for every kind above (FBM is clamped to [0,1],
    // height falloff peaks at 1 at y = yBase).
    float majorantSigmaT = 0.0f;

    void recomputeMajorant() {
        float3 t = (sigmaA + sigmaS) * density;
        float m = fmaxf(t.x, fmaxf(t.y, t.z));
        majorantSigmaT = fmaxf(m, 0.0f);
    }
};

// Backward-compat alias. The old name implied homogeneous-only; the struct
// now supports heterogeneous density too. New code should use `VolumeMedium`.
using HomogeneousMedium = VolumeMedium;
