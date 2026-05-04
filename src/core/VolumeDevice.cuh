#pragma once
// Device-side helpers for the participating-medium integrator. Included by
// all three path-trace kernels (OptiXPrograms.cu, PathTraceKernel.cu,
// PathTraceKernelSplit.cu) so the noise / density / delta-tracking code
// lives in one place.
//
// All functions are `static __forceinline__ __device__` so each translation
// unit gets its own copy with no link-time symbol clashes — this matches the
// pattern used elsewhere in the codebase for shared device helpers and works
// for both nvcc -rdc=true (OptiX) and standalone .cu compilation.

#include "core/Types.h"
#include "core/Math.h"
#include "core/VolumeMedium.h"
#include "accel/AABB.h"
#include "gpu/Random.h"

#ifndef M_PI_F
#  define M_PI_F 3.14159265358979323846f
#endif

// ── Hash + value noise + FBM ────────────────────────────────
// Cheap 3D value noise built on a Jenkins-style integer hash. Returns
// a smoothed value in [0, 1] at any real-valued point. Not stunning
// for film, perfectly fine for animated smoke at game-engine quality.

static __forceinline__ __device__ uint32_t volHash3(int ix, int iy, int iz) {
    // Mix three lattice coords into a single uint32 using large primes
    // chosen for poor correlation across nearby cells.
    uint32_t h = (uint32_t)ix * 73856093u
               ^ (uint32_t)iy * 19349663u
               ^ (uint32_t)iz * 83492791u;
    h += (h << 10u);
    h ^= (h >> 6u);
    h += (h << 3u);
    h ^= (h >> 11u);
    h += (h << 15u);
    return h;
}

static __forceinline__ __device__ float volHashFloat(int ix, int iy, int iz) {
    return (float)volHash3(ix, iy, iz) * (1.0f / 4294967296.0f);
}

static __forceinline__ __device__ float volSmoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

// Trilinearly-interpolated 3D value noise; periodicity is the integer lattice.
static __forceinline__ __device__ float volValueNoise3(float3 p) {
    int ix = (int)floorf(p.x);
    int iy = (int)floorf(p.y);
    int iz = (int)floorf(p.z);
    float fx = p.x - (float)ix;
    float fy = p.y - (float)iy;
    float fz = p.z - (float)iz;
    float ux = volSmoothstep(fx);
    float uy = volSmoothstep(fy);
    float uz = volSmoothstep(fz);

    float c000 = volHashFloat(ix,     iy,     iz    );
    float c100 = volHashFloat(ix + 1, iy,     iz    );
    float c010 = volHashFloat(ix,     iy + 1, iz    );
    float c110 = volHashFloat(ix + 1, iy + 1, iz    );
    float c001 = volHashFloat(ix,     iy,     iz + 1);
    float c101 = volHashFloat(ix + 1, iy,     iz + 1);
    float c011 = volHashFloat(ix,     iy + 1, iz + 1);
    float c111 = volHashFloat(ix + 1, iy + 1, iz + 1);

    float x00 = c000 + (c100 - c000) * ux;
    float x10 = c010 + (c110 - c010) * ux;
    float x01 = c001 + (c101 - c001) * ux;
    float x11 = c011 + (c111 - c011) * ux;
    float y0  = x00 + (x10 - x00) * uy;
    float y1  = x01 + (x11 - x01) * uy;
    return y0 + (y1 - y0) * uz;
}

// Standard FBM: sum of value noise at increasing frequencies, normalised so
// the result stays in [0, 1] (assuming gain*sum_i gain^i ≤ 1, which holds
// for the defaults gain=0.5 and any octave count).
static __forceinline__ __device__ float volFBM(
    float3 p, int octaves, float lacunarity, float gain)
{
    float sum = 0.0f;
    float amp = 1.0f;
    float ampSum = 0.0f;
    float freq = 1.0f;
    // Cap octaves to avoid runaway loops if the host passes garbage.
    if (octaves < 1) octaves = 1;
    if (octaves > 8) octaves = 8;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * volValueNoise3(p * freq);
        ampSum += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    return (ampSum > 0.0f) ? (sum / ampSum) : 0.0f;
}

// ── Density evaluator ───────────────────────────────────────
// Returns the density-multiplier in [0, 1] at world-space point `p`. The
// caller multiplies this by the medium's base sigma_t to get the local
// extinction. Outside the bounded box callers should not invoke this — but
// it returns 0 in that case for safety.
static __forceinline__ __device__ float mediumDensityAt(
    float3 p, const VolumeMedium& medium)
{
    if (medium.bounded) {
        if (p.x < medium.bmin.x || p.x > medium.bmax.x ||
            p.y < medium.bmin.y || p.y > medium.bmax.y ||
            p.z < medium.bmin.z || p.z > medium.bmax.z) {
            return 0.0f;
        }
    }

    switch (medium.densityKind) {
        default:
        case VolumeDensity_Constant:
            return 1.0f;

        case VolumeDensity_HeightFalloff: {
            float h = fmaxf(p.y - medium.yBase, 0.0f);
            float scale = fmaxf(medium.falloffHeight, 1e-3f);
            return expf(-h / scale);
        }

        case VolumeDensity_FBM: {
            float3 q = p * medium.fbmFrequency;
            float n = volFBM(q, medium.fbmOctaves, medium.fbmLacunarity, medium.fbmGain);
            return clampf(n, 0.0f, 1.0f);
        }

        case VolumeDensity_HeightFBM: {
            float h = fmaxf(p.y - medium.yBase, 0.0f);
            float scale = fmaxf(medium.falloffHeight, 1e-3f);
            float heightTerm = expf(-h / scale);
            float3 q = p * medium.fbmFrequency;
            float n = volFBM(q, medium.fbmOctaves, medium.fbmLacunarity, medium.fbmGain);
            // Lerp between the smooth height profile and the height profile
            // modulated by FBM. Mix=0 → pure height fog; Mix=1 → patchy
            // smoke that mostly hugs the ground.
            float modulator = (1.0f - medium.heightFBMMix) + medium.heightFBMMix * (2.0f * n);
            float v = heightTerm * modulator;
            return clampf(v, 0.0f, 1.0f);
        }
    }
}

// ── Volume bounds intersection ──────────────────────────────
// Clip the segment [tmin, tmax] of a ray to the volume's bounds. For
// unbounded media the segment is returned unchanged (with `tmax` capped to
// a finite value to keep delta tracking from looping forever on miss rays).
static __forceinline__ __device__ bool volumeIntersect(
    float3 origin, float3 direction,
    float tmin, float tmax,
    const VolumeMedium& medium,
    float& tEnter, float& tExit)
{
    if (medium.bounded) {
        AABB box(medium.bmin, medium.bmax);
        float3 invDir = safeInvDir(direction);
        return box.intersectT(origin, invDir, tmin, tmax, tEnter, tExit);
    }
    tEnter = tmin;
    tExit = fminf(tmax, 1e6f);
    return tExit > tEnter;
}

// ── Phase function ──────────────────────────────────────────
static __forceinline__ __device__ float phaseHGEval(float cosTheta, float g) {
    // cosTheta = dot(wo, wi) in PBRT outward-direction convention, where
    // both wo and wi point away from the scatter point. Forward scatter
    // (photon barely deflected) corresponds to cosTheta = -1, hence the
    // +2g·cosTheta term — this is PBRT's convention, not Wikipedia's.
    float g2 = g * g;
    float denom = 1.0f + g2 + 2.0f * g * cosTheta;
    float inv = rsqrtf(fmaxf(denom, 1e-6f));
    float inv3 = inv * inv * inv;
    return (1.0f - g2) * (0.25f / M_PI_F) * inv3;
}

static __forceinline__ __device__ float3 phaseHGSample(float3 wo, float g, uint32_t& rng) {
    float u1 = pcg32_float(rng);
    float u2 = pcg32_float(rng);
    float cosTheta;
    if (fabsf(g) < 1e-3f) {
        cosTheta = 1.0f - 2.0f * u1;
    } else {
        // Inverse CDF for HG in PBRT convention: forward scatter corresponds
        // to cosTheta = -1, so the result is negated relative to the
        // Wikipedia derivation. Pairs with the +2g term in phaseHGEval.
        float sqr = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
        cosTheta = -(1.0f + g * g - sqr * sqr) / (2.0f * g);
        cosTheta = clampf(cosTheta, -1.0f, 1.0f);
    }
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * M_PI_F * u2;
    // Build an orthonormal basis around `wo`. Robust to wo aligned with Y.
    float3 a = (fabsf(wo.y) < 0.999f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0);
    float3 T = normalize(cross(a, wo));
    float3 B = cross(wo, T);
    float3 dir = T * (sinTheta * cosf(phi)) + B * (sinTheta * sinf(phi)) + wo * cosTheta;
    return normalize(dir);
}

// ── Delta tracking ──────────────────────────────────────────
// Sample a scatter event in a heterogeneous medium between [tEnter, tExit].
// Uses a constant majorant `medium.majorantSigmaT` and unbiased null-collision
// rejection. Returns true with `tHit` set on a real scatter event; returns
// false (no scatter) otherwise.
//
// Throughput math at the caller: on a real scatter event, multiply by
// sigma_s(p) / sigma_t(p) (which simplifies to constant single-scatter
// albedo for our uniform-σ-with-density-scalar model). On no-scatter the
// caller must NOT additionally multiply by transmittance — delta tracking's
// rejection probability already accounts for it.
static __forceinline__ __device__ bool volumeDeltaTrack(
    float3 origin, float3 direction,
    float tEnter, float tExit,
    const VolumeMedium& medium,
    uint32_t& rng,
    float& tHit)
{
    float mu = medium.majorantSigmaT;
    if (mu <= 0.0f) return false;
    float3 sigmaTBase = (medium.sigmaA + medium.sigmaS) * medium.density;
    // Use the max channel of σ_t to evaluate scatter probability (consistent
    // with `mu` being the per-channel max).
    float sigmaTBaseMax = fmaxf(sigmaTBase.x, fmaxf(sigmaTBase.y, sigmaTBase.z));
    if (sigmaTBaseMax <= 0.0f) return false;

    float t = tEnter;
    // Hard cap to prevent infinite loops on extremely thin media inside very
    // large volumes (e.g. a near-empty box the size of BistroExterior).
    const int kMaxSteps = 256;
    for (int step = 0; step < kMaxSteps; ++step) {
        float u1 = fmaxf(pcg32_float(rng), 1e-7f);
        t += -logf(u1) / mu;
        if (t >= tExit) return false;
        float3 p = origin + direction * t;
        float density = mediumDensityAt(p, medium);
        float sigmaTReal = density * sigmaTBaseMax;
        float pAccept = sigmaTReal / mu;
        if (pcg32_float(rng) < pAccept) {
            tHit = t;
            return true;
        }
        // null collision; loop
    }
    return false;
}

// ── Ratio tracking ──────────────────────────────────────────
// Compute the per-channel transmittance through [tEnter, tExit]. Standard
// ratio tracking with a constant scalar majorant — correctly handles the
// chromatic σ_t case channel-by-channel because `(1 - σ_t/μ)` is in [0, 1]
// per channel when μ ≥ max(σ_t).
static __forceinline__ __device__ float3 volumeRatioTrack(
    float3 origin, float3 direction,
    float tEnter, float tExit,
    const VolumeMedium& medium,
    uint32_t& rng)
{
    float mu = medium.majorantSigmaT;
    if (mu <= 0.0f) return make_float3(1.0f, 1.0f, 1.0f);
    float3 sigmaTBase = (medium.sigmaA + medium.sigmaS) * medium.density;

    float3 T = make_float3(1.0f, 1.0f, 1.0f);
    float t = tEnter;
    const int kMaxSteps = 256;
    for (int step = 0; step < kMaxSteps; ++step) {
        float u1 = fmaxf(pcg32_float(rng), 1e-7f);
        t += -logf(u1) / mu;
        if (t >= tExit) return T;
        float3 p = origin + direction * t;
        float density = mediumDensityAt(p, medium);
        float3 sigmaTReal = sigmaTBase * density;
        float invMu = 1.0f / mu;
        T.x *= fmaxf(1.0f - sigmaTReal.x * invMu, 0.0f);
        T.y *= fmaxf(1.0f - sigmaTReal.y * invMu, 0.0f);
        T.z *= fmaxf(1.0f - sigmaTReal.z * invMu, 0.0f);
        // Early out — once luminance is negligible the remaining steps add
        // nothing visible and waste RNG draws.
        float lum = 0.2126f * T.x + 0.7152f * T.y + 0.0722f * T.z;
        if (lum < 1e-5f) return make_float3(0.0f, 0.0f, 0.0f);
    }
    return T;
}

// Convenience: clip the shadow segment to the volume bounds and ratio-track
// the result. Distance `dist` is the surface-to-light distance; for
// directional lights pass a large value (it will be clipped by the bounds).
static __forceinline__ __device__ float3 volumeShadowTransmittance(
    float3 origin, float3 direction, float dist,
    const VolumeMedium& medium,
    uint32_t& rng)
{
    if (!medium.enabled || medium.majorantSigmaT <= 0.0f) {
        return make_float3(1.0f, 1.0f, 1.0f);
    }
    float tEnter, tExit;
    if (!volumeIntersect(origin, direction, 0.0f, dist, medium, tEnter, tExit)) {
        return make_float3(1.0f, 1.0f, 1.0f);
    }
    return volumeRatioTrack(origin, direction, tEnter, tExit, medium, rng);
}

// ── Single-scatter albedo ───────────────────────────────────
// For our uniform-σ-with-scalar-density model, the in-scatter throughput
// multiplier at a delta-tracked scatter event is simply σ_s/σ_t (the
// density factor cancels). Returns 0 on channels where σ_t is degenerate.
static __forceinline__ __device__ float3 mediumSingleScatterAlbedo(
    const VolumeMedium& medium)
{
    float3 a = medium.sigmaA + medium.sigmaS;
    float3 s = medium.sigmaS;
    return make_float3(
        a.x > 1e-7f ? s.x / a.x : 0.0f,
        a.y > 1e-7f ? s.y / a.y : 0.0f,
        a.z > 1e-7f ? s.z / a.z : 0.0f);
}
