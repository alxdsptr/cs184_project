#pragma once
// ReSTIR GI device-side helpers shared between the GI initial-candidates,
// temporal-reuse and spatial-reuse kernels in ReSTIRGI.cu. Header-only so
// every kernel sees the same definitions without LTO/RDC headaches.
//
// Reuses `restirEvalBrdf`, `restirLuminance` and friends from ReSTIRDevice.cuh
// (the BRDF model is shared between DI and GI — same surface representation).

#include "render/ReSTIRGI.h"
#include "render/ReSTIRDevice.cuh"
#include "core/Math.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ── Reservoir lifecycle ────────────────────────────────────────────────
__device__ inline void giReservoirReset(GIReservoir& r) {
    r.visiblePos     = make_float3(0, 0, 0);
    r.visibleNormal  = make_float3(0, 1, 0);
    r.samplePos      = make_float3(0, 0, 0);
    r.sampleNormal   = make_float3(0, 1, 0);
    r.sampleRadiance = make_float3(0, 0, 0);
    r.pHat           = 0.0f;
    r.W              = 0.0f;
    r.M              = 0.0f;
    r.isEnv          = 0u;
    r.valid          = 0u;
}

// Connection-direction + geometry term computed at a visible point q for a
// stored sample (x_s, n_s). Returns false when the geometric configuration
// makes the sample unusable (back-facing, zero solid angle, etc.). Outputs:
//   wi   = unit direction q → x_s (or sample direction for env)
//   r2   = squared distance q → x_s (1.0 for env so divisions stay finite)
//   cosQ = max(0, dot(n_q, wi))
//   cosS = max(0, dot(n_s, -wi)) — for env hits set to 1.0
__device__ inline bool giConnect(
    const float3& q, const float3& nq,
    const GIReservoir& r,
    float3& wi, float& r2, float& cosQ, float& cosS)
{
    if (r.isEnv) {
        wi   = r.samplePos;            // direction stored directly
        r2   = 1.0f;                   // env contributes per-direction; no 1/r^2
        cosQ = fmaxf(dot(nq, wi), 0.0f);
        cosS = 1.0f;
        return cosQ > 0.0f;
    }
    float3 d = r.samplePos - q;
    float  d2 = dot(d, d);
    if (d2 < 1e-8f) return false;
    float invLen = rsqrtf(d2);
    wi  = d * invLen;
    r2  = d2;
    cosQ = fmaxf(dot(nq, wi), 0.0f);
    cosS = fmaxf(dot(r.sampleNormal, -wi), 0.0f);
    return (cosQ > 0.0f) && (cosS > 0.0f);
}

// Target pdf in solid-angle measure at the visible point:
//   p̂(q, sample) = luminance(f_r(q, V, wi) * Lo) * cos(θ_q)
// For environment samples, drop the cos_s/r^2 term (env is a directional
// integrand). For surface samples we keep the geometry term implicit in the
// solid-angle measure — `cos_s/r^2` is the direction Jacobian when the
// sample point parameterization is converted, handled in `giJacobian` below.
__device__ inline float giEvalTargetPdf(
    const ReSTIRSurface& surf,
    const GIReservoir&   r,
    float3& wi)
{
    float r2 = 0.0f, cosQ = 0.0f, cosS = 0.0f;
    if (!giConnect(surf.position, surf.normal, r, wi, r2, cosQ, cosS))
        return 0.0f;

    float3 brdf = restirEvalBrdf(surf, wi);
    float fLum  = restirLuminance(brdf);
    if (fLum <= 0.0f) return 0.0f;

    float Lum = restirLuminance(r.sampleRadiance);
    if (Lum <= 0.0f) return 0.0f;

    return fLum * Lum * cosQ;
}

// Jacobian of the change-of-variables from the source visible point to the
// destination one, for a surface (non-env) sample point. Working in
// solid-angle measure on the visible point side, the same x_s subtends a
// different solid angle when seen from q'. From Ouyang et al.:
//   J = (cos_s_dst / r2_dst) / (cos_s_src / r2_src)
// For env samples the direction is invariant, so J = 1.
__device__ inline float giJacobian(
    const float3& qDst, const float3& nqDst,
    const float3& qSrc,
    const GIReservoir& r)
{
    if (r.isEnv) return 1.0f;
    float3 dDst = r.samplePos - qDst;
    float3 dSrc = r.samplePos - qSrc;
    float  r2Dst = dot(dDst, dDst);
    float  r2Src = dot(dSrc, dSrc);
    if (r2Dst < 1e-8f || r2Src < 1e-8f) return 0.0f;
    float invDst = rsqrtf(r2Dst);
    float invSrc = rsqrtf(r2Src);
    float cosSDst = fmaxf(dot(r.sampleNormal, -dDst * invDst), 0.0f);
    float cosSSrc = fmaxf(dot(r.sampleNormal, -dSrc * invSrc), 0.0f);
    if (cosSSrc <= 0.0f) return 0.0f;
    float jac = (cosSDst * r2Src) / (cosSSrc * r2Dst);
    // Symmetric clamp instead of asymmetric reject: rejecting jac<0.1 but
    // not jac>10 (or vice versa) creates a selection bias that systematically
    // favours the brighter side. Clamp on both ends keeps the ratio finite
    // without skewing the distribution.
    if (jac > 10.0f) jac = 10.0f;
    if (jac < 0.1f)  jac = 0.1f;
    (void)nqDst;
    return jac;
}

// ── Reservoir streaming primitives (Bitterli 2020 Alg. 2) ──────────────
// Update the dst reservoir with a freshly-drawn candidate. `wCandidate` is
// the standard RIS weight (target / source pdf).
__device__ inline bool giReservoirUpdate(
    GIReservoir& r, float& wSum,
    const float3& visiblePos, const float3& visibleNormal,
    bool isEnv, const float3& samplePos, const float3& sampleNormal,
    const float3& sampleRadiance,
    float pHat, float wCandidate, float u01)
{
    // Match paper Algorithm 1: M counts every candidate considered, not just
    // the ones with non-zero weight. Skipping the M increment on zero-weight
    // candidates inflates W = wSum/(M·pHat) and produces overexposure under
    // temporal+spatial reuse — same root cause as the DI bug.
    r.M += 1.0f;
    if (!(wCandidate > 0.0f)) return false;
    wSum  += wCandidate;
    if (u01 * wSum < wCandidate) {
        r.visiblePos     = visiblePos;
        r.visibleNormal  = visibleNormal;
        r.samplePos      = samplePos;
        r.sampleNormal   = sampleNormal;
        r.sampleRadiance = sampleRadiance;
        r.pHat           = pHat;
        r.isEnv          = isEnv ? 1u : 0u;
        r.valid          = 1u;
        return true;
    }
    return false;
}

// Combine reservoir `src` into `dst` evaluated at `dstSurf`. The Jacobian
// converts src's solid-angle measure to dst's so the pHat ratio is in
// matching units.
__device__ inline bool giReservoirCombine(
    GIReservoir& dst, float& wSum,
    const ReSTIRSurface& dstSurf,
    const GIReservoir& src,
    float u01)
{
    if (!src.valid || src.M <= 0.0f) {
        dst.M += src.M;
        return false;
    }
    // Re-evaluate the target pdf of src's sample at dst's surface.
    float3 wi;
    GIReservoir tmp = src;
    float pHatAtDst = giEvalTargetPdf(dstSurf, tmp, wi);
    if (!(pHatAtDst > 0.0f)) {
        dst.M += src.M;
        return false;
    }
    float jac = giJacobian(dstSurf.position, dstSurf.normal,
                           src.visiblePos, src);
    if (!(jac > 0.0f)) {
        dst.M += src.M;
        return false;
    }
    float w = pHatAtDst * src.W * src.M * jac;
    bool accepted = false;
    if (w > 0.0f) {
        wSum += w;
        if (u01 * wSum < w) {
            dst.visiblePos     = dstSurf.position;
            dst.visibleNormal  = dstSurf.normal;
            dst.samplePos      = src.samplePos;
            dst.sampleNormal   = src.sampleNormal;
            dst.sampleRadiance = src.sampleRadiance;
            dst.pHat           = pHatAtDst;
            dst.isEnv          = src.isEnv;
            dst.valid          = 1u;
            accepted = true;
        }
    }
    dst.M += src.M;
    return accepted;
}

__device__ inline void giReservoirFinalize(GIReservoir& r, float wSum) {
    if (!r.valid || r.pHat <= 0.0f || r.M <= 0.0f) {
        r.W = 0.0f;
        return;
    }
    r.W = wSum / (r.M * r.pHat);
}
