#pragma once
// ReSTIR DI device-side helpers shared between the CUDA ReSTIR kernel
// (render/ReSTIR.cu) and the OptiX raygen (backend/OptiXPrograms.cu).
//
// Contains the target-pdf evaluator, the reservoir streaming primitives
// (Bitterli Alg. 2), and a tiny generateRay duplicate so the OptiX raygen
// can cast a camera ray identical to the main kernel's.
//
// All functions are header-only `__device__ inline` so including this from
// multiple translation units does not cause multiple-definition link errors
// under nvrtc / optixir.

#include "render/ReSTIR.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/RayTypes.h"
#include "core/Math.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

__device__ inline float restirLuminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ inline float restirGgxD(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * denom * denom + 1e-14f);
}
__device__ inline float restirSmithG1(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    float cos2 = NdotX * NdotX;
    return 2.0f * NdotX / (NdotX + sqrtf(a2 + (1.0f - a2) * cos2) + 1e-7f);
}
__device__ inline float3 restirFresnelSchlick(float cosTheta, float3 F0) {
    float t = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    return F0 + (make_float3(1,1,1) - F0) * t5;
}

__device__ inline float3 restirEvalBrdf(
    const ReSTIRSurface& s, const float3& L)
{
    float NdotL = fmaxf(dot(s.normal, L), 0.0f);
    float NdotV = fmaxf(dot(s.normal, s.viewDir), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0,0,0);
    if (s.pureDiffuse) return s.albedo * (1.0f / M_PI_F);

    float3 H = normalize(s.viewDir + L);
    float NdotH = fmaxf(dot(s.normal, H), 0.0f);
    float LdotH = fmaxf(dot(L, H), 0.0f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), s.albedo, s.metallic);
    float3 F = restirFresnelSchlick(LdotH, F0);
    float Dt = restirGgxD(NdotH, s.roughness);
    float alpha = s.roughness * s.roughness;
    float Gt = restirSmithG1(NdotL, alpha) * restirSmithG1(NdotV, alpha);

    float3 spec = F * (Dt * Gt / (4.0f * NdotL * NdotV + 1e-7f));
    float3 kd = (make_float3(1,1,1) - F) * (1.0f - s.metallic);
    float3 diff = kd * s.albedo * (1.0f / M_PI_F);
    return diff + spec;
}

// Target pdf for RIS: luminance(Le) * |BRDF * NdotL| * geometry, NO visibility.
// Returns 0 if the sample is back-facing on either surface.
__device__ inline float restirEvalTargetPdf(
    const ReSTIRSurface& s,
    const GPUAreaLight&  light,
    float b1, float b2)
{
    float b0 = 1.0f - b1 - b2;
    if (b0 < 0.0f || b1 < 0.0f || b2 < 0.0f) return 0.0f;
    float3 pOnLight = light.v0 + light.e1 * b1 + light.e2 * b2;
    float3 toL = pOnLight - s.position;
    float  dist2 = fmaxf(dot(toL, toL), 1e-6f);
    float  dist  = sqrtf(dist2);
    float3 L = toL * (1.0f / dist);

    float NdotL = fmaxf(dot(s.normal, L), 0.0f);
    float lightNdot = fmaxf(dot(light.normal, -L), 0.0f);
    if (NdotL <= 0.0f || lightNdot <= 0.0f) return 0.0f;

    float Lum = restirLuminance(light.emission);
    if (Lum <= 0.0f) return 0.0f;

    float3 brdf = restirEvalBrdf(s, L);
    float  fLum = restirLuminance(brdf) * NdotL;
    if (fLum <= 0.0f) return 0.0f;

    float geom = lightNdot / dist2;
    return Lum * fLum * geom;
}

// ── Reservoir primitives (Bitterli 2020 Alg. 2) ─────────────────
__device__ inline void restir_reservoirReset(ReSTIRReservoir& r) {
    r.lightIndex = 0xFFFFFFFFu;
    r.baryB1 = 0.0f;
    r.baryB2 = 0.0f;
    r.pHat   = 0.0f;
    r.W      = 0.0f;
    r.M      = 0.0f;
}

__device__ inline bool restir_reservoirUpdate(
    ReSTIRReservoir& r, float& wSum,
    uint32_t lightIdx, float b1, float b2, float pHat,
    float wCandidate, float u01)
{
    if (!(wCandidate > 0.0f)) return false;
    wSum += wCandidate;
    r.M  += 1.0f;
    if (u01 * wSum < wCandidate) {
        r.lightIndex = lightIdx;
        r.baryB1     = b1;
        r.baryB2     = b2;
        r.pHat       = pHat;
        return true;
    }
    return false;
}

__device__ inline void restir_reservoirFinalize(ReSTIRReservoir& r, float wSum) {
    if (r.lightIndex == 0xFFFFFFFFu || r.pHat <= 0.0f || r.M <= 0.0f) {
        r.W = 0.0f;
        return;
    }
    r.W = wSum / (r.M * r.pHat);
}

__device__ inline bool restir_reservoirCombine(
    ReSTIRReservoir& dst, float& wSum,
    const ReSTIRReservoir& src, float pHatAtDst, float u01)
{
    if (src.lightIndex == 0xFFFFFFFFu || src.M <= 0.0f) {
        dst.M += src.M;
        return false;
    }
    float w = pHatAtDst * src.W * src.M;
    bool accepted = false;
    if (w > 0.0f) {
        wSum += w;
        if (u01 * wSum < w) {
            dst.lightIndex = src.lightIndex;
            dst.baryB1     = src.baryB1;
            dst.baryB2     = src.baryB2;
            dst.pHat       = pHatAtDst;
            accepted = true;
        }
    }
    dst.M += src.M;
    return accepted;
}
