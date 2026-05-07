#pragma once
// Backend-agnostic NEE (next-event estimation) helpers for the path tracers.
//
// Three light types — area, point, directional — each get a helper that
// encapsulates the shadow-trace + volumetric attenuation + BRDF + accumulator
// logic shared across PathTraceKernel.cu, PathTraceKernelSplit.cu, and the
// OptiX raygens in OptiXBackend. CUDA / OptiX divergence is bridged by a
// `TraceShadowFn` template parameter (the same pattern used by
// `volumeSingleScatterInScatter` in render/VolumeNEE.cuh): callers pass a
// lambda wrapping their backend's transmissive-shadow primitive
// (`cudaTraceTransmissiveShadow` for CUDA / `traceShadowRay` for OptiX).
//
// Mono / split BRDF divergence is bridged by `BrdfFn` and `PdfFn` callables
// supplied by the kernel — `evalNEEBrdf` / `evalNEEBrdfPdf` below cover both
// (mono passes primaryLobeOverride=false; split forwards its actual flag).
//
// What stays in the kernel (intentionally):
//   - Area-light selection (BVH-vs-CDF for mono; CDF-only for the others)
//   - ReSTIR-DI activation predicate + reservoir read + restirSkip handling
//   - Final accumulation form (radiance vs pathRadiance vs emissiveContrib)
//   - End-of-frame luminance clamp policy

#include "core/Math.h"
#include "core/VolumeMedium.h"
#include "core/VolumeDevice.cuh"
#include "gpu/AreaLightGPU.h"
#include "gpu/DeviceScene.h"
#include "gpu/LightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "render/PathTraceHelpers.cuh"

// ── BRDF + PDF wrappers covering both kernels ────────────────
// Mono kernels always use the full Cook-Torrance mixture. The split kernels
// force a single lobe at the primary hit (so diffBucket * albedo + specBucket
// recovers the full radiance after demodulation) and switch back to the
// mixture for indirect bounces.
__device__ inline float3 evalNEEBrdf(
    const GPUMaterial& mat, float3 N, float3 V, float3 Ld, float3 albedo,
    bool primaryLobeOverride, int pickedBucket)
{
    if (primaryLobeOverride) {
        return (pickedBucket == 0)
            ? materialDiffuseLobe(mat, N, V, Ld, albedo)
            : materialSpecularLobe(mat, N, V, Ld, albedo);
    }
    return materialBsdfEvaluate(mat, N, V, Ld, albedo);
}

__device__ inline float evalNEEBrdfPdf(
    const GPUMaterial& mat, float3 N, float3 V, float3 Ld, float3 albedo,
    bool primaryLobeOverride, int pickedBucket)
{
    if (primaryLobeOverride) {
        float NdotL = fmaxf(dot(N, Ld), 0.0f);
        return (pickedBucket == 0)
            ? bsdfDiffusePdf(NdotL)
            : bsdfSpecularPdf(N, V, Ld, mat.roughness);
    }
    float spProb = materialSpecProb(mat, N, V, albedo);
    return materialMixturePdf(mat, N, V, Ld, spProb);
}

// ── Area-light NEE contribution (single sampled light) ───────
// Caller has already chosen the light (BVH / CDF / ReSTIR reservoir) and the
// barycentric coords. Helper does: visibility (shadow trace + volumetric
// attenuation), Le fetch, BRDF, MIS-vs-ReSTIR estimator, optional firefly
// clamp. Returns the contribution to add to the kernel's radiance accumulator
// (already multiplied by throughput).
//
//   - If `useReSTIR`: estimator is f(x) * W (no MIS — ReSTIR *is* the
//     light-side strategy at the primary hit). `pSelect` is unused;
//     `restirW` is the reservoir's contribution weight.
//   - Else: power-heuristic MIS between light sampling (using `pSelect`) and
//     BSDF sampling (queried via `brdfPdf`).
//
// `fireflyClamp` > 0 caps the per-contribution luminance (mono ReSTIR-DI uses
// 10 to kill near-grazing reservoir spikes; split uses 10 on every NEE
// contribution; mono non-ReSTIR uses 0 = no clamp, relying on the end-of-loop
// luminance clamp).
template <typename TraceShadowFn, typename BrdfFn, typename PdfFn>
__device__ inline float3 evalAreaLightNEEContribution(
    float3 throughput, float3 hitPos, float3 N,
    const GPUAreaLight& light, float b0, float b1, float b2,
    float pSelect,
    bool useReSTIR, float restirW,
    const VolumeMedium& medium, uint32_t& rng,
    TraceShadowFn traceShadow,
    BrdfFn brdfEval, PdfFn brdfPdf,
    float fireflyClamp)
{
    float3 lightV0  = light.v0;
    float3 lightV1  = light.v0 + light.e1;
    float3 lightV2  = light.v0 + light.e2;
    float3 lightPos = lightV0 * b0 + lightV1 * b1 + lightV2 * b2;

    float3 toLight = lightPos - hitPos;
    float dist2 = fmaxf(dot(toLight, toLight), 1e-6f);
    float dist  = sqrtf(dist2);
    float3 Ld   = toLight * (1.0f / dist);

    float NdotL    = fmaxf(dot(N, Ld), 0.0f);
    float lightNdot = fmaxf(dot(light.normal, -Ld), 0.0f);
    if (NdotL <= 0.0f || lightNdot <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    // Shadow ray with glass transparency. The trace callable returns
    // (1,1,1) for an unobstructed segment, the glass tint when the segment
    // crosses transmissive surfaces, and (0,0,0) on opaque occlusion.
    float3 shadowOrigin = hitPos + N * 0.001f;
    float3 st = traceShadow(shadowOrigin, Ld, dist);
    // Volumetric attenuation along the surface→area-light shadow segment.
    // Returns (1,1,1) when no medium is enabled. Always called so the RNG
    // draw count is the same whether the surface ray was occluded or not.
    st = st * volumeShadowTransmittance(shadowOrigin, Ld, dist, medium, rng);
    if (luminance(st) < 1e-6f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 brdf = brdfEval(Ld, NdotL);
    float3 Le   = sampleAreaLightLe(light, b0, b1, b2);

    float3 contrib;
    if (useReSTIR) {
        // ReSTIR estimator: f(x) * W where f is the unshadowed integrand
        // (BRDF * Le * G * NdotL) and W is the reservoir's contribution
        // weight. No MIS against BSDF sampling here — ReSTIR *is* our
        // light-side strategy at the primary hit, and mixing it with BSDF
        // MIS requires a bounded-weight variant that's beyond scope.
        float geom = lightNdot / dist2;
        contrib = throughput * st * brdf * Le * (NdotL * geom) * restirW;
    } else {
        float pTri     = pSelect;
        float pArea    = pTri / fmaxf(light.area, 1e-7f);
        float pdfOmega = pArea * dist2 / fmaxf(lightNdot, 1e-7f);
        float pdfBsdf  = brdfPdf(Ld, NdotL);
        float weight   = powerHeuristic(pdfOmega, pdfBsdf);
        contrib = throughput * st * brdf * Le *
                  (NdotL / fmaxf(pdfOmega, 1e-7f)) * weight;
    }
    // Per-frame firefly clamp on the contribution. Without this a
    // near-grazing ReSTIR sample (or any single-frame outlier on the split
    // path) produces a single-frame ~50-luminance spike that survives the
    // accumulator / NRD temporal filter for many frames as a shimmering
    // bright speck (M7 flash-and-decay / "water ripples").
    if (fireflyClamp > 0.0f) contrib = clampFirefly(contrib, fireflyClamp);
    return contrib;
}

// ── All point lights (looped) ────────────────────────────────
// Point lights are delta emitters — BSDF sampling can never hit them, so we
// always sample them light-side without MIS. Returns the unweighted sum across
// all lights (caller multiplies by throughput).
//
// `fireflyClamp` > 0 applies clampFirefly per-light contribution (split
// kernels use 10; mono uses 0 = no per-contribution clamp).
template <typename TraceShadowFn, typename BrdfFn>
__device__ inline float3 evalAllPointLightsNEE(
    const DeviceSceneData& scene, const VolumeMedium& medium,
    float3 hitPos, float3 N,
    uint32_t& rng,
    TraceShadowFn traceShadow, BrdfFn brdfEval,
    float fireflyClamp)
{
    float3 direct = make_float3(0.0f, 0.0f, 0.0f);
    for (uint32_t li = 0; li < scene.pointLightCount; li++) {
        GPUPointLight light = scene.d_pointLights[li];
        float3 toL = light.position - hitPos;
        float d2  = fmaxf(dot(toL, toL), 1e-6f);
        float d   = sqrtf(d2);
        float3 Ld = toL * (1.0f / d);
        float NdotL = fmaxf(dot(N, Ld), 0.0f);
        if (NdotL <= 0.0f) continue;

        // Shadow + volumetric attenuation along the surface→point-light segment.
        float3 shadowOrigin = hitPos + N * 0.001f;
        float3 st = traceShadow(shadowOrigin, Ld, d);
        st = st * volumeShadowTransmittance(shadowOrigin, Ld, d, medium, rng);
        if (luminance(st) < 1e-6f) continue;

        // Standard quadratic attenuation. The 1e-4 floor keeps the divisor
        // away from zero for lights placed on the surface itself.
        float attenDen = light.constantAttenuation
                       + light.linearAttenuation  * d
                       + light.quadraticAttenuation * d2;
        float atten = 1.0f / fmaxf(attenDen, 1e-4f);
        float3 Li = light.color * (light.intensity * atten);

        float3 brdf = brdfEval(Ld, NdotL);
        float3 contrib = brdf * st * Li * NdotL;
        if (fireflyClamp > 0.0f) contrib = clampFirefly(contrib, fireflyClamp);
        direct += contrib;
    }
    return direct;
}

// ── All directional lights (looped) ──────────────────────────
// Same shape as point lights but no distance attenuation, infinite shadow
// tmax. Caller multiplies the returned sum by throughput.
template <typename TraceShadowFn, typename BrdfFn>
__device__ inline float3 evalAllDirectionalLightsNEE(
    const DeviceSceneData& scene, const VolumeMedium& medium,
    float3 hitPos, float3 N,
    uint32_t& rng,
    TraceShadowFn traceShadow, BrdfFn brdfEval,
    float fireflyClamp)
{
    float3 direct = make_float3(0.0f, 0.0f, 0.0f);
    for (uint32_t li = 0; li < scene.directionalLightCount; li++) {
        GPUDirectionalLight light = scene.d_directionalLights[li];
        float3 Ld = light.direction;
        float NdotL = fmaxf(dot(N, Ld), 0.0f);
        if (NdotL <= 0.0f) continue;

        // Shadow trace with infinite tmax (sun-style directional). Volumetric
        // ratio-tracks through the medium bounding box (gracefully skips for
        // unbounded media).
        float3 shadowOrigin = hitPos + N * 0.001f;
        float3 st = traceShadow(shadowOrigin, Ld, 1e30f);
        st = st * volumeShadowTransmittance(shadowOrigin, Ld, 1e30f, medium, rng);
        if (luminance(st) < 1e-6f) continue;

        float3 brdf = brdfEval(Ld, NdotL);
        float3 contrib = brdf * st * light.color * NdotL;
        if (fireflyClamp > 0.0f) contrib = clampFirefly(contrib, fireflyClamp);
        direct += contrib;
    }
    return direct;
}

// ── Emissive-hit MIS weight (path bounce lands on a light) ───
// Computes the power-heuristic weight for an emissive-triangle hit at bounce > 0,
// MIS-ing the BSDF-sampled path against the light-sampling strategy. `pTri` is
// the probability that NEE would have selected this triangle from the previous
// surface — caller computes it (mono uses BVH PDF when available, others use
// the CDF weight ratio). Returns 1.0 if the triangle isn't visible from the
// previous surface (light face oriented away).
__device__ inline float computeEmissiveMISWeight(
    const GPUAreaLight& light,
    float3 hitPos, float3 prevSurfacePos, float prevBsdfPdf,
    float pTri)
{
    float3 toLight = hitPos - prevSurfacePos;
    float dist2 = fmaxf(dot(toLight, toLight), 1e-6f);
    float3 wi = normalize(toLight);
    float lightNdot = fmaxf(dot(light.normal, -wi), 0.0f);
    if (lightNdot <= 0.0f) return 1.0f;

    float pArea  = pTri / fmaxf(light.area, 1e-7f);
    float pLight = pArea * dist2 / fmaxf(lightNdot, 1e-7f);
    return powerHeuristic(prevBsdfPdf, pLight);
}
