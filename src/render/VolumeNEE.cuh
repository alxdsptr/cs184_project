#pragma once
// Shared single-scatter next-event-estimation for participating media.
// Used by both the CUDA path-trace kernels (PathTraceKernel.cu /
// PathTraceKernelSplit.cu) and the OptiX raygens (OptiXPrograms.cu) so the
// volume integrator stays in one place. The shadow-ray casting is abstracted
// via a callable so each backend uses its native ray query (CUDA SAH-BVH vs
// OptiX GAS), without forcing this header to include either.
//
// The callable's contract:
//   float3 traceShadow(float3 origin, float3 dir, float dist);
// Returns RGB transmittance along [origin, origin + dir*dist] through scene
// geometry only — exactly (0,0,0) on opaque occlusion, (1,1,1) when nothing
// is hit, and a partial RGB attenuation when the ray crosses transmissive
// surfaces (glass). Volumetric (medium) transmittance is applied separately
// inside this function via volumeShadowTransmittance() so callers don't need
// to track it.

#include "core/VolumeMedium.h"
#include "core/VolumeDevice.cuh"
#include "gpu/DeviceScene.h"
#include "gpu/LightGPU.h"
#include "gpu/Random.h"
#include "render/PathTraceHelpers.cuh"   // luminance, powerHeuristic, sampleAreaLightLe, sampleAreaLightIndex

// Compute the in-scattered radiance arriving at `mediumPos` along outgoing
// direction `wo`, summing all lights with shadow-ray + volume-transmittance
// attenuation. Does NOT multiply by σ_s/σ_t (single-scatter albedo) — the
// caller multiplies by mediumSingleScatterAlbedo(medium) so the same function
// works in callers that have already folded the albedo into `throughput`.
//
// `traceShadow` is the backend-specific shadow-ray function (see top of file).
template <typename TraceShadowFn>
__device__ inline float3 volumeSingleScatterInScatter(
    const DeviceSceneData& scene,
    const VolumeMedium&    medium,
    float3                 mediumPos,
    float3                 wo,
    uint32_t&              rng,
    TraceShadowFn          traceShadow)
{
    float3 inScatter = make_float3(0.0f, 0.0f, 0.0f);

    // ── Area lights — picked CDF entry, MIS with phase function ────────
    if (scene.d_areaLights && scene.areaLightCount > 0 &&
        scene.d_areaLightCDF && scene.areaLightTotalWeight > 0.0f)
    {
        float u = pcg32_float(rng);
        uint32_t li = sampleAreaLightIndex(scene.d_areaLightCDF,
                                            scene.areaLightCount, u);

        GPUAreaLight light = scene.d_areaLights[li];
        float r1 = pcg32_float(rng), r2 = pcg32_float(rng);
        float su = sqrtf(r1);
        float b0 = 1.0f - su;
        float b1 = su * (1.0f - r2);
        float b2 = su * r2;
        float3 lp = light.v0 * b0
                  + (light.v0 + light.e1) * b1
                  + (light.v0 + light.e2) * b2;
        float3 toL = lp - mediumPos;
        float d2 = fmaxf(dot(toL, toL), 1e-6f);
        float d  = sqrtf(d2);
        float3 Ld = toL * (1.0f / d);
        float lNdot = fmaxf(dot(light.normal, -Ld), 0.0f);
        if (lNdot > 0.0f) {
            float3 st = traceShadow(mediumPos, Ld, d);
            float3 volST = volumeShadowTransmittance(mediumPos, Ld, d, medium, rng);
            st = st * volST;
            if (luminance(st) > 1e-6f) {
                float pTri    = light.weight / scene.areaLightTotalWeight;
                float pArea   = pTri / fmaxf(light.area, 1e-7f);
                float pdfOmega = pArea * d2 / fmaxf(lNdot, 1e-7f);
                float phase   = phaseHGEval(dot(wo, Ld), medium.anisotropy);
                float w       = powerHeuristic(pdfOmega, phase);
                float3 Le     = sampleAreaLightLe(light, b0, b1, b2);
                inScatter += st * Le * (phase / fmaxf(pdfOmega, 1e-7f)) * w;
            }
        }
    }

    // ── Point lights — delta emitters, no MIS ──────────────────────────
    if (scene.d_pointLights && scene.pointLightCount > 0) {
        for (uint32_t li = 0; li < scene.pointLightCount; li++) {
            GPUPointLight light = scene.d_pointLights[li];
            float3 toL = light.position - mediumPos;
            float d2 = fmaxf(dot(toL, toL), 1e-6f);
            float d  = sqrtf(d2);
            float3 Ld = toL * (1.0f / d);
            float3 st = traceShadow(mediumPos, Ld, d);
            float3 volST = volumeShadowTransmittance(mediumPos, Ld, d, medium, rng);
            st = st * volST;
            if (luminance(st) < 1e-6f) continue;
            float attenDen = light.constantAttenuation
                           + light.linearAttenuation * d
                           + light.quadraticAttenuation * d2;
            float atten = 1.0f / fmaxf(attenDen, 1e-4f);
            float3 Li = light.color * (light.intensity * atten);
            float phase = phaseHGEval(dot(wo, Ld), medium.anisotropy);
            inScatter += st * Li * phase;
        }
    }

    // ── Directional lights — delta emitters, infinite distance ─────────
    if (scene.d_directionalLights && scene.directionalLightCount > 0) {
        for (uint32_t li = 0; li < scene.directionalLightCount; li++) {
            GPUDirectionalLight light = scene.d_directionalLights[li];
            float3 Ld = light.direction;
            float3 st = traceShadow(mediumPos, Ld, 1e30f);
            // Volume transmittance over the full ray segment; ratio tracking
            // clips to the medium bounds itself.
            float3 volST = volumeShadowTransmittance(mediumPos, Ld, 1e30f, medium, rng);
            st = st * volST;
            if (luminance(st) < 1e-6f) continue;
            float phase = phaseHGEval(dot(wo, Ld), medium.anisotropy);
            inScatter += st * light.color * phase;
        }
    }

    return inScatter;
}
