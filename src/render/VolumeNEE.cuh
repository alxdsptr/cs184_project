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

#include "core/Math.h"
#include "core/VolumeMedium.h"
#include "core/VolumeDevice.cuh"
#include "gpu/DeviceScene.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/LightGPU.h"
#include "gpu/Random.h"

#ifndef M_PI_F
#  define M_PI_F 3.14159265358979323846f
#endif

namespace volume_nee_detail {

static __forceinline__ __device__ float luminance3(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// MIS power heuristic — duplicated from PathTraceHelpers.cuh / OptiXPrograms.cu
// so this header doesn't pull in either set of helpers (both define their own
// `powerHeuristic` at file scope and we'd hit ODR conflicts). Local + static
// keeps the symbol from leaking.
static __forceinline__ __device__ float powerHeuristic2(float a, float b) {
    float a2 = a * a;
    float b2 = b * b;
    return a2 / fmaxf(a2 + b2, 1e-7f);
}

}  // namespace volume_nee_detail

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
    using volume_nee_detail::luminance3;
    using volume_nee_detail::powerHeuristic2;

    float3 inScatter = make_float3(0.0f, 0.0f, 0.0f);

    // ── Area lights — picked CDF entry, MIS with phase function ────────
    if (scene.d_areaLights && scene.areaLightCount > 0 &&
        scene.d_areaLightCDF && scene.areaLightTotalWeight > 0.0f)
    {
        // Inline binary search over the area-light CDF. Mirrors
        // `sampleAreaLightIndex` in PathTraceHelpers.cuh / OptiXPrograms.cu
        // — duplicated here so this header doesn't depend on either.
        float u = pcg32_float(rng);
        uint32_t lo = 0, hi = scene.areaLightCount;
        while (lo < hi) {
            uint32_t mid = (lo + hi) >> 1;
            if (u <= scene.d_areaLightCDF[mid]) hi = mid;
            else                                lo = mid + 1;
        }
        uint32_t li = (lo >= scene.areaLightCount) ? (scene.areaLightCount - 1) : lo;

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
            if (luminance3(st) > 1e-6f) {
                float pTri    = light.weight / scene.areaLightTotalWeight;
                float pArea   = pTri / fmaxf(light.area, 1e-7f);
                float pdfOmega = pArea * d2 / fmaxf(lNdot, 1e-7f);
                float phase   = phaseHGEval(dot(wo, Ld), medium.anisotropy);
                float w       = powerHeuristic2(pdfOmega, phase);
                // Le evaluation: textured area lights need a UV lookup. The
                // texture sample lives in callers' helper headers (PathTrace
                // CUDA + OptiX both define `sampleAreaLightLe`); rather than
                // duplicate here, inline the math.
                float3 Le;
                if (light.emissiveTex == 0) {
                    Le = light.emission;
                } else {
                    float u_uv = light.uv0.x * b0 + light.uv1.x * b1 + light.uv2.x * b2;
                    float v_uv = light.uv0.y * b0 + light.uv1.y * b1 + light.uv2.y * b2;
                    float4 tx  = tex2D<float4>(light.emissiveTex, u_uv, v_uv);
                    Le = make_float3(tx.x, tx.y, tx.z) * light.emission;
                }
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
            if (luminance3(st) < 1e-6f) continue;
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
            if (luminance3(st) < 1e-6f) continue;
            float phase = phaseHGEval(dot(wo, Ld), medium.anisotropy);
            inScatter += st * light.color * phase;
        }
    }

    return inScatter;
}
