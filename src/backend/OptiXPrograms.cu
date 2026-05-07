// ── OptiX device programs ─────────────────────────────────────
// Raygen: outer bounce loop + shading (ported from PathTraceKernel.cu).
// Closest-hit: records primitive index + barycentrics into payload.
// Miss (radiance): marks "no hit".
// Any-hit (shadow): accumulates glass transmittance, ignores or terminates.
// Miss (shadow): no-op (transmittance slots carry final value).

#include <optix.h>
#include <cuda_fp16.h>
#include <surface_indirect_functions.h>

#include "backend/OptiXLaunchParams.h"
#include "gpu/NRDHelpers.cuh"
#include "core/Halton.h"
#include "core/VolumeMedium.h"
#include "core/VolumeDevice.cuh"
#include "gpu/LightGPU.h"
#include "gpu/Random.h"
#include "gpu/BRDF.h"
#include "accel/LightBVHSample.h"
#include "render/PathTraceHelpers.cuh"
#include "render/GBufferWriters.cuh"
#include "render/NEEHelpers.cuh"
#include "render/Finalizers.cuh"
#include "render/ReSTIRDevice.cuh"
#include "render/ReSTIRGIDevice.cuh"
#include "render/VolumeNEE.cuh"

extern "C" __constant__ LaunchParams params;

// ── Payload packing helpers ──────────────────────────────────
static __forceinline__ __device__ uint32_t floatBitsToUint(float f) {
    return __float_as_uint(f);
}
static __forceinline__ __device__ float uintBitsToFloat(uint32_t u) {
    return __uint_as_float(u);
}

// Radiance payload (5 slots): hit, primIdx, baryU, baryV, tHit
struct RadiancePayload {
    uint32_t hit;
    uint32_t primIdx;
    float    baryU;
    float    baryV;
    float    tHit;
};

static __forceinline__ __device__ RadiancePayload traceRadianceRay(
    OptixTraversableHandle h,
    float3 origin, float3 dir, float tmin, float tmax)
{
    uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    optixTrace(
        h,
        origin, dir,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        /*SBT offset*/0,
        /*SBT stride*/2,
        /*miss SBT  */0,
        p0, p1, p2, p3, p4);
    RadiancePayload r;
    r.hit     = p0;
    r.primIdx = p1;
    r.baryU   = uintBitsToFloat(p2);
    r.baryV   = uintBitsToFloat(p3);
    r.tHit    = uintBitsToFloat(p4);
    return r;
}

// Shadow payload (4 slots): transmittance xyz + occluded flag
static __forceinline__ __device__ float3 traceShadowRay(
    OptixTraversableHandle h,
    float3 origin, float3 dir, float tmin, float tmax)
{
    uint32_t p0 = floatBitsToUint(1.0f);
    uint32_t p1 = floatBitsToUint(1.0f);
    uint32_t p2 = floatBitsToUint(1.0f);
    uint32_t p3 = 0;  // occluded flag
    optixTrace(
        h,
        origin, dir,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
        /*SBT offset*/1,
        /*SBT stride*/2,
        /*miss SBT  */1,
        p0, p1, p2, p3);
    if (p3 != 0) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(uintBitsToFloat(p0), uintBitsToFloat(p1), uintBitsToFloat(p2));
}

// ── Raygen ────────────────────────────────────────────────────
extern "C" __global__ void __raygen__path_trace()
{
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    const DeviceSceneData& scene = params.scene;
    const CameraParams&    camera = params.camera;
    uint32_t samplesPerPixel = params.spp < 1u ? 1u : params.spp;
    uint32_t maxBounces = params.maxBounces;
    bool enableEnvironment = params.enableEnvironment != 0;
    OptixTraversableHandle handle = params.handle;

    float3 radianceSum = make_float3(0, 0, 0);
    bool gbufferWritten = false;

    // DLSSOnly publishes to `gbuffer.hdrColor`; in that path the sub-pixel
    // offset must exactly match `camera.jitterOffset` (Halton). An extra
    // per-sample random offset would confuse DLSS's temporal reprojection
    // and produce ghosting / shimmer.
    const bool dlssPublish = (params.gbuffer.hdrColor != 0);

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
        // Mix frameIndex so replay (which resets sampleIndex per pose) doesn't
        // generate identical noise patterns frame-to-frame — DLSS / DLSS-RR /
        // NRD need uncorrelated noise to temporally accumulate.
        uint32_t seedSalt = params.sampleIndex + camera.frameIndex * 0x9E3779B9u;
        uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                                  seedSalt * 0x85EBCA6Bu + s);

        // ReSTIR (DI / GI / PT) reservoirs were generated against the surface
        // hit by the unjittered camera ray (only camera.jitterOffset, no
        // per-sample random sub-pixel jitter). To consume them at this pixel
        // the primary ray must hit the SAME surface — otherwise the stored
        // pHat / W is for a different shading point than the integrand we
        // evaluate below, producing strong overexposure on glossy/reflective
        // surfaces. Mirrors PathTraceKernel.cu's `restirActiveBounce0`.
        bool restirActiveBounce0 =
            (s == 0) && (
                (scene.restirEnabled   != 0 && scene.d_restirReservoirs != nullptr) ||
                (scene.restirGIEnabled != 0 && scene.d_restirGIIndirect != nullptr) ||
                (scene.restirPTEnabled != 0 && scene.d_restirPTIndirect != nullptr));

        float jx, jy;
        if (dlssPublish || restirActiveBounce0) {
            jx = camera.jitterOffset.x;
            jy = camera.jitterOffset.y;
        } else {
            jx = pcg32_float(rng) - 0.5f;
            jy = pcg32_float(rng) - 0.5f;
            jx += camera.jitterOffset.x;
            jy += camera.jitterOffset.y;
        }

        Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

        float3 throughput = make_float3(1, 1, 1);
        float3 radiance   = make_float3(0, 0, 0);
        bool firstBounce  = true;
        // True only when the previous bounce used a *delta* BSDF (perfect
        // mirror / glass refraction). See PathTraceKernel.cu for the longer
        // comment: Cook-Torrance specular is NOT delta and must not set this.
        bool lastBounceDelta = false;
        bool havePrevSurface = false;
        float3 prevSurfacePos = make_float3(0, 0, 0);
        float prevBsdfPdf = 1.0f;

        for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
            RadiancePayload rp = traceRadianceRay(
                handle, ray.origin, ray.direction, ray.tmin, ray.tmax);

            bool didHit = (rp.hit != 0);

            // ── Participating-medium integrator ──────────────────────
            // Bounded heterogeneous volume integration via delta tracking with
            // ratio tracking for through-transmittance. See core/VolumeDevice.cuh
            // for the algorithm details. Skipped entirely when the medium is
            // disabled or the majorant is zero — non-volumetric scenes pay only
            // a single branch.
            {
                float tMaxSegment = didHit ? rp.tHit : ray.tmax;
                if (scene.medium.enabled && scene.medium.majorantSigmaT > 0.0f) {
                    float tEnter, tExit;
                    if (volumeIntersect(ray.origin, ray.direction, ray.tmin, tMaxSegment,
                                        scene.medium, tEnter, tExit))
                    {
                        float tHit = 0.0f;
                        bool scattered = volumeDeltaTrack(
                            ray.origin, ray.direction, tEnter, tExit,
                            scene.medium, rng, tHit);
                        if (scattered) {
                            float3 mediumPos = ray.origin + ray.direction * tHit;
                            float3 wo = -ray.direction;
                            // σ_s/σ_t single-scatter albedo. Beer-Lambert
                            // transmittance up to tHit is handled inside the
                            // delta-tracking acceptance probability — do NOT
                            // multiply throughput by an explicit transmittance.
                            throughput = throughput * mediumSingleScatterAlbedo(scene.medium);

                            // Single-scatter NEE shared with the CUDA kernels
                            // (render/VolumeNEE.cuh). Shadow-ray adapter routes
                            // through OptiX's GAS via traceShadowRay; the anyhit
                            // program already accumulates glass tinting, so the
                            // attenuation comes back ready to multiply.
                            float3 inScatter = volumeSingleScatterInScatter(
                                scene, scene.medium, mediumPos, wo, rng,
                                [&](float3 o, float3 d, float t) {
                                    float tmax = (t >= 1e29f) ? 1e30f : fmaxf(t - 0.002f, 0.001f);
                                    return traceShadowRay(handle, o, d, 0.001f, tmax);
                                });
                            radiance += throughput * inScatter;

                            // Continue from the scatter point in a phase-sampled direction.
                            float3 newDir = phaseHGSample(wo, scene.medium.anisotropy, rng);
                            ray.origin = mediumPos;
                            ray.direction = newDir;
                            ray.tmin = 0.001f;
                            ray.tmax = 1e30f;
                            lastBounceDelta = false;
                            continue;
                        }
                        // No scatter inside the volume — apply ratio-tracked
                        // transmittance. The before/after-volume spans are
                        // vacuum, so they contribute 1.
                        float3 T = volumeRatioTrack(
                            ray.origin, ray.direction, tEnter, tExit,
                            scene.medium, rng);
                        throughput = throughput * T;
                    }
                }
            }

            if (!didHit) {
                if (enableEnvironment) {
                    bool shForThisBounce = (bounce > 0) && !lastBounceDelta;
                    float3 envColor = sampleEnvironmentForBounce(
                        ray.direction, scene.envMapTex,
                        scene.d_shEnvCoeffs, scene.envUseSH != 0,
                        !shForThisBounce);
                    envColor = clampEnvLuminance(envColor, 100.0f);
                    radiance += throughput * envColor;
                }
                // Sky pixel: sentinel g-buffer so NRD treats it as sky (viewZ
                // beyond denoising range) and DLSS sees uniform far depth.
                // Motion vector stays 0 — sky doesn't reproject by camera.
                if (firstBounce && !gbufferWritten) {
                    writeSkyGBufferSentinel(
                        params.gbuffer.viewZ, params.gbuffer.motionVectors,
                        params.gbuffer.ndcDepth, x, y);
                    gbufferWritten = true;
                    firstBounce = false;
                }
                break;
            }

            // Reconstruct hit record from payload + scene buffers.
            HitRecord hit;
            hit.t             = rp.tHit;
            hit.primitiveIndex = (int)rp.primIdx;
            float baryU = rp.baryU;
            float baryV = rp.baryV;
            float baryW = 1.0f - baryU - baryV;

            uint32_t triIdx = rp.primIdx;
            uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
            uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
            uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
            float3 v0 = scene.d_positions[i0];
            float3 v1 = scene.d_positions[i1];
            float3 v2 = scene.d_positions[i2];
            hit.position = v0 * baryW + v1 * baryU + v2 * baryV;

            float3 geomN = normalize(cross(v1 - v0, v2 - v0));
            hit.shadingNormal = geomN;
            hit.normal        = geomN;
            hit.frontFace = (dot(ray.direction, geomN) < 0.0f);
            hit.uv = make_float2(baryU, baryV);
            hit.materialIndex = scene.d_materialIndices ? scene.d_materialIndices[triIdx] : -1;

            GPUMaterial mat;
            if (hit.materialIndex >= 0 && (uint32_t)hit.materialIndex < scene.materialCount)
                mat = scene.d_materials[hit.materialIndex];
            else {
                mat.albedo = make_float3(0.8f, 0.2f, 0.8f);
                mat.roughness = 0.5f;
                mat.metallic = 0.0f;
                mat.emission = make_float3(0,0,0);
                mat.emissionStrength = 0.0f;
                mat.ior = 1.5f;
                mat.transmission = 0.0f;
                mat.albedoTex = 0;
                mat.metallicRoughTex = 0;
                mat.emissiveTex = 0;
                mat.normalTex = 0;
                mat.specularGlossTex = 0;
                mat.useSpecularGlossiness = 0;
                mat.specularGlossAlphaIsGlossiness = 0;
                mat.useFBXCustomPacking = 0;
                mat.useFBXUEPacking = 0;
                mat.specularColor = make_float3(1.0f, 1.0f, 1.0f);
                mat.glossiness = 0.5f;
                mat.pureDiffuse = 0;
            }

            float2 texUV = make_float2(0.0f, 0.0f);
            if (scene.d_uvs) {
                float2 uv0 = scene.d_uvs[i0];
                float2 uv1 = scene.d_uvs[i1];
                float2 uv2 = scene.d_uvs[i2];
                texUV = uv0 * baryW + uv1 * baryU + uv2 * baryV;
            }

            float3 albedo;
            float3 emissiveColor;
            applyMaterialTextures(mat, texUV, albedo, emissiveColor);

            float3 N = hit.shadingNormal;
            if (scene.d_normals) {
                float3 n0 = scene.d_normals[i0];
                float3 n1 = scene.d_normals[i1];
                float3 n2 = scene.d_normals[i2];
                N = normalize(n0 * baryW + n1 * baryU + n2 * baryV);
            }
            if (mat.transmission <= 0.0f && mat.normalTex != 0 && scene.d_tangents) {
                N = applyInterpolatedNormalMap(N, scene, i0, i1, i2, baryU, baryV, baryW,
                                               mat.normalTex, texUV);
            }
            if (mat.transmission <= 0.0f) {
                if (dot(N, ray.direction) > 0) N = -N;
            }

            if (firstBounce) {
                if (!gbufferWritten) {
                    // Previous-frame world-space hit position. For static
                    // geometry this == hit.position, so reprojecting through
                    // prevViewProjMatrix captures camera motion only. For
                    // animated geometry we interpolate the previous frame's
                    // posed vertex positions with the same barycentrics, so
                    // the motion vector tracks the mesh's movement too —
                    // critical for DLSS/NRD temporal reuse on moving objects.
                    float3 hitPosPrev = hit.position;
                    if (scene.d_positionsPrev) {
                        float3 v0p = scene.d_positionsPrev[i0];
                        float3 v1p = scene.d_positionsPrev[i1];
                        float3 v2p = scene.d_positionsPrev[i2];
                        hitPosPrev = v0p * baryW + v1p * baryU + v2p * baryV;
                    }
                    float  viewZprim, clipCurrZ;
                    float2 mvPx;
                    computePrimaryReproject(camera, hit.position, hitPosPrev,
                                            params.width, params.height,
                                            viewZprim, mvPx, clipCurrZ);

                    if (params.aux.d_linearDepth)   params.aux.d_linearDepth[pixelIdx] = viewZprim;
                    if (params.aux.d_albedo)        params.aux.d_albedo[pixelIdx]      = albedo;
                    if (params.aux.d_normal)        params.aux.d_normal[pixelIdx]      = N;
                    if (params.aux.d_motionVectors) params.aux.d_motionVectors[pixelIdx] = mvPx;

                    // DLSSOnly / NRD: also write to Vulkan-shared surfaces so
                    // post-processing can read them as VkImages directly. Only
                    // the first-sample primary hit wins (averaging across SPP
                    // would soften silhouettes and break temporal reprojection).
                    writePrimaryGBufferSurfaces(
                        params.gbuffer.viewZ, params.gbuffer.motionVectors,
                        params.gbuffer.ndcDepth,
                        x, y, viewZprim, mvPx, clipCurrZ);

                    gbufferWritten = true;
                }
                firstBounce = false;
            }

            // Glass / transmissive
            if (mat.transmission > 0.0f) {
                GlassBounce gb = sampleGlassBounce(
                    ray.direction, hit.position, N,
                    hit.frontFace, mat.ior, albedo, rng);
                throughput      = throughput * gb.throughputMul;
                ray.origin      = gb.newOrigin;
                ray.direction   = gb.newDir;
                ray.tmin        = 0.001f;
                ray.tmax        = 1e30f;
                lastBounceDelta = true;
                prevSurfacePos  = hit.position;
                prevBsdfPdf     = 1.0f;
                havePrevSurface = true;

                if (dot(N, ray.direction) > 0) N = -N;
                if (bounce >= 6) {
                    if (pcg32_float(rng) > 0.9f) break;
                }
                continue;
            }

            bool isEmissive = mat.emissionStrength > 0.0f &&
                              (emissiveColor.x > 0.0f || emissiveColor.y > 0.0f || emissiveColor.z > 0.0f);
            if (isEmissive) {
                float3 Le = emissiveColor * mat.emissionStrength;
                float weight = 1.0f;
                // MIS the emissive hit against the light-sampling strategy.
                // OptiX mono uses CDF-only pTri (the light BVH path is mono
                // CUDA only — see PathTraceKernel.cu).
                if (bounce > 0 && havePrevSurface && !lastBounceDelta && scene.d_triangleAreaLightIndex) {
                    int areaLightIndex = scene.d_triangleAreaLightIndex[(uint32_t)hit.primitiveIndex];
                    if (areaLightIndex >= 0 && scene.d_areaLights && scene.areaLightCount > 0) {
                        GPUAreaLight light = scene.d_areaLights[areaLightIndex];
                        float pTri = light.weight / fmaxf(scene.areaLightTotalWeight, 1e-7f);
                        weight = computeEmissiveMISWeight(
                            light, hit.position, prevSurfacePos, prevBsdfPdf, pTri);
                    }
                }
                radiance += throughput * Le * weight;
                if (mat.emissiveTex != 0) {
                    // fall through
                } else {
                    break;
                }
            }

            // NEE callables — defined once per bounce, used by every light type.
            // OptiX traceShadow wraps the GAS / anyhit pair to match the
            // signature of the CUDA `cudaTraceTransmissiveShadow` adapter.
            // Mono kernels always use the full mixture (primaryLobeOverride=false).
            float3 V = -ray.direction;
            auto traceShadow = [&](float3 o, float3 d, float dist) {
                float tmax = (dist >= 1e29f) ? 1e30f : fmaxf(dist - 0.002f, 0.001f);
                return traceShadowRay(handle, o, d, 0.001f, tmax);
            };
            auto neeBrdf = [&](float3 Ld, float NdotL) {
                return evalNEEBrdf(mat, N, V, Ld, albedo, /*primaryLobeOverride=*/false, 0);
            };
            auto neePdf = [&](float3 Ld, float NdotL) {
                return evalNEEBrdfPdf(mat, N, V, Ld, albedo, /*primaryLobeOverride=*/false, 0);
            };

            // Direct lighting NEE
            if (scene.d_areaLights && scene.areaLightCount > 0 &&
                scene.d_areaLightCDF && scene.areaLightTotalWeight > 0.0f) {
                // ReSTIR DI is applied at the primary hit (bounce 0) only —
                // the reservoir buffer is populated for camera rays only.
                bool restirActive = (scene.restirEnabled != 0) &&
                                    (scene.d_restirReservoirs != nullptr) &&
                                    (bounce == 0) && (s == 0);
                bool restirSkip = false;
                uint32_t lightIndex = 0;
                float    b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
                float    restirW = 0.0f;
                if (restirActive) {
                    const ReSTIRReservoir* res =
                        reinterpret_cast<const ReSTIRReservoir*>(scene.d_restirReservoirs);
                    ReSTIRReservoir r = res[pixelIdx];
                    if (r.lightIndex == 0xFFFFFFFFu || r.W <= 0.0f || r.pHat <= 0.0f) {
                        // Empty reservoir (e.g. surface-culled) → skip NEE at
                        // this bounce. Indirect via BSDF / GI/PT consumption
                        // below still works, so the image is unbiased on
                        // average.
                        restirSkip = true;
                    } else {
                        lightIndex = r.lightIndex;
                        b1 = r.baryB1;
                        b2 = r.baryB2;
                        b0 = 1.0f - b1 - b2;
                        restirW = r.W;
                    }
                } else {
                    lightIndex = sampleAreaLightIndex(
                        scene.d_areaLightCDF, scene.areaLightCount,
                        pcg32_float(rng));

                    float r1 = pcg32_float(rng);
                    float r2 = pcg32_float(rng);
                    float su = sqrtf(r1);
                    b0 = 1.0f - su;
                    b1 = su * (1.0f - r2);
                    b2 = su * r2;
                }

                if (!restirSkip) {
                GPUAreaLight light = scene.d_areaLights[lightIndex];
                float pSelect = light.weight / fmaxf(scene.areaLightTotalWeight, 1e-7f);
                // Mono ReSTIR-DI: cap=10 firefly clamp on the f*W estimator
                // (matches PathTraceKernel.cu / Split — M7 flash-and-decay
                // protection). Mono non-ReSTIR: no per-contribution clamp;
                // the end-of-loop luminance clamp catches outliers.
                float fireflyClamp = restirActive ? 10.0f : 0.0f;
                radiance += evalAreaLightNEEContribution(
                    throughput, hit.position, N, light, b0, b1, b2, pSelect,
                    restirActive, restirW,
                    scene.medium, rng, traceShadow, neeBrdf, neePdf, fireflyClamp);
                } // end !restirSkip
            }

            // Point lights are delta emitters — always sampled, regardless of
            // whether area lights also exist. See PathTraceKernel.cu comment.
            else if (scene.d_pointLights && scene.pointLightCount > 0) {
                float3 direct = evalAllPointLightsNEE(
                    scene, scene.medium, hit.position, N, rng,
                    traceShadow, neeBrdf, /*fireflyClamp=*/0.0f);
                radiance += throughput * direct;
            }

            if (scene.d_directionalLights && scene.directionalLightCount > 0) {
                float3 direct = evalAllDirectionalLightsNEE(
                    scene, scene.medium, hit.position, N, rng,
                    traceShadow, neeBrdf, /*fireflyClamp=*/0.0f);
                radiance += throughput * direct;
            }

            // ReSTIR PT / GI consumption at the primary hit on sample s==0.
            // PT takes precedence (its postfix already contains GI's 1-bounce
            // NEE plus k more bounces' worth of light transport). Either branch
            // adds the pre-computed indirect estimate and skips continuation
            // bounces; the direct lighting at the primary hit was already
            // added above. Restricted to bounce==0 because the PT/GI buffer is
            // populated for camera rays only — higher bounces fall through to
            // plain BSDF sampling.
            if (scene.restirPTEnabled != 0 && scene.d_restirPTIndirect != nullptr &&
                bounce == 0 && s == 0)
            {
                float3 indirect = scene.d_restirPTIndirect[pixelIdx];
                radiance += throughput * indirect;
                break;
            }
            if (scene.restirGIEnabled != 0 && scene.d_restirGIIndirect != nullptr &&
                bounce == 0 && s == 0)
            {
                float3 indirect = scene.d_restirGIIndirect[pixelIdx];
                radiance += throughput * indirect;
                break;
            }

            // BRDF sampling — V was hoisted earlier for NEE.
            float specProb = materialSpecProb(mat, N, V, albedo);
            float3 newDir;

            if (pcg32_float(rng) < specProb) {
                float a = mat.roughness * mat.roughness;
                float u1 = pcg32_float(rng);
                float u2 = pcg32_float(rng);
                float cosTheta = sqrtf((1.0f - u1) / (1.0f + (a*a - 1.0f) * u1 + 1e-7f));
                float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
                float phi = 2.0f * M_PI_F * u2;
                float3 localH = make_float3(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi));
                float3 T, B;
                buildONB(N, T, B);
                float3 H = localToWorld(localH, T, N, B);
                newDir = ray.direction - H * (2.0f * dot(ray.direction, H));
                newDir = normalize(newDir);
                // Cook-Torrance specular lobe is NOT delta — MIS is valid.
                lastBounceDelta = false;
            } else {
                float u1 = pcg32_float(rng);
                float u2 = pcg32_float(rng);
                float dummyPdf;
                float3 localDir = sampleCosineHemisphere(u1, u2, dummyPdf);
                float3 T, B;
                buildONB(N, T, B);
                newDir = localToWorld(localDir, T, N, B);
                lastBounceDelta = false;
            }

            float NdotL_new = dot(N, newDir);
            if (NdotL_new < 1e-6f) break;

            float pdf = materialMixturePdf(mat, N, V, newDir, specProb);
            if (pdf < 1e-7f) break;

            float3 brdf = materialBsdfEvaluate(mat, N, V, newDir, albedo);
            throughput = throughput * brdf * (NdotL_new / (pdf + 1e-7f));

            prevSurfacePos = hit.position;
            prevBsdfPdf = pdf;
            havePrevSurface = true;

            if (bounce >= 2) {
                float lum = 0.2126f * throughput.x + 0.7152f * throughput.y + 0.0722f * throughput.z;
                float p = fminf(fmaxf(lum, 0.05f), 0.95f);
                if (pcg32_float(rng) >= p) break;
                throughput = throughput * (1.0f / p);
            }

            ray.origin    = hit.position + N * 0.001f;
            ray.direction = newDir;
            ray.tmin      = 0.001f;
            ray.tmax      = 1e30f;
        }

        monoAccumulateSppSample(radiance, radianceSum);
    }

    monoFinalizePixel(radianceSum,
                      params.accum, params.output, params.gbuffer.hdrColor,
                      pixelIdx, x, y, params.sampleIndex, samplesPerPixel);
}

// ── Miss: radiance ────────────────────────────────────────────
extern "C" __global__ void __miss__radiance()
{
    optixSetPayload_0(0);  // hit = false
}

// ── Closest-hit: radiance ─────────────────────────────────────
extern "C" __global__ void __closesthit__radiance()
{
    float2 bary = optixGetTriangleBarycentrics();
    optixSetPayload_0(1);
    optixSetPayload_1(optixGetPrimitiveIndex());
    optixSetPayload_2(__float_as_uint(bary.x));
    optixSetPayload_3(__float_as_uint(bary.y));
    optixSetPayload_4(__float_as_uint(optixGetRayTmax()));
}

// ── Miss: shadow (no-op; leaves transmittance untouched) ──────
extern "C" __global__ void __miss__shadow()
{
    // Unoccluded miss — payload already holds accumulated transmittance.
}

// ── Any-hit: shadow (glass-transparent occlusion) ─────────────
extern "C" __global__ void __anyhit__shadow()
{
    uint32_t primIdx = optixGetPrimitiveIndex();
    const DeviceSceneData& scene = params.scene;

    int matIdx = scene.d_materialIndices ? scene.d_materialIndices[primIdx] : -1;
    if (matIdx < 0 || (uint32_t)matIdx >= scene.materialCount) {
        // Unknown material: treat as opaque.
        optixSetPayload_3(1);
        optixTerminateRay();
        return;
    }
    GPUMaterial mat = scene.d_materials[matIdx];
    if (mat.transmission > 0.0f) {
        // Attenuate by albedo for colored glass; near-white treated as transparent.
        float tx = __uint_as_float(optixGetPayload_0());
        float ty = __uint_as_float(optixGetPayload_1());
        float tz = __uint_as_float(optixGetPayload_2());
        float albLum = 0.2126f * mat.albedo.x + 0.7152f * mat.albedo.y + 0.0722f * mat.albedo.z;
        if (albLum < 0.9f) {
            tx *= mat.albedo.x;
            ty *= mat.albedo.y;
            tz *= mat.albedo.z;
        }
        optixSetPayload_0(__float_as_uint(tx));
        optixSetPayload_1(__float_as_uint(ty));
        optixSetPayload_2(__float_as_uint(tz));
        optixIgnoreIntersection();
        return;
    }
    // Opaque hit: terminate.
    optixSetPayload_3(1);
    optixTerminateRay();
}

// ─────────────────────────────────────────────────────────────────────────────
// Split-output raygen (NRD).
// Mirrors PathTraceKernelSplit.cu's algorithm but uses optixTrace + the same
// hit/miss/anyhit programs as the regular raygen above. Writes 7 Vulkan-shared
// surfaces (diff/spec radiance + g-buffer) instead of an HDR accum buffer.
//
// Per-pixel SPP averaging happens in this raygen (NRD wants the mean, and
// averaging cuts single-sample bucket spikes that read as ripples after
// temporal filtering). Each spp-loop iteration does one full path.
// ─────────────────────────────────────────────────────────────────────────────

extern "C" __global__ void __raygen__path_trace_split()
{
    uint3 idx = optixGetLaunchIndex();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    const DeviceSceneData& scene  = params.scene;
    const CameraParams&    camera = params.camera;
    uint32_t samplesPerPixel = params.spp < 1u ? 1u : params.spp;
    uint32_t maxBounces      = params.maxBounces;
    bool enableEnvironment   = params.enableEnvironment != 0;
    OptixTraversableHandle handle = params.handle;

    // Per-pixel running state across spp samples (demod buckets, primary
    // g-buffer, DLSS-RR primary hit pos / ray dir / metallic for the post-spp
    // mirror-ray spec hitT trace and metallic-aware spec-albedo F0).
    SplitAccumState acc{};

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
        // Mix frameIndex so replay (which resets sampleIndex per pose) doesn't
        // generate identical noise patterns frame-to-frame — DLSS-RR / NRD
        // need uncorrelated noise to temporally accumulate.
        uint32_t seedSalt = params.sampleIndex + camera.frameIndex * 0x9E3779B9u;
        uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                                  seedSalt * 0x85EBCA6Bu + s);

        // NRD/DLSS require the ray-gen sub-pixel offset to exactly match
        // `camera.jitterOffset` (Halton). See PathTraceKernelSplit.cu for
        // the longer comment — an extra per-sample random offset makes
        // history reprojection chase the wrong sub-pixel and shows up as
        // still-frame "water wave" jitter.
        float jx = camera.jitterOffset.x;
        float jy = camera.jitterOffset.y;

        Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

        float3 throughput      = make_float3(1, 1, 1);
        float3 pathRadiance    = make_float3(0, 0, 0);
        float3 emissiveContrib = make_float3(0, 0, 0);

        // Per-sample primary-hit state for bucket classification.
        // `pg` mirrors what we'll capture into `acc.primary` on the first hit.
        bool haveGbuffer = false;
        PrimaryGBuffer pg{};
        int    pickedBucket     = 0;       // 0 = diff, 1 = spec
        float  bucketHitDist    = 0.0f;
        bool   bucketHitDistSet = false;

        bool firstBounce      = true;
        bool lastBounceDelta  = false;
        bool havePrevSurface  = false;
        float3 prevSurfacePos = make_float3(0, 0, 0);
        float prevBsdfPdf     = 1.0f;

        for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
            // Whether the primary lobe override is active this iteration.
            bool primaryLobeOverride = false;

            RadiancePayload rp = traceRadianceRay(
                handle, ray.origin, ray.direction, ray.tmin, ray.tmax);
            bool didHit = (rp.hit != 0);

            // ── Participating-medium integrator (NRD/DLSS-RR-compatible) ──
            // See PathTraceKernelSplit.cu for the bucket-routing rationale:
            // single-scatter NEE goes into the emissive bucket so NRD's
            // diff/spec demodulation invariant holds, and DLSS-RR sees the
            // in-scatter via its noisy-color input. Scatter terminates the
            // path (single-scatter only).
            {
                float segmentDistance = didHit ? rp.tHit : ray.tmax;
                if (scene.medium.enabled && scene.medium.majorantSigmaT > 0.0f &&
                    segmentDistance > 0.0f)
                {
                    float tEnter, tExit;
                    if (volumeIntersect(ray.origin, ray.direction, ray.tmin, segmentDistance,
                                        scene.medium, tEnter, tExit))
                    {
                        float tScatter;
                        bool scattered = volumeDeltaTrack(
                            ray.origin, ray.direction, tEnter, tExit,
                            scene.medium, rng, tScatter);
                        if (scattered) {
                            float3 mediumPos = ray.origin + ray.direction * tScatter;
                            float3 wo = -ray.direction;
                            float3 ssAlbedo = mediumSingleScatterAlbedo(scene.medium);
                            // Single-scatter NEE shared with the CUDA kernels
                            // (render/VolumeNEE.cuh). Shadow ray uses OptiX's
                            // GAS via traceShadowRay; the anyhit program handles
                            // glass tinting so the returned attenuation is ready
                            // to use.
                            float3 inScatter = volumeSingleScatterInScatter(
                                scene, scene.medium, mediumPos, wo, rng,
                                [&](float3 o, float3 d, float t) {
                                    float tmax = (t >= 1e29f) ? 1e30f : fmaxf(t - 0.002f, 0.001f);
                                    return traceShadowRay(handle, o, d, 0.001f, tmax);
                                });

                            // Route into the emissive bucket. Path terminates.
                            float3 contrib = throughput * ssAlbedo * inScatter;
                            emissiveContrib += clampFirefly(contrib, 10.0f);
                            break;
                        }
                        // No scatter — ratio-track transmittance for surface span.
                        float3 T = volumeRatioTrack(
                            ray.origin, ray.direction, tEnter, tExit,
                            scene.medium, rng);
                        throughput = throughput * T;
                    }
                }
            }

            if (!didHit) {
                if (enableEnvironment) {
                    bool shForThisBounce = (bounce > 0) && !lastBounceDelta;
                    float3 envColor = sampleEnvironmentForBounce(
                        ray.direction, scene.envMapTex,
                        scene.d_shEnvCoeffs, scene.envUseSH != 0,
                        !shForThisBounce);
                    envColor = clampEnvLuminance(envColor, 20.0f);
                    if (bounce == 0) {
                        // Primary-ray miss: no surface → no diff/spec bucket
                        // to demodulate into. Route the sky through the
                        // emissive channel so it bypasses NRD denoise and
                        // the composite shader picks it up via `+ emis`.
                        emissiveContrib = envColor;
                    } else {
                        pathRadiance += clampFirefly(throughput * envColor, 10.0f);
                    }
                }
                // Sky pixel sentinel viewZ — only the first sample's miss wins.
                if (firstBounce && !haveGbuffer) {
                    pg.viewZ = 1.0e6f;
                    pg.mvPx  = make_float2(0.0f, 0.0f);
                    // Don't set haveGbuffer — sky pixels don't contribute to
                    // diff/spec bucket and must not trigger the demodulation
                    // path below. We still want a sentinel viewZ written, so
                    // capture it via outPrimary fields directly.
                    if (!acc.gbufferWritten) {
                        acc.primary.viewZ = pg.viewZ;
                        acc.primary.mvPx  = pg.mvPx;
                        // Leave normal/roughness/albedo at their defaults.
                    }
                }
                break;
            }

            // Reconstruct hit record from payload + scene buffers.
            HitRecord hit;
            hit.t              = rp.tHit;
            hit.primitiveIndex = (int)rp.primIdx;
            float baryU = rp.baryU;
            float baryV = rp.baryV;
            float baryW = 1.0f - baryU - baryV;

            uint32_t triIdx = rp.primIdx;
            uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
            uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
            uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
            float3 v0 = scene.d_positions[i0];
            float3 v1 = scene.d_positions[i1];
            float3 v2 = scene.d_positions[i2];
            hit.position = v0 * baryW + v1 * baryU + v2 * baryV;

            float3 geomN = normalize(cross(v1 - v0, v2 - v0));
            hit.shadingNormal = geomN;
            hit.normal        = geomN;
            hit.frontFace = (dot(ray.direction, geomN) < 0.0f);
            hit.uv = make_float2(baryU, baryV);
            hit.materialIndex = scene.d_materialIndices ? scene.d_materialIndices[triIdx] : -1;

            GPUMaterial mat;
            if (hit.materialIndex >= 0 && (uint32_t)hit.materialIndex < scene.materialCount)
                mat = scene.d_materials[hit.materialIndex];
            else {
                mat.albedo = make_float3(0.8f, 0.2f, 0.8f);
                mat.roughness = 0.5f; mat.metallic = 0.0f;
                mat.emission = make_float3(0,0,0); mat.emissionStrength = 0.0f;
                mat.ior = 1.5f; mat.transmission = 0.0f;
                mat.albedoTex = 0; mat.metallicRoughTex = 0;
                mat.emissiveTex = 0; mat.normalTex = 0; mat.specularGlossTex = 0;
                mat.useSpecularGlossiness = 0; mat.specularGlossAlphaIsGlossiness = 0;
                mat.useFBXCustomPacking = 0; mat.useFBXUEPacking = 0;
                mat.specularColor = make_float3(1.0f, 1.0f, 1.0f);
                mat.glossiness = 0.5f; mat.pureDiffuse = 0;
            }

            float2 texUV = make_float2(0.0f, 0.0f);
            if (scene.d_uvs) {
                float2 uv0 = scene.d_uvs[i0];
                float2 uv1 = scene.d_uvs[i1];
                float2 uv2 = scene.d_uvs[i2];
                texUV = uv0 * baryW + uv1 * baryU + uv2 * baryV;
            }

            float3 albedo;
            float3 emissiveColor;
            applyMaterialTextures(mat, texUV, albedo, emissiveColor);

            float3 N = hit.shadingNormal;
            if (scene.d_normals) {
                float3 n0 = scene.d_normals[i0];
                float3 n1 = scene.d_normals[i1];
                float3 n2 = scene.d_normals[i2];
                N = normalize(n0 * baryW + n1 * baryU + n2 * baryV);
            }
            if (mat.transmission <= 0.0f && mat.normalTex != 0 && scene.d_tangents) {
                N = applyInterpolatedNormalMap(N, scene, i0, i1, i2, baryU, baryV, baryW,
                                               mat.normalTex, texUV);
            }
            if (mat.transmission <= 0.0f) {
                if (dot(N, ray.direction) > 0) N = -N;
            }

            // Primary-hit g-buffer capture + bucket classification.
            if (firstBounce) {
                pg.albedo    = albedo;
                pg.normal    = N;
                pg.roughness = mat.roughness;
                pg.hitPos    = hit.position;
                pg.rayDir    = ray.direction;
                pg.metallic  = mat.metallic;
                pg.viewZ     = nrd_helpers::computeViewZ(hit.position, camera.position, camera.forward);
                // Animated-geometry-aware motion vector — see comment at the
                // matching site in __raygen__path_trace.
                if (scene.d_positionsPrev) {
                    float3 v0p = scene.d_positionsPrev[i0];
                    float3 v1p = scene.d_positionsPrev[i1];
                    float3 v2p = scene.d_positionsPrev[i2];
                    float3 hitPosPrev = v0p * baryW + v1p * baryU + v2p * baryV;
                    pg.mvPx = nrd_helpers::computeMotionVectorPxAnimated(
                        hit.position, hitPosPrev,
                        camera.viewProjMatrix, camera.prevViewProjMatrix,
                        params.width, params.height);
                } else {
                    pg.mvPx = nrd_helpers::computeMotionVectorPx(
                        hit.position, camera.viewProjMatrix, camera.prevViewProjMatrix,
                        params.width, params.height);
                }
                {
                    float3 ndc = mat4_transformPoint(camera.viewProjMatrix, hit.position);
                    pg.ndcZ = clampf(ndc.z * 0.5f + 0.5f, 0.0f, 1.0f);
                }

                float3 V = -ray.direction;
                float specProb = materialSpecProb(mat, N, V, albedo);
                pickedBucket = (pcg32_float(rng) < specProb) ? 1 : 0;
                float pickedP = (pickedBucket == 1) ? specProb : (1.0f - specProb);
                throughput = throughput * (1.0f / fmaxf(pickedP, 1e-4f));

                haveGbuffer = true;
                firstBounce = false;
                primaryLobeOverride = true;
            }

            // Glass / transmissive (delta BSDF — never affects bucket classification).
            if (mat.transmission > 0.0f) {
                GlassBounce gb = sampleGlassBounce(
                    ray.direction, hit.position, N,
                    hit.frontFace, mat.ior, albedo, rng);
                throughput      = throughput * gb.throughputMul;
                ray.origin      = gb.newOrigin;
                ray.direction   = gb.newDir;
                ray.tmin = 0.001f; ray.tmax = 1e30f;
                lastBounceDelta = true;
                prevSurfacePos  = hit.position; prevBsdfPdf = 1.0f; havePrevSurface = true;
                if (bounce >= 6 && pcg32_float(rng) > 0.9f) break;
                continue;
            }

            // First indirect surface distance (drives RELAX spatial filter radius).
            if (!bucketHitDistSet && bounce == 1) {
                bucketHitDist = hit.t;
                bucketHitDistSet = true;
            }

            bool isEmissive = mat.emissionStrength > 0.0f &&
                (emissiveColor.x > 0.0f || emissiveColor.y > 0.0f || emissiveColor.z > 0.0f);
            if (isEmissive) {
                float3 Le = emissiveColor * mat.emissionStrength;
                float weight = 1.0f;
                // MIS the emissive hit against light-sampling (CDF pTri).
                if (bounce > 0 && havePrevSurface && !lastBounceDelta && scene.d_triangleAreaLightIndex) {
                    int ali = scene.d_triangleAreaLightIndex[(uint32_t)hit.primitiveIndex];
                    if (ali >= 0 && scene.d_areaLights && scene.areaLightCount > 0) {
                        GPUAreaLight light = scene.d_areaLights[ali];
                        float pTri = light.weight / fmaxf(scene.areaLightTotalWeight, 1e-7f);
                        weight = computeEmissiveMISWeight(
                            light, hit.position, prevSurfacePos, prevBsdfPdf, pTri);
                    }
                }
                if (bounce == 0) {
                    // Primary emissive routes to the non-denoised emissive
                    // bucket (NRD's diff/spec demodulation has no surface to
                    // demodulate against). Composite adds it as `+ emis`.
                    emissiveContrib = Le * weight;
                } else {
                    pathRadiance += clampFirefly(throughput * Le * weight, 10.0f);
                }
                if (mat.emissiveTex == 0) break;
            }

            // NEE callables — defined once per bounce, used by every light type.
            // Split kernels respect the primary-hit lobe override so the demod
            // buckets stay invariant; indirect bounces fall back to the mixture.
            float3 V = -ray.direction;
            auto traceShadow = [&](float3 o, float3 d, float dist) {
                float tmax = (dist >= 1e29f) ? 1e30f : fmaxf(dist - 0.002f, 0.001f);
                return traceShadowRay(handle, o, d, 0.001f, tmax);
            };
            auto neeBrdf = [&](float3 Ld, float NdotL) {
                return evalNEEBrdf(mat, N, V, Ld, albedo, primaryLobeOverride, pickedBucket);
            };
            auto neePdf = [&](float3 Ld, float NdotL) {
                return evalNEEBrdfPdf(mat, N, V, Ld, albedo, primaryLobeOverride, pickedBucket);
            };

            // NEE area lights.  ReSTIR DI replaces the per-frame CDF pick at
            // the primary hit (s==0, bounce==0) with the resampled
            // reservoir's selected sample. The lobe-override still applies
            // — ReSTIR DI feeds the chosen bucket only, multiplied by
            // 1/pickedP via `throughput`. Estimator: f * W (no MIS).
            bool restirActive = primaryLobeOverride && (s == 0) &&
                                (scene.restirEnabled != 0) &&
                                (scene.d_restirReservoirs != nullptr);
            if (scene.d_areaLights && scene.areaLightCount > 0 &&
                scene.d_areaLightCDF && scene.areaLightTotalWeight > 0.0f)
            {
                uint32_t li;
                float b0, b1, b2;
                float restirW = 0.0f;
                bool restirSkip = false;
                if (restirActive) {
                    const ReSTIRReservoir* res =
                        reinterpret_cast<const ReSTIRReservoir*>(scene.d_restirReservoirs);
                    ReSTIRReservoir r = res[pixelIdx];
                    if (r.lightIndex == 0xFFFFFFFFu || r.W <= 0.0f || r.pHat <= 0.0f) {
                        restirSkip = true;
                    } else {
                        li = r.lightIndex;
                        b1 = r.baryB1;
                        b2 = r.baryB2;
                        b0 = 1.0f - b1 - b2;
                        restirW = r.W;
                    }
                } else {
                    li = sampleAreaLightIndex(scene.d_areaLightCDF, scene.areaLightCount,
                                              pcg32_float(rng));
                    float r1 = pcg32_float(rng), r2 = pcg32_float(rng);
                    float su = sqrtf(r1);
                    b0 = 1.0f - su;
                    b1 = su * (1.0f - r2);
                    b2 = su * r2;
                }
                if (!restirSkip) {
                GPUAreaLight light = scene.d_areaLights[li];
                float pSelect = light.weight / fmaxf(scene.areaLightTotalWeight, 1e-7f);
                // Split kernel always clamps NEE contributions at cap=10 — RELAX
                // is sensitive to single-sample spikes (water-ripple artifact).
                pathRadiance += evalAreaLightNEEContribution(
                    throughput, hit.position, N, light, b0, b1, b2, pSelect,
                    restirActive, restirW,
                    scene.medium, rng, traceShadow, neeBrdf, neePdf,
                    /*fireflyClamp=*/10.0f);
                } // end !restirSkip
            }

            // Point lights: only sampled when no area lights (matches the
            // non-split CUDA / OptiX raygens). Mixed scenes prefer area-light
            // emissive geometry; explicit point lights are a fallback for
            // scenes that have no area lights at all.
            else if (scene.d_pointLights && scene.pointLightCount > 0) {
                float3 direct = evalAllPointLightsNEE(
                    scene, scene.medium, hit.position, N, rng,
                    traceShadow, neeBrdf, /*fireflyClamp=*/10.0f);
                pathRadiance += throughput * direct;
            }

            if (scene.d_directionalLights && scene.directionalLightCount > 0) {
                float3 direct = evalAllDirectionalLightsNEE(
                    scene, scene.medium, hit.position, N, rng,
                    traceShadow, neeBrdf, /*fireflyClamp=*/10.0f);
                pathRadiance += throughput * direct;
            }

            // ReSTIR PT / GI consumption at the primary hit on sample s==0.
            // PT takes precedence; both branches add the pre-computed indirect
            // estimate and skip continuation bounces.
            //
            // CRITICAL: do NOT multiply by `throughput`. throughput == 1/pickedP
            // from the bucket override; the bucket routing redistributes
            // pathRadiance to the picked demod bucket already. Multiplying
            // by throughput would scale indirect by 1/pickedP, and the
            // demod-composite recovery becomes 2*indirect (double-counted).
            if (primaryLobeOverride && s == 0) {
                if (scene.restirPTEnabled != 0 && scene.d_restirPTIndirect != nullptr) {
                    float3 indirect = scene.d_restirPTIndirect[pixelIdx];
                    pathRadiance += indirect;
                    break;
                }
                if (scene.restirGIEnabled != 0 && scene.d_restirGIIndirect != nullptr) {
                    float3 indirect = scene.d_restirGIIndirect[pixelIdx];
                    pathRadiance += indirect;
                    break;
                }
            }

            // BSDF sampling for next bounce. Forced lobe at primary hit.
            // V was hoisted earlier for NEE.
            float specProb = materialSpecProb(mat, N, V, albedo);
            float3 newDir;
            bool sampleSpecular;
            if (primaryLobeOverride) {
                sampleSpecular = (pickedBucket == 1);
            } else {
                sampleSpecular = (pcg32_float(rng) < specProb);
            }
            if (sampleSpecular) {
                float a = mat.roughness * mat.roughness;
                float u1 = pcg32_float(rng), u2 = pcg32_float(rng);
                float cosT = sqrtf((1.0f - u1) / (1.0f + (a*a - 1.0f)*u1 + 1e-7f));
                float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT*cosT));
                float phi  = 2.0f * M_PI_F * u2;
                float3 lH  = make_float3(sinT*cosf(phi), cosT, sinT*sinf(phi));
                float3 T, B; buildONB(N, T, B);
                float3 H = localToWorld(lH, T, N, B);
                newDir = normalize(ray.direction - H * (2.0f * dot(ray.direction, H)));
                lastBounceDelta = false;
            } else {
                float u1 = pcg32_float(rng), u2 = pcg32_float(rng);
                float dummy;
                float3 lD = sampleCosineHemisphere(u1, u2, dummy);
                float3 T, B; buildONB(N, T, B);
                newDir = localToWorld(lD, T, N, B);
                lastBounceDelta = false;
            }
            float NdotLn = dot(N, newDir);
            if (NdotLn < 1e-6f) break;
            float pdf;
            float3 brdf;
            if (primaryLobeOverride) {
                if (pickedBucket == 0) {
                    pdf  = bsdfDiffusePdf(NdotLn);
                    brdf = materialDiffuseLobe(mat, N, V, newDir, albedo);
                } else {
                    pdf  = bsdfSpecularPdf(N, V, newDir, mat.roughness);
                    brdf = materialSpecularLobe(mat, N, V, newDir, albedo);
                }
            } else {
                pdf  = materialMixturePdf(mat, N, V, newDir, specProb);
                brdf = materialBsdfEvaluate(mat, N, V, newDir, albedo);
            }
            if (pdf < 1e-7f) break;
            throughput = throughput * brdf * (NdotLn / (pdf + 1e-7f));
            prevSurfacePos = hit.position; prevBsdfPdf = pdf; havePrevSurface = true;
            if (bounce >= 2) {
                float lum = 0.2126f*throughput.x + 0.7152f*throughput.y + 0.0722f*throughput.z;
                float p = fminf(fmaxf(lum, 0.05f), 0.95f);
                if (pcg32_float(rng) >= p) break;
                throughput = throughput * (1.0f / p);
            }
            ray.origin    = hit.position + N * 0.001f;
            ray.direction = newDir;
            ray.tmin      = 0.001f;
            ray.tmax      = 1e30f;
        } // end bounce loop

        splitAccumulateSppSample(acc, pathRadiance, emissiveContrib,
                                 haveGbuffer, pickedBucket, pg,
                                 bucketHitDist, bucketHitDistSet);
    } // end spp loop

    auto traceMirror = [&](float3 origin, float3 dir) -> float {
        RadiancePayload mrp = traceRadianceRay(handle, origin, dir, 0.001f, 1e30f);
        // Miss returns 1e4f — RR treats hitT=0 as "no reflection", so a
        // long-but-finite distance keeps the reflection-speed estimate sane.
        return mrp.hit ? mrp.tHit : 1.0e4f;
    };
    splitFinalizeAndWrite(acc, samplesPerPixel,
                          traceMirror, params.splitSurfaces, x, y,
                          /*applyDlssRRMinAlbedoGuard=*/true,         // OptiX: RTXPT min-albedo bump
                          SplitHdrClampPolicy::MaxChannel10);         // OptiX: RTXPT max-channel 10
}

// ─────────────────────────────────────────────────────────────────────────
// ReSTIR DI initial-candidates raygen (OptiX path).
//
// Mirrors `kReSTIR_InitCandidates` from render/ReSTIR.cu but traces the
// primary ray against the OptiX GAS via `traceRadianceRay()` instead of the
// CUDA SAH BVH. Output layout matches exactly — the CUDA temporal / spatial
// passes consume `params.restirReservoirsCurr` unchanged.
//
// All RIS helpers (target pdf, reservoir streaming) come from
// render/ReSTIRDevice.cuh so this program is a thin harness around the
// closest-hit + material-resolve logic already proven in the megakernel.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__ void __raygen__restir_init_candidates()
{
    uint3 idx = optixGetLaunchIndex();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    const DeviceSceneData& scene  = params.scene;
    const CameraParams&    camera = params.camera;
    OptixTraversableHandle handle = params.handle;

    ReSTIRReservoir r; restir_reservoirReset(r);
    ReSTIRSurface   surf{};
    surf.valid = 0.0f;

    // Dedicated RNG stream (matches the CUDA kernel's seeding salt).
    // Mix camera.frameIndex so the canonical sample changes every frame
    // even when sampleIndex is reset to 0 by camera motion.
    uint32_t seedSalt = params.sampleIndex + camera.frameIndex * 0x9E3779B9u;
    uint32_t rng = pcg32_seed(pixelIdx * 0xA1B2C3D4u + seedSalt,
                              seedSalt * 0xDEADBEEFu + 1u);

    // Primary ray — same jitter as the main kernel so the reservoir lines up
    // with the shading point that will consume it.
    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;
    Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

    RadiancePayload rp = traceRadianceRay(
        handle, ray.origin, ray.direction, ray.tmin, ray.tmax);

    bool eligible = (rp.hit != 0) && scene.d_areaLights &&
                    scene.areaLightCount > 0 && scene.d_lightBVHNodes;

    if (eligible) {
        uint32_t triIdx = rp.primIdx;
        uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
        uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
        uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
        float baryU = rp.baryU;
        float baryV = rp.baryV;
        float baryW = 1.0f - baryU - baryV;
        float3 v0 = scene.d_positions[i0];
        float3 v1 = scene.d_positions[i1];
        float3 v2 = scene.d_positions[i2];
        float3 hitPos = v0 * baryW + v1 * baryU + v2 * baryV;

        int matIdx = scene.d_materialIndices ? scene.d_materialIndices[triIdx] : -1;
        if (matIdx < 0 || (uint32_t)matIdx >= scene.materialCount) {
            eligible = false;
        }
        if (eligible) {
            GPUMaterial mat = scene.d_materials[matIdx];

            // Shading normal — interpolate vertex normals, flip to face the ray.
            float3 N;
            if (scene.d_normals) {
                N = normalize(scene.d_normals[i0] * baryW +
                              scene.d_normals[i1] * baryU +
                              scene.d_normals[i2] * baryV);
            } else {
                N = normalize(cross(v1 - v0, v2 - v0));
            }
            if (dot(N, ray.direction) > 0.0f) N = -N;

            // Texture UV + albedo / metallic / roughness resolve.
            float2 uv = make_float2(0.0f, 0.0f);
            if (scene.d_uvs) {
                uv = scene.d_uvs[i0] * baryW +
                     scene.d_uvs[i1] * baryU +
                     scene.d_uvs[i2] * baryV;
            }
            float3 albedo = mat.albedo;
            if (mat.albedoTex != 0) {
                float4 t = tex2D<float4>(mat.albedoTex, uv.x, uv.y);
                albedo = albedo * make_float3(t.x, t.y, t.z);
            }
            if (mat.metallicRoughTex != 0) {
                float4 mrT = tex2D<float4>(mat.metallicRoughTex, uv.x, uv.y);
                mat.roughness *= mrT.y;
                mat.metallic  *= mrT.z;
            }

            surf.position    = hitPos;
            surf.normal      = N;
            surf.albedo      = albedo;
            // Floor roughness so pHat's GGX term doesn't spike near zero —
            // keeps RIS well-conditioned on near-mirror surfaces. Matches
            // the CUDA kernel.
            surf.roughness   = fmaxf(mat.roughness, 0.04f);
            surf.metallic    = mat.metallic;
            surf.pureDiffuse = mat.pureDiffuse ? 1u : 0u;
            surf.viewDir     = -ray.direction;
            surf.valid       = 1.0f;

            // Cached specProb for downstream passes.
            {
                float NdotV = fmaxf(dot(surf.normal, surf.viewDir), 0.0f);
                float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), surf.albedo, surf.metallic);
                float t = 1.0f - fminf(fmaxf(NdotV, 0.0f), 1.0f);
                float t5 = t*t*t*t*t;
                float3 F = F0 + (make_float3(1,1,1) - F0) * t5;
                float specW = luminance(F);
                float3 kd = (make_float3(1,1,1) - F) * (1.0f - surf.metallic);
                float diffW = luminance(kd * surf.albedo);
                float p = specW / fmaxf(specW + diffW, 1e-7f);
                surf.specProb = fminf(fmaxf(p, 0.1f), 0.9f);
            }

            // Animated-geometry-aware prev-pixel for temporal reservoir reuse.
            float3 hitPosPrevDI = hitPos;
            if (scene.d_positionsPrev) {
                float3 v0p = scene.d_positionsPrev[i0];
                float3 v1p = scene.d_positionsPrev[i1];
                float3 v2p = scene.d_positionsPrev[i2];
                hitPosPrevDI = v0p * baryW + v1p * baryU + v2p * baryV;
            }
            float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hitPosPrevDI);
            surf.prevPixel = make_float2(
                (clipPrev.x + 1.0f) * 0.5f * (float)params.width,
                (1.0f - clipPrev.y) * 0.5f * (float)params.height);

            // ── RIS: draw M candidates from the light BVH, stream them ──
            uint32_t M = params.restirNumCandidates;
            float wSum = 0.0f;
            for (uint32_t i = 0; i < M; i++) {
                float u = pcg32_float(rng);
                uint32_t slot = 0;
                float pSelect = 0.0f;
                if (!lightBVH_sample(scene.d_lightBVHNodes,
                                     scene.lightBVHRootIndex,
                                     surf.position, u, slot, pSelect) ||
                    !(pSelect > 0.0f)) {
                    // Parity with render/ReSTIR.cu kReSTIR_InitCandidates:
                    // count failed BVH descents toward M so finalize divides
                    // by the true number of candidates considered. Otherwise
                    // shading points where the BVH frequently rejects (above
                    // / behind a tight light cluster) get an inflated W.
                    r.M += 1.0f;
                    continue;
                }
                uint32_t lightIdx = scene.d_lightOrderedIndices[slot];
                GPUAreaLight light = scene.d_areaLights[lightIdx];

                float r1 = pcg32_float(rng);
                float r2 = pcg32_float(rng);
                float su = sqrtf(r1);
                float cb1 = su * (1.0f - r2);
                float cb2 = su * r2;

                float areaPdf = pSelect / fmaxf(light.area, 1e-7f);
                float pHat = restirEvalTargetPdf(surf, light, cb1, cb2);
                float wCand = (areaPdf > 0.0f) ? (pHat / areaPdf) : 0.0f;

                restir_reservoirUpdate(r, wSum, lightIdx, cb1, cb2, pHat,
                                       wCand, pcg32_float(rng));
            }
            restir_reservoirFinalize(r, wSum);
        }
    }

    if (params.restirReservoirsCurr) params.restirReservoirsCurr[pixelIdx] = r;
    if (params.restirSurfacesCurr)   params.restirSurfacesCurr[pixelIdx]   = surf;
}

// ─────────────────────────────────────────────────────────────────────────
// ReSTIR DI visibility-reuse raygen (OptiX path).
//
// Mirrors `kReSTIR_Visibility` from render/ReSTIR.cu but traces the shadow
// ray against the OptiX GAS instead of the CUDA SAH BVH. One ray per pixel
// toward the held sample's point on the light; zeros W on occlusion so the
// subsequent temporal / spatial reuse can't propagate occluded samples
// (Bitterli 2020 Alg. 5 lines 6-9).
//
// Reuses the existing shadow SBT (offset 1, miss 1) so we get the same
// glass-transparency handling as the main shading shadow ray.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__ void __raygen__restir_visibility()
{
    uint3 idx = optixGetLaunchIndex();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    if (!params.restirReservoirsCurr || !params.restirSurfacesCurr) return;
    ReSTIRReservoir r = params.restirReservoirsCurr[pixelIdx];
    if (r.lightIndex == 0xFFFFFFFFu || r.W <= 0.0f) return;

    ReSTIRSurface s = params.restirSurfacesCurr[pixelIdx];
    if (s.valid < 0.5f) return;

    const DeviceSceneData& scene = params.scene;
    GPUAreaLight light = scene.d_areaLights[r.lightIndex];
    float b0 = 1.0f - r.baryB1 - r.baryB2;
    float3 pOnLight = light.v0 * b0
                    + (light.v0 + light.e1) * r.baryB1
                    + (light.v0 + light.e2) * r.baryB2;

    float3 origin = s.position + s.normal * 1e-3f;
    float3 toL    = pOnLight - origin;
    float  dist   = sqrtf(fmaxf(dot(toL, toL), 1e-12f));
    float3 dir    = toL * (1.0f / dist);
    // Pull tmax in slightly so the light triangle itself doesn't register.
    float  tmax   = fmaxf(dist - 2e-3f, 1e-4f);

    float3 transmittance = traceShadowRay(
        params.handle, origin, dir, 1e-3f, tmax);
    // Treat fully-occluded (transmittance == 0) as a kill. Glass paths return
    // partial transmittance > 0; we keep those samples — they're what the
    // final-shading shadow ray will also accumulate against.
    float lum = luminance(transmittance);
    if (lum <= 1e-6f) {
        r.W = 0.0f;
        params.restirReservoirsCurr[pixelIdx] = r;
    }
}

// ── ReSTIR GI raygen (lives in OptiXProgramsGI.inl to keep this file
// readable). Compiled into the same module as the rest of the OptiX
// programs so it sees `params`, `traceRadianceRay`, etc.
#include "backend/OptiXProgramsGI.inl"

// ── ReSTIR PT raygen — multi-bounce postfix random walk past the
// reconnection vertex. Same idea as the GI raygen but with a longer path.
#include "backend/OptiXProgramsPT.inl"
