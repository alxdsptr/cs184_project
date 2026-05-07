#include "render/PathTraceKernel.h"

#ifdef PATHTRACER_NRD_DLSS_ENABLED

#include "render/PathTraceHelpers.cuh"
#include "render/GBufferWriters.cuh"
#include "render/NEEHelpers.cuh"
#include "render/ReSTIR.h"
#include "render/VolumeNEE.cuh"
#include "render/VolumeShadowCuda.cuh"
#include "gpu/NRDHelpers.cuh"
#include "accel/BVH.h"
#include "gpu/Random.h"
#include "gpu/BRDF.h"
#include "util/CudaCheck.h"
#include "core/VolumeMedium.h"
#include "core/VolumeDevice.cuh"

#include <cuda_fp16.h>

// Path classification policy at the primary hit:
//   - Roll one random number r against specProb to pick a bucket (diff or spec).
//   - At the PRIMARY hit only, NEE and BSDF sampling are restricted to the
//     chosen lobe (diffuse-only BRDF / cosine PDF, or specular-only BRDF /
//     GGX PDF). Throughput is scaled by 1/pickedP to keep the estimator
//     unbiased at the bucket level.
//   - Indirect bounces beyond the primary use the full mixture BRDF, since by
//     then the bucket assignment is already fixed and we just need correct
//     unbiased path integration from that point on.
//
// Rationale: if we put the full (diffuse + specular) BRDF into a single bucket
// each frame, NRD's temporal mean of each bucket approaches the full radiance,
// and the composite ends up double-counting (diff*alb + spec ~ 2x). Routing
// only the diffuse-lobe contribution through the diffuse bucket (and vice
// versa) makes diff*alb + spec recover the true primary-hit radiance.

__global__ void pathTraceKernelSplit(
    DeviceSceneData       scene,
    CameraParams          camera,
    SplitSurfaceOutputs   surfaces,
    uint32_t              width,
    uint32_t              height,
    uint32_t              sampleIndex,
    bool                  enableEnvironment,
    uint32_t              maxBounces,
    uint32_t              samplesPerPixel)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const uint32_t pixelIdx = y * width + x;

    // Accumulators averaged across samplesPerPixel (spp). NRD sees the mean,
    // so averaging N samples in-kernel reduces per-frame variance by ~N and
    // substantially cuts the single-sample bucket spikes that read as water
    // ripples after temporal filtering.
    float3 demodDiffSum = make_float3(0, 0, 0);
    float3 demodSpecSum = make_float3(0, 0, 0);
    float3 emissiveSum  = make_float3(0, 0, 0);
    float  diffHitSum = 0.0f; uint32_t diffHitCount = 0;
    float  specHitSum = 0.0f; uint32_t specHitCount = 0;
    // DLSS-RR: noisy un-demodulated combined color (diff*alb + spec + emissive)
    // = pathRadiance (already lobe-routed and 1/pickedP scaled, so unbiased
    //   over both buckets) + emissiveContrib.
    // hitT averaged across all samples (not just one bucket) — DLSS-RR wants
    // the per-pixel specular hit distance regardless of which lobe was picked.
    float3 noisyColorSum = make_float3(0, 0, 0);
    float  anyHitSum = 0.0f; uint32_t anyHitCount = 0;

    // G-buffer captured from the first sample that produces a primary opaque
    // hit. NRD only consumes one g-buffer per pixel, not an average.
    bool   gbufferWritten = false;
    float3 outPrimaryAlbedo   = make_float3(0, 0, 0);
    float3 outPrimaryNormal   = make_float3(0, 1, 0);
    float  outPrimaryRoughness = 1.0f;
    // Sky-pixel sentinel viewZ: large positive value → NRD treats as "far".
    // viewZ=0 would be misread as the near plane and corrupt disocclusion.
    float  outPrimaryViewZ     = 1.0e6f;
    float2 outPrimaryMvPx      = make_float2(0.0f, 0.0f);
    float  outPrimaryNdcZ      = 1.0f;  // DLSS-style NDC depth (1 = far)
    // DLSS-RR fix: capture primary hit position + metallic for an explicit
    // mirror-ray spec hitT trace after the spp loop, and for a metallic-aware
    // F0 in the spec albedo guide buffer. Per-spp lobe-bounce hitT is biased
    // (diffuse-bucket samples land elsewhere than the spec lobe) → shimmer.
    float3 outPrimaryHitPos   = make_float3(0, 0, 0);
    float3 outPrimaryRayDir   = make_float3(0, 0, -1);
    float  outPrimaryMetallic = 0.0f;

    if (samplesPerPixel < 1) samplesPerPixel = 1;

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
        // Unique RNG subseed per (pixel, frame, sample-in-frame).
        uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                                  sampleIndex * 0x85EBCA6Bu + s);

    // NRD/DLSS require that the sub-pixel offset actually used at ray-gen
    // exactly matches `CommonSettings::cameraJitter` (Halton). Adding a
    // per-sample random offset here would make the real sample land
    // somewhere else sub-pixel-wise, so NRD's history reprojection is off
    // by up to a full pixel — manifesting as a persistent water-wave
    // jitter on static frames.
    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;

    Ray ray = generateRay(x, y, width, height, camera, jx, jy);

    float3 throughput = make_float3(1, 1, 1);
    float3 pathRadiance = make_float3(0, 0, 0);
    float3 emissiveContrib = make_float3(0, 0, 0);

    // Primary-hit state for the g-buffer + bucket classification.
    bool haveGbuffer = false;
    float3 primaryAlbedo = make_float3(0, 0, 0);
    float3 primaryNormal = make_float3(0, 1, 0);
    float primaryRoughness = 1.0f;
    float primaryViewZ = 0.0f;
    float2 primaryMvPx = make_float2(0.0f, 0.0f);
    float primaryNdcZ  = 1.0f;
    // DLSS-RR fix: snapshot of primary-hit world-space position, view ray dir
    // and metallic. Lifted out of the inner loop so the `!gbufferWritten` block
    // can copy them to the per-pixel `outPrimary*` set without recomputing.
    float3 primaryHitPos = make_float3(0, 0, 0);
    float3 primaryRayDir = make_float3(0, 0, -1);
    float  primaryMetallic = 0.0f;
    int   pickedBucket = 0;       // 0 = diffuse, 1 = specular
    float bucketHitDist = 0.0f;    // world-space distance to first indirect surface
    bool  bucketHitDistSet = false;

    bool firstBounce = true;
    // True only when the previous bounce used a *delta* BSDF (perfect mirror /
    // glass refraction). See PathTraceKernel.cu for the longer comment — the
    // short version: Cook-Torrance specular is NOT delta and must not set this.
    bool lastBounceDelta = false;
    bool havePrevSurface = false;
    float3 prevSurfacePos = make_float3(0, 0, 0);
    float prevBsdfPdf = 1.0f;

    for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
        // True only during the iteration where the primary opaque hit is
        // classified into a bucket. Used to restrict NEE and BSDF sampling
        // at the primary surface to the chosen lobe so that diff+spec
        // buckets partition the primary-hit radiance rather than duplicate it.
        bool primaryLobeOverride = false;

        HitRecord hit; hit.t = ray.tmax;
        bool didHit = false;
        if (scene.d_bvhNodes && scene.totalTriangles > 0) {
            didHit = bvh_closestHit(
                ray, scene.d_bvhNodes, scene.bvhRootIndex,
                scene.d_positions, scene.d_indices, scene.d_materialIndices,
                hit);
        }

        // ── Participating-medium integrator (NRD/DLSS-RR-compatible) ────
        // Volumetric scattering doesn't fit NRD's diff/spec demodulated
        // buckets (view-dependent radiance, no surface albedo to demodulate
        // by), so single-scatter NEE is routed into the emissive bucket.
        //   • emissiveSum: untouched by NRD's diff/spec denoisers, summed
        //     across spp and composited as-is — accepts noise gracefully.
        //   • noisyColorSum: composes pathRadiance + emissiveContrib, so
        //     DLSS-RR sees the in-scatter as part of its ML-denoised input.
        // Scatter events terminate the path (single-scatter only). This is
        // biased relative to the megakernel's full multi-scatter integrator
        // but matches typical real-time engine fog and keeps the bucket
        // math clean — multi-scatter would need either further emissive
        // accumulation or a lying injection into diff/spec, which breaks
        // NRD's demodulation invariant.
        {
            float segmentDistance = didHit ? hit.t : ray.tmax;
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
                        // Single-scatter NEE across all lights, shared with the
                        // megakernel + OptiX raygens via render/VolumeNEE.cuh.
                        float3 inScatter = volumeSingleScatterInScatter(
                            scene, scene.medium, mediumPos, wo, rng,
                            [&](float3 o, float3 d, float t) {
                                return cudaTraceTransmissiveShadow(scene, o, d, t);
                            });

                        // Add to emissive bucket. Survives NRD untouched and
                        // appears in DLSS-RR's noisy color (pathRadiance +
                        // emissiveContrib). Scatter terminates the path.
                        float3 contrib = throughput * ssAlbedo * inScatter;
                        emissiveContrib += clampFirefly(contrib, 10.0f);
                        break;
                    }
                    // No scatter inside the volume — apply ratio-tracked
                    // transmittance to surface contributions behind the fog.
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
                    // Primary-ray miss: there is no surface, so there's no
                    // diff/spec bucket to demodulate into. Route the sky
                    // through the emissive channel — it bypasses NRD denoise
                    // and the composite shader adds it as `+ emis`. Without
                    // this the env color silently drops on the floor: the
                    // demod block below only runs when haveGbuffer is true.
                    emissiveContrib = envColor;
                } else {
                    pathRadiance += clampFirefly(throughput * envColor, 10.0f);
                }
            }
            break;
        }

        GPUMaterial mat;
        if (hit.materialIndex >= 0 && (uint32_t)hit.materialIndex < scene.materialCount)
            mat = scene.d_materials[hit.materialIndex];
        else {
            mat.albedo = make_float3(0.8f, 0.2f, 0.8f);
            mat.roughness = 0.5f; mat.metallic = 0.0f;
            mat.emission = make_float3(0,0,0); mat.emissionStrength = 0.0f;
            mat.useSpecularGlossiness = 0;
            mat.specularGlossAlphaIsGlossiness = 0;
            mat.useFBXCustomPacking = 0;
            mat.useFBXUEPacking = 0;
            mat.specularColor = make_float3(1.0f, 1.0f, 1.0f);
            mat.glossiness = 0.5f;
            mat.specularGlossTex = 0;
        }

        uint32_t triIdx = (uint32_t)hit.primitiveIndex;
        uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
        uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
        uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
        float baryU = hit.uv.x, baryV = hit.uv.y;
        float baryW = 1.0f - baryU - baryV;

        float2 texUV = make_float2(0.0f, 0.0f);
        if (scene.d_uvs) {
            texUV = scene.d_uvs[i0] * baryW + scene.d_uvs[i1] * baryU + scene.d_uvs[i2] * baryV;
        }

        float3 albedo;
        float3 emissiveColor;
        applyMaterialTextures(mat, texUV, albedo, emissiveColor);

        float3 N = hit.shadingNormal;
        if (scene.d_normals) {
            N = normalize(scene.d_normals[i0] * baryW + scene.d_normals[i1] * baryU + scene.d_normals[i2] * baryV);
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
            primaryAlbedo = albedo;
            primaryNormal = N;
            primaryRoughness = mat.roughness;
            primaryHitPos    = hit.position;
            primaryRayDir    = ray.direction;
            primaryMetallic  = mat.metallic;
            primaryViewZ = nrd_helpers::computeViewZ(hit.position, camera.position, camera.forward);
            primaryMvPx = nrd_helpers::computeMotionVectorPx(
                hit.position, camera.viewProjMatrix, camera.prevViewProjMatrix, width, height);
            // NDC depth for DLSS — RELAX wants linear viewZ (already above),
            // DLSS Super-Resolution wants post-perspective clip.z/clip.w.
            // mat4_transformPoint does the perspective divide so .z is NDC z in
            // [-1,1] (GL); remap to DLSS's [0,1] convention.
            {
                float3 ndc = mat4_transformPoint(camera.viewProjMatrix, hit.position);
                primaryNdcZ = clampf(ndc.z * 0.5f + 0.5f, 0.0f, 1.0f);
            }

            float3 V = -ray.direction;
            float specProb = materialSpecProb(mat, N, V, albedo);
            pickedBucket = (pcg32_float(rng) < specProb) ? 1 : 0;
            // Correct for the bucket pick: divide the lobe-only contribution
            // by the selected probability. Combined with forcing NEE/BSDF at
            // the primary hit to the chosen lobe, this makes
            // E[demodDiff*alb + demodSpec] = primary-hit radiance (unbiased).
            float pickedP = (pickedBucket == 1) ? specProb : (1.0f - specProb);
            throughput = throughput * (1.0f / fmaxf(pickedP, 1e-4f));

            haveGbuffer = true;
            firstBounce = false;
            primaryLobeOverride = true;
        }

        // Glass (delta BSDF) — skipped for classification, treated as specular.
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

        // RELAX's spatial filter radius is driven by the distance from the
        // primary surface to the first secondary hit of the chosen lobe.
        // Using `bounce == 1` (the iteration immediately after primary hit)
        // keeps this stable across frames — any later / lobe-dependent choice
        // makes hitT jitter with the BSDF sample, which RELAX misreads as a
        // depth change and the filter radius swims (→ water ripples).
        if (!bucketHitDistSet && bounce == 1) {
            bucketHitDist = hit.t;
            bucketHitDistSet = true;
        }

        bool isEmissive = mat.emissionStrength > 0.0f &&
            (emissiveColor.x > 0.0f || emissiveColor.y > 0.0f || emissiveColor.z > 0.0f);
        if (isEmissive) {
            float3 Le = emissiveColor * mat.emissionStrength;
            float weight = 1.0f;
            // MIS the emissive hit against the light-sampling strategy (CDF
            // pTri only — the split kernel doesn't use the light BVH).
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
                // Primary emissive routes to the non-denoised emissive bucket
                // (NRD's diff/spec demodulation has no surface to demodulate
                // against). Composite layer adds it as `+ emis`.
                emissiveContrib = Le * weight;
            } else {
                pathRadiance += clampFirefly(throughput * Le * weight, 10.0f);
            }
            if (mat.emissiveTex == 0) break;
        }

        // NEE callables — defined once per bounce, used by every light type.
        // Split-kernel BRDF/PDF respect the primary-hit lobe override so the
        // demod buckets stay invariant; indirect bounces fall back to the
        // mixture (primaryLobeOverride is false there).
        float3 V = -ray.direction;
        auto traceShadow = [&](float3 o, float3 d, float dist) {
            return cudaTraceTransmissiveShadow(scene, o, d, dist);
        };
        auto neeBrdf = [&](float3 Ld, float NdotL) {
            return evalNEEBrdf(mat, N, V, Ld, albedo, primaryLobeOverride, pickedBucket);
        };
        auto neePdf = [&](float3 Ld, float NdotL) {
            return evalNEEBrdfPdf(mat, N, V, Ld, albedo, primaryLobeOverride, pickedBucket);
        };

        // NEE area lights.  ReSTIR DI replaces the per-frame CDF pick at the
        // primary hit (s==0, bounce==0) with the resampled reservoir's
        // selected sample. Lobe-override still applies — ReSTIR DI feeds the
        // chosen bucket only, multiplied by 1/pickedP via `throughput`. The
        // estimator becomes f * W (no MIS against BSDF — see PathTraceKernel.cu
        // for the same comment).
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
                li = sampleAreaLightIndex(scene.d_areaLightCDF, scene.areaLightCount, pcg32_float(rng));
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
            // is sensitive to single-sample spikes (see PathTraceKernelSplit
            // header comment on bucket-pickedP and NRD's water-ripple artifact).
            pathRadiance += evalAreaLightNEEContribution(
                throughput, hit.position, N, light, b0, b1, b2, pSelect,
                restirActive, restirW,
                scene.medium, rng, traceShadow, neeBrdf, neePdf,
                /*fireflyClamp=*/10.0f);
            } // end !restirSkip
        }

        // Point lights: only sampled when no area lights, matching
        // PathTraceKernel.cu and the OptiX raygens. Scenes with emissive
        // textures use area lights; point lights are a fallback for scenes
        // that ship no area lights at all.
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
        // PT takes precedence over GI (PT subsumes GI's 1-bounce NEE plus k
        // more bounces of light transport). Either branch adds the
        // pre-computed indirect estimate and skips continuation bounces.
        //
        // CRITICAL: do NOT multiply by `throughput` here. At the primary hit
        // throughput == 1/pickedP from the bucket override; the bucket
        // routing below already redistributes pathRadiance to the picked
        // demod bucket. Multiplying by throughput would scale the indirect
        // by 1/pickedP, and the demod-composite recovery becomes:
        //    E[pickedP * indirect/pickedP] summed over both buckets = 2*indirect.
        // i.e. the indirect ends up double-counted. Adding a bare
        // `pathRadiance += indirect` lets each bucket carry the full
        // indirect; demod averages restore exactly `indirect` after composite.
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

        // BRDF sampling for the next bounce. At the primary hit the lobe is
        // forced to match `pickedBucket`; at subsequent hits we use the full
        // mixture since the bucket is already locked in. V was hoisted earlier
        // for NEE.
        float specProb = materialSpecProb(mat, N, V, albedo);
        float3 newDir;
        bool sampleSpecularLobe;
        if (primaryLobeOverride) {
            sampleSpecularLobe = (pickedBucket == 1);
        } else {
            sampleSpecularLobe = (pcg32_float(rng) < specProb);
        }
        if (sampleSpecularLobe) {
            float a = mat.roughness * mat.roughness;
            float u1 = pcg32_float(rng), u2 = pcg32_float(rng);
            float cosT = sqrtf((1.0f - u1) / (1.0f + (a*a - 1.0f)*u1 + 1e-7f));
            float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT*cosT));
            float phi = 2.0f * M_PI_F * u2;
            float3 lH = make_float3(sinT*cosf(phi), cosT, sinT*sinf(phi));
            float3 T, B; buildONB(N, T, B);
            float3 H = localToWorld(lH, T, N, B);
            newDir = normalize(ray.direction - H * (2.0f * dot(ray.direction, H)));
            // Cook-Torrance specular lobe is NOT delta — MIS is valid.
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
        ray.origin = hit.position + N * 0.001f;
        ray.direction = newDir;
        ray.tmin = 0.001f; ray.tmax = 1e30f;
    }

    // Sanitize and clamp.
    if (isnan(pathRadiance.x) || isnan(pathRadiance.y) || isnan(pathRadiance.z) ||
        isinf(pathRadiance.x) || isinf(pathRadiance.y) || isinf(pathRadiance.z)) {
        pathRadiance = make_float3(0,0,0);
    }
    // Per-channel clamp. A luminance-only clamp at 200 lets a single saturated
    // green firefly through at ~280 (since g-weight is 0.72); RELAX then takes
    // ~30 frames to fade it. A per-channel cap at 15 kills those spikes hard.
    pathRadiance.x = fminf(fmaxf(pathRadiance.x, 0.0f), 15.0f);
    pathRadiance.y = fminf(fmaxf(pathRadiance.y, 0.0f), 15.0f);
    pathRadiance.z = fminf(fmaxf(pathRadiance.z, 0.0f), 15.0f);

    // Demodulate by albedo so NRD sees the irradiance component; composite
    // remultiplies. Guard against zero albedo (pure metallic → specular bucket).
    float3 demodDiff = make_float3(0,0,0);
    float3 demodSpec = make_float3(0,0,0);
    if (haveGbuffer) {
        if (pickedBucket == 0) {
            float3 invA = make_float3(
                1.0f / fmaxf(primaryAlbedo.x, 1e-3f),
                1.0f / fmaxf(primaryAlbedo.y, 1e-3f),
                1.0f / fmaxf(primaryAlbedo.z, 1e-3f));
            demodDiff = pathRadiance * invA;
        } else {
            demodSpec = pathRadiance;
        }
    }

        // Accumulate this sample's contribution.
        demodDiffSum = demodDiffSum + demodDiff;
        demodSpecSum = demodSpecSum + demodSpec;
        emissiveSum  = emissiveSum  + emissiveContrib;
        if (haveGbuffer && bucketHitDistSet) {
            if (pickedBucket == 0) { diffHitSum += bucketHitDist; diffHitCount++; }
            else                    { specHitSum += bucketHitDist; specHitCount++; }
            // DLSS-RR specHitT: cheap approximation — feed the first secondary
            // hit distance for whichever lobe was rolled this sample. For
            // diffuse-bucket samples this is the cosine-sampled bounce, which
            // approximates "where reflections land" well enough for matte/
            // semi-glossy surfaces. Glossy mirrors will overwhelmingly land in
            // the spec bucket so they get the GGX-sampled distance directly.
            anyHitSum += bucketHitDist; anyHitCount++;
        }
        // Noisy combined color: pathRadiance already incorporates 1/pickedP, so
        // E_buckets[pathRadiance] = full primary-hit radiance. Adding emissive
        // gives the un-demodulated color DLSS-RR wants.
        noisyColorSum = noisyColorSum + pathRadiance + emissiveContrib;
        // G-buffer: first sample that produced a primary hit wins. Averaging
        // normals / viewZ across samples would soften silhouettes and break
        // NRD's disocclusion test, so we don't.
        if (!gbufferWritten && haveGbuffer) {
            outPrimaryAlbedo    = primaryAlbedo;
            outPrimaryNormal    = primaryNormal;
            outPrimaryRoughness = primaryRoughness;
            outPrimaryViewZ     = primaryViewZ;
            outPrimaryMvPx      = primaryMvPx;
            outPrimaryNdcZ      = primaryNdcZ;
            outPrimaryHitPos    = primaryHitPos;
            outPrimaryRayDir    = primaryRayDir;
            outPrimaryMetallic  = primaryMetallic;
            gbufferWritten = true;
        }
    } // end spp loop

    // Average per-pixel radiance over the samples taken.
    float invSpp = 1.0f / (float)samplesPerPixel;
    float3 demodDiffAvg = demodDiffSum * invSpp;
    float3 demodSpecAvg = demodSpecSum * invSpp;
    float3 emissiveAvg  = emissiveSum  * invSpp;
    float3 noisyColorAvg = noisyColorSum * invSpp;
    // HitDist: average only over samples that actually filled the bucket, so
    // pixels where one sample went diffuse and the others specular don't bias
    // the diff-bucket hitT toward zero.
    float diffHitAvg = diffHitCount > 0 ? (diffHitSum / (float)diffHitCount) : 0.0f;
    float specHitAvg = specHitCount > 0 ? (specHitSum / (float)specHitCount) : 0.0f;
    float anyHitAvg  = anyHitCount  > 0 ? (anyHitSum  / (float)anyHitCount)  : 0.0f;

    // DLSS-RR specular albedo: F0 = lerp(0.04, primaryAlbedo, metallic) per
    // the integration guide §3.4.2 + Appendix EnvBRDFApprox2. We now preserve
    // the primary-hit metallic into outPrimaryMetallic so dielectric vs metal
    // surfaces get the right F0. NoV uses the actual primary ray direction
    // (not the unjittered camera.forward) so the spec-albedo guide buffer
    // moves smoothly across frames. Sky / no-hit pixels default to 0.5.
    float3 specF0 = lerp(make_float3(0.04f, 0.04f, 0.04f),
                         outPrimaryAlbedo, outPrimaryMetallic);
    float NoV = fmaxf(-dot(outPrimaryRayDir, outPrimaryNormal), 0.0f);
    float3 specAlbedoAvg = envBRDFApprox2(specF0,
                                          outPrimaryRoughness * outPrimaryRoughness,
                                          NoV);
    if (!gbufferWritten) {
        specAlbedoAvg = make_float3(0.5f, 0.5f, 0.5f);
    }

    // DLSS-RR specular hit distance (§3.4.9): "World Space distance between
    // the Specular Ray Origin and Hit Point. Specular Ray Origin must be on
    // the Primary Surface." The previous implementation fed `anyHitAvg`,
    // which averages secondary-bounce distances across BOTH lobes — diffuse-
    // bucket samples land on a cosine-sampled bounce, NOT where the spec
    // reflection would land. That makes the value flicker frame-to-frame
    // depending on which lobe the bucket roll picks, producing the surface-
    // wide motion shimmer we measured (heatmap shows broad surface activity,
    // not just edges). Trace ONE explicit mirror ray per pixel from the
    // primary hit along the perfect-reflection direction: deterministic,
    // sub-pixel-stable, and matches the canonical-reflection semantics RR
    // expects for deriving specular MV.
    float rrSpecHitT = 0.0f;
    if (gbufferWritten) {
        float3 rd = outPrimaryRayDir;
        float3 N  = outPrimaryNormal;
        float3 mirrorDir = normalize(rd - N * (2.0f * dot(rd, N)));
        Ray   mr;
        mr.origin    = outPrimaryHitPos + N * 0.001f;
        mr.direction = mirrorDir;
        mr.tmin      = 0.001f;
        mr.tmax      = 1e30f;
        HitRecord mhit; mhit.t = mr.tmax;
        bool mDidHit = false;
        if (scene.d_bvhNodes && scene.totalTriangles > 0) {
            mDidHit = bvh_closestHit(
                mr, scene.d_bvhNodes, scene.bvhRootIndex,
                scene.d_positions, scene.d_indices, scene.d_materialIndices,
                mhit);
        }
        // If we miss (sky / outside scene), report a long but finite distance
        // — RR uses hitT to derive the speed of the reflected feature; a 0
        // here would be misread as "no reflection at all".
        rrSpecHitT = mDidHit ? mhit.t : 1.0e4f;
    }
    if (isnan(rrSpecHitT) || isinf(rrSpecHitT) || rrSpecHitT < 0.0f) rrSpecHitT = 0.0f;

    float4 diffTexel = nrd_helpers::packRadianceHitDist(demodDiffAvg, diffHitAvg);
    float4 specTexel = nrd_helpers::packRadianceHitDist(demodSpecAvg, specHitAvg);
    float4 normTexel = nrd_helpers::packNormalRoughness(outPrimaryNormal, outPrimaryRoughness);
    float4 albTexel  = make_float4(
        fminf(fmaxf(outPrimaryAlbedo.x, 0.0f), 1.0f),
        fminf(fmaxf(outPrimaryAlbedo.y, 0.0f), 1.0f),
        fminf(fmaxf(outPrimaryAlbedo.z, 0.0f), 1.0f),
        1.0f);
    float4 emTexel = make_float4(emissiveAvg.x, emissiveAvg.y, emissiveAvg.z, 1.0f);

    // surf2Dwrite writes sizeof(T) bytes at the given BYTE offset. For
    // RGBA16F textures (8 bytes/texel) we must NOT write `float4` (16 bytes)
    // at `x * 8` — that spills into the next pixel and silently corrupts the
    // NRD inputs (which looks exactly like "the denoiser has no effect").
    // Pack to a ushort4 carrying four __half bit patterns instead.
    auto packHalf4 = [](float4 v) -> ushort4 {
        __half hx = __float2half(v.x);
        __half hy = __float2half(v.y);
        __half hz = __float2half(v.z);
        __half hw = __float2half(v.w);
        ushort4 r;
        r.x = *reinterpret_cast<unsigned short*>(&hx);
        r.y = *reinterpret_cast<unsigned short*>(&hy);
        r.z = *reinterpret_cast<unsigned short*>(&hz);
        r.w = *reinterpret_cast<unsigned short*>(&hw);
        return r;
    };

    if (surfaces.diffuseRadianceHitDist) {
        ushort4 p = packHalf4(diffTexel);
        surf2Dwrite<ushort4>(p, surfaces.diffuseRadianceHitDist, x * 8, y); // RGBA16F = 8B
    }
    if (surfaces.specularRadianceHitDist) {
        ushort4 p = packHalf4(specTexel);
        surf2Dwrite<ushort4>(p, surfaces.specularRadianceHitDist, x * 8, y);
    }
    if (surfaces.normalRoughness) {
        uchar4 nr;
        nr.x = (unsigned char)(normTexel.x * 255.0f + 0.5f);
        nr.y = (unsigned char)(normTexel.y * 255.0f + 0.5f);
        nr.z = (unsigned char)(normTexel.z * 255.0f + 0.5f);
        nr.w = (unsigned char)(normTexel.w * 255.0f + 0.5f);
        surf2Dwrite<uchar4>(nr, surfaces.normalRoughness, x * 4, y); // RGBA8 = 4B
    }
    if (surfaces.viewZ)
        surf2Dwrite<float>(outPrimaryViewZ, surfaces.viewZ, x * 4, y); // R32F = 4B
    if (surfaces.ndcDepth)
        surf2Dwrite<float>(outPrimaryNdcZ, surfaces.ndcDepth, x * 4, y); // R32F = 4B
    if (surfaces.motionVectors) {
        // RG16F = 4B. surf2Dwrite doesn't expose an __half2 overload — write
        // as a ushort2 whose bit pattern is a pair of halves.
        __half hx = __float2half(outPrimaryMvPx.x);
        __half hy = __float2half(outPrimaryMvPx.y);
        ushort2 packed;
        packed.x = *reinterpret_cast<unsigned short*>(&hx);
        packed.y = *reinterpret_cast<unsigned short*>(&hy);
        surf2Dwrite<ushort2>(packed, surfaces.motionVectors, x * 4, y);
    }
    if (surfaces.albedo) {
        uchar4 a4;
        a4.x = (unsigned char)(albTexel.x * 255.0f + 0.5f);
        a4.y = (unsigned char)(albTexel.y * 255.0f + 0.5f);
        a4.z = (unsigned char)(albTexel.z * 255.0f + 0.5f);
        // Alpha = surface-valid mask. 0 on primary-miss (sky) pixels so the
        // composite shader can suppress NRD's stale OUT_SPEC values there;
        // see composite_tonemap.frag.
        a4.w = gbufferWritten ? 255 : 0;
        surf2Dwrite<uchar4>(a4, surfaces.albedo, x * 4, y);
    }
    if (surfaces.emissive) {
        ushort4 p = packHalf4(emTexel);
        surf2Dwrite<ushort4>(p, surfaces.emissive, x * 8, y); // RGBA16F = 8B
    }

    // ── DLSS-RR specific surfaces (only set in Mode::DLSSRR) ─────
    if (surfaces.hdrColor) {
        // Final NaN/firefly guard before publishing the noisy color.
        float3 c = noisyColorAvg;
        if (isnan(c.x) || isnan(c.y) || isnan(c.z) ||
            isinf(c.x) || isinf(c.y) || isinf(c.z)) c = make_float3(0,0,0);
        c.x = fminf(fmaxf(c.x, 0.0f), 30.0f);
        c.y = fminf(fmaxf(c.y, 0.0f), 30.0f);
        c.z = fminf(fmaxf(c.z, 0.0f), 30.0f);
        // Add primary emissive avg too, since RR consumes a single combined
        // color. Note: noisyColorSum already added emissiveContrib above.
        ushort4 p = packHalf4(make_float4(c.x, c.y, c.z, 1.0f));
        surf2Dwrite<ushort4>(p, surfaces.hdrColor, x * 8, y); // RGBA16F = 8B
    }
    if (surfaces.worldNormalRoughness) {
        // RGBA16F: world-space shading normal in xyz (fp16), linear roughness in w.
        // DLSS-RR §3.4.3 — RGB16/32 float, packed roughness via Roughness_Mode_Packed.
        ushort4 p = packHalf4(make_float4(outPrimaryNormal.x,
                                          outPrimaryNormal.y,
                                          outPrimaryNormal.z,
                                          outPrimaryRoughness));
        surf2Dwrite<ushort4>(p, surfaces.worldNormalRoughness, x * 8, y);
    }
    if (surfaces.specAlbedo) {
        ushort4 p = packHalf4(make_float4(
            clampf(specAlbedoAvg.x, 0.0f, 4.0f),
            clampf(specAlbedoAvg.y, 0.0f, 4.0f),
            clampf(specAlbedoAvg.z, 0.0f, 4.0f),
            1.0f));
        surf2Dwrite<ushort4>(p, surfaces.specAlbedo, x * 8, y);
    }
    if (surfaces.specHitT) {
        // World-space scalar; NGX rejects NaN/inf. `rrSpecHitT` is a single
        // mirror-ray trace from the primary hit (§3.4.9 semantics).
        surf2Dwrite<float>(rrSpecHitT, surfaces.specHitT, x * 4, y);
    }
}

void launchPathTraceKernelSplit(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    SplitSurfaceOutputs surfaces,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerPixel)
{
    if (samplesPerPixel < 1) samplesPerPixel = 1;
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    pathTraceKernelSplit<<<grid, block>>>(
        scene, camera, surfaces, width, height, sampleIndex,
        enableEnvironment, maxBounces, samplesPerPixel);
    CUDA_CHECK(cudaGetLastError());
}

#endif // PATHTRACER_NRD_DLSS_ENABLED
