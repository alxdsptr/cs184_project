#include "render/PathTraceKernel.h"

#ifdef PATHTRACER_NRD_DLSS_ENABLED

#include "render/PathTraceHelpers.cuh"
#include "render/GBufferWriters.cuh"
#include "render/NEEHelpers.cuh"
#include "render/Finalizers.cuh"
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

    // Per-pixel running state across spp samples. NRD sees the mean, so
    // averaging N samples in-kernel reduces per-frame variance by ~N and
    // substantially cuts the single-sample bucket spikes that read as water
    // ripples after temporal filtering. The g-buffer slot is captured by
    // the first opaque hit only — NRD consumes one per pixel, not an average.
    SplitAccumState acc{};

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
    // `pg` mirrors what we'll capture into `acc.primary` on the first hit.
    bool haveGbuffer = false;
    PrimaryGBuffer pg{};
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
            pg.albedo    = albedo;
            pg.normal    = N;
            pg.roughness = mat.roughness;
            pg.hitPos    = hit.position;
            pg.rayDir    = ray.direction;
            pg.metallic  = mat.metallic;
            pg.viewZ     = nrd_helpers::computeViewZ(hit.position, camera.position, camera.forward);
            pg.mvPx      = nrd_helpers::computeMotionVectorPx(
                hit.position, camera.viewProjMatrix, camera.prevViewProjMatrix, width, height);
            // NDC depth for DLSS — RELAX wants linear viewZ (already above),
            // DLSS Super-Resolution wants post-perspective clip.z/clip.w.
            // mat4_transformPoint does the perspective divide so .z is NDC z in
            // [-1,1] (GL); remap to DLSS's [0,1] convention.
            {
                float3 ndc = mat4_transformPoint(camera.viewProjMatrix, hit.position);
                pg.ndcZ = clampf(ndc.z * 0.5f + 0.5f, 0.0f, 1.0f);
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

        splitAccumulateSppSample(acc, pathRadiance, emissiveContrib,
                                 haveGbuffer, pickedBucket, pg,
                                 bucketHitDist, bucketHitDistSet);
    } // end spp loop

    auto traceMirror = [&](float3 origin, float3 dir) -> float {
        if (!scene.d_bvhNodes || scene.totalTriangles == 0) return 1.0e4f;
        Ray mr;
        mr.origin = origin; mr.direction = dir;
        mr.tmin = 0.001f; mr.tmax = 1e30f;
        HitRecord mhit; mhit.t = mr.tmax;
        bool mDidHit = bvh_closestHit(
            mr, scene.d_bvhNodes, scene.bvhRootIndex,
            scene.d_positions, scene.d_indices, scene.d_materialIndices, mhit);
        // Miss (sky / outside scene) returns a long but finite distance — RR
        // uses hitT to derive the speed of the reflected feature; a 0 here
        // would be misread as "no reflection at all".
        return mDidHit ? mhit.t : 1.0e4f;
    };
    splitFinalizeAndWrite(acc, samplesPerPixel,
                          traceMirror, surfaces, x, y,
                          /*applyDlssRRMinAlbedoGuard=*/false,        // CUDA: no guard
                          SplitHdrClampPolicy::PerChannel30);          // CUDA: per-channel 30
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
