#include "render/PathTraceKernel.h"
#include "render/PathTraceHelpers.cuh"
#include "render/GBufferWriters.cuh"
#include "render/NEEHelpers.cuh"
#include "render/Finalizers.cuh"
#include "core/Halton.h"
#include "core/VolumeMedium.h"
#include "core/VolumeDevice.cuh"
#include "gpu/Random.h"
#include "gpu/BRDF.h"
#include "accel/BVH.h"
#include "accel/LightBVHSample.h"
#include "render/ReSTIR.h"
#include "render/VolumeNEE.cuh"
#include "render/VolumeShadowCuda.cuh"
#include "util/CudaCheck.h"

#include <cuda_fp16.h>
#include <surface_indirect_functions.h>

// ── Path Trace Kernel ────────────────────────────────────────
__global__ void pathTraceKernel(
    DeviceSceneData scene,
    CameraParams    camera,
    float4*         d_accumBuffer,
    float4*         d_outputBuffer,
    AuxBufferPtrs   auxBuffers,
    uint32_t        width,
    uint32_t        height,
    uint32_t        sampleIndex,
    bool            enableEnvironment,
    uint32_t        maxBounces,
    uint32_t        samplesPerPixel,
    PrimaryHitSurfaces gbuffer)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint32_t pixelIdx = y * width + x;

    if (samplesPerPixel < 1) samplesPerPixel = 1;

    // Sum of per-sample radiance over this frame's spp. Added to the accum
    // buffer as one batch; caller advances the sample counter by `spp`.
    float3 radianceSum = make_float3(0, 0, 0);
    bool gbufferWritten = false;

    // DLSSOnly publishes to `gbuffer.hdrColor`; in that path the sub-pixel
    // offset must exactly match `camera.jitterOffset` (Halton) — DLSS does
    // its own sub-pixel reconstruction and an extra per-sample random offset
    // just feeds it noise it interprets as motion, producing ghosting.
    const bool dlssPublish = (gbuffer.hdrColor != 0);

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
    // Unique RNG subseed per (pixel, frame, sample-in-frame).
    uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                              sampleIndex * 0x85EBCA6Bu + s);

    // ReSTIR (DI / GI) reservoirs are evaluated against the surface that
    // ReSTIRInitCandidates traced — that pass uses ONLY camera.jitterOffset
    // (halton, no per-sample random sub-pixel jitter). When ReSTIR is active
    // for this bounce/sample, we must hit the same surface here, otherwise
    // the reservoir's stored pHat/W is for a different (x,y) on the
    // primary-hit surface than the integrand evaluated below — the resulting
    // f * W mismatch produces a strong overexposure on glossy/reflective
    // surfaces where small sub-pixel jitters land on different geometry.
    bool restirActiveBounce0 =
        (s == 0) && (
            (scene.restirEnabled != 0 && scene.d_restirReservoirs != nullptr) ||
            (scene.restirGIEnabled != 0 && scene.d_restirGIIndirect != nullptr) ||
            (scene.restirPTEnabled != 0 && scene.d_restirPTIndirect != nullptr));

    float jx, jy;
    if (dlssPublish || restirActiveBounce0) {
        jx = camera.jitterOffset.x;
        jy = camera.jitterOffset.y;
    } else {
        // Native (no DLSS, no ReSTIR): per-sample random sub-pixel jitter for AA.
        jx = pcg32_float(rng) - 0.5f;
        jy = pcg32_float(rng) - 0.5f;
        jx += camera.jitterOffset.x;
        jy += camera.jitterOffset.y;
    }

    Ray ray = generateRay(x, y, width, height, camera, jx, jy);

    float3 throughput = make_float3(1, 1, 1);
    float3 radiance   = make_float3(0, 0, 0);
    bool firstBounce  = true;
    // True only when the previous bounce used a *delta* BSDF (perfect mirror /
    // glass refraction) whose sampling pdf is a Dirac. In that case we cannot
    // MIS the next emissive hit with light-sampling (the latter has pdf = 0 in
    // the delta direction), so weight defaults to 1. Cook-Torrance specular
    // lobes are NOT delta — they have a finite roughness and a real pdf — so
    // this flag stays false after GGX sampling, letting MIS do its job.
    bool lastBounceDelta = false;
    bool havePrevSurface = false;
    float3 prevSurfacePos = make_float3(0.0f, 0.0f, 0.0f);
    float prevBsdfPdf = 1.0f;

    for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
        HitRecord hit;
        hit.t = ray.tmax;

        bool didHit = false;
        if (scene.d_bvhNodes && scene.totalTriangles > 0) {
            didHit = bvh_closestHit(
                ray, scene.d_bvhNodes, scene.bvhRootIndex,
                scene.d_positions, scene.d_indices, scene.d_materialIndices,
                hit);
        }

        // ── Participating-medium integrator ──────────────────────
        // Bounded heterogeneous volume integration via delta tracking with
        // ratio tracking for through-transmittance. See core/VolumeDevice.cuh
        // for the algorithm details. Skipped entirely when the medium is
        // disabled or the majorant is zero — non-volumetric scenes pay only
        // a single branch.
        {
            float tMaxSegment = didHit ? hit.t : ray.tmax;
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

                        // Single-scatter NEE across all lights, with shadow rays
                        // through transmissive geometry + medium transmittance.
                        // Shared with PathTraceKernelSplit.cu and the OptiX raygens
                        // — see render/VolumeNEE.cuh.
                        float3 inScatter = volumeSingleScatterInScatter(
                            scene, scene.medium, mediumPos, wo, rng,
                            [&](float3 o, float3 d, float t) {
                                return cudaTraceTransmissiveShadow(scene, o, d, t);
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
                    // No scatter — apply ratio-tracked transmittance to the
                    // surface segment so the surface contribution is properly
                    // attenuated.
                    float3 T = volumeRatioTrack(
                        ray.origin, ray.direction, tEnter, tExit,
                        scene.medium, rng);
                    throughput = throughput * T;
                }
            }
        }

        if (!didHit) {
            if (enableEnvironment) {
                // Use SH shortcut for indirect non-delta bounces. Primary rays
                // and delta (mirror/glass) bounces still see the full HDR to
                // keep the directly-visible sky and mirror reflections sharp.
                bool shForThisBounce = (bounce > 0) && !lastBounceDelta;
                float3 envColor = sampleEnvironmentForBounce(
                    ray.direction, scene.envMapTex,
                    scene.d_shEnvCoeffs, scene.envUseSH != 0,
                    !shForThisBounce);
                // Clamp extremely bright HDR texels (sun etc.) to prevent
                // fireflies when a path through glass hits a hot pixel.
                envColor = clampEnvLuminance(envColor, 100.0f);
                radiance += throughput * envColor;
            }
            // Sky pixel: write sentinel g-buffer so DLSS / NRD treat it as far.
            if (firstBounce && !gbufferWritten) {
                writeSkyGBufferSentinel(
                    gbuffer.viewZ, gbuffer.motionVectors, gbuffer.ndcDepth, x, y);
                gbufferWritten = true;
                firstBounce = false;
            }
            break;
        }

        // Fetch material
        GPUMaterial mat;
        if (hit.materialIndex >= 0 && (uint32_t)hit.materialIndex < scene.materialCount)
            mat = scene.d_materials[hit.materialIndex];
        else {
            mat.albedo = make_float3(0.8f, 0.2f, 0.8f);
            mat.roughness = 0.5f;
            mat.metallic = 0.0f;
            mat.emission = make_float3(0,0,0);
            mat.emissionStrength = 0.0f;
            mat.useSpecularGlossiness = 0;
            mat.specularGlossAlphaIsGlossiness = 0;
            mat.useFBXCustomPacking = 0;
            mat.useFBXUEPacking = 0;
            mat.specularColor = make_float3(1.0f, 1.0f, 1.0f);
            mat.glossiness = 0.5f;
            mat.specularGlossTex = 0;
        }

        // Fetch vertex indices and barycentric coords for interpolation
        uint32_t triIdx = (uint32_t)hit.primitiveIndex;
        uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
        uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
        uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
        float baryU = hit.uv.x, baryV = hit.uv.y;
        float baryW = 1.0f - baryU - baryV;

        // Interpolate actual texture UVs from vertex data
        float2 texUV = make_float2(0.0f, 0.0f);
        if (scene.d_uvs) {
            float2 uv0 = scene.d_uvs[i0];
            float2 uv1 = scene.d_uvs[i1];
            float2 uv2 = scene.d_uvs[i2];
            texUV = uv0 * baryW + uv1 * baryU + uv2 * baryV;
        }

        // Apply albedo / MR / SG textures and resolve emissive. Mutates `mat`.
        float3 albedo;
        float3 emissiveColor;
        applyMaterialTextures(mat, texUV, albedo, emissiveColor);

        // Interpolate vertex normals if available
        float3 N = hit.shadingNormal;
        if (scene.d_normals) {
            float3 n0 = scene.d_normals[i0];
            float3 n1 = scene.d_normals[i1];
            float3 n2 = scene.d_normals[i2];
            N = normalize(n0 * baryW + n1 * baryU + n2 * baryV);
        }

        // Back-face flip FIRST (on the clean geometric/interpolated N), THEN
        // normal-map perturbation. Rationale: a strong normal map at grazing
        // angles can push the perturbed N across the horizon so
        // `dot(N_perturbed, rayDir) > 0` even though we genuinely hit the
        // front face. Flipping after perturbation would then mirror the
        // shading normal into the surface, producing black spots / inverted
        // highlights. Flipping first, based on the geometry, is unambiguous;
        // `applyNormalMap` below rebuilds T/B from the flipped N, so the whole
        // TBN frame comes along.
        //
        // Glass is excluded from the flip AND from normal mapping — refraction
        // stays on the true geometric surface.
        const bool isOpaque = (mat.transmission <= 0.0f);
        const bool backFacing = isOpaque && (dot(N, ray.direction) > 0.0f);
        if (backFacing) N = -N;

        // Capture the interpolated tangent handedness so the debug-viz branch
        // below can colour-code UV-seam drift. -2 = "no normal map applied".
        float debugHandedness = -2.0f;
        bool  debugNormalMapped = false;
        // Debug viz #3 (back-face-after-perturb) now detects a different —
        // but still interesting — failure: a perturbed N that flips to the
        // wrong side of the geometric surface even after we pre-aligned N.
        // This indicates `ts.z < 0` (bogus tangent-space normal, blue channel
        // below 0.5) or T/B drift strong enough that the reconstructed world
        // normal dives back through the horizon.
        float3 debugNPreFlip = N;
        const float3 debugRayDir = ray.direction;
        if (mat.normalTex != 0 && scene.d_tangents && isOpaque && scene.enableNormalMap) {
            float4 tangent;
            N = applyInterpolatedNormalMap(N, scene, i0, i1, i2, baryU, baryV, baryW,
                                           mat.normalTex, texUV, &tangent);
            debugHandedness   = tangent.w;
            debugNormalMapped = true;
            debugNPreFlip     = N;
        }

        // Debug: publish (position, perturbed N) into the sparse arrow grid.
        // Only the first sample of the first bounce writes, and only for the
        // pixel that lands on a grid cell (x % stride == 0 && y % stride == 0
        // ... matching how we size the buffer on the host). This way the
        // overlay shows deterministic positions — no flickering between spp.
        if (firstBounce && s == 0 && scene.d_debugArrows &&
            scene.debugArrowStride > 0)
        {
            uint32_t stride = (uint32_t)scene.debugArrowStride;
            if ((x % stride) == 0 && (y % stride) == 0) {
                uint32_t gx = x / stride;
                uint32_t gy = y / stride;
                uint32_t idx = gy * (uint32_t)scene.debugArrowWidth + gx;
                if ((int)gx < scene.debugArrowWidth &&
                    (int)gy < scene.debugArrowHeight)
                {
                    scene.d_debugArrows[2u * idx + 0u] =
                        make_float4(hit.position.x, hit.position.y, hit.position.z, 1.0f);
                    scene.d_debugArrows[2u * idx + 1u] =
                        make_float4(N.x, N.y, N.z, 0.0f);
                }
            }
        }

        // Primary-hit debug visualization. Runs only on the first bounce of
        // the first sample — any more would just average noise onto a
        // deterministic value. Writes a false-colour radiance and breaks the
        // path; the outer spp loop also bails because we set
        // `samplesPerPixel = 1` effectively by zeroing the remaining iters.
        if (firstBounce && scene.debugNormalViz != 0) {
            float3 debugColor = make_float3(0.0f, 0.0f, 0.0f);
            if (scene.debugNormalViz == 1) {
                // Perturbed world-space normal as RGB in [0,1].
                debugColor = N * 0.5f + make_float3(0.5f, 0.5f, 0.5f);
            } else if (scene.debugNormalViz == 2) {
                // Handedness visualization. Pixels without a normal map are
                // black (so you can tell them apart from the lit ones). With a
                // normal map: interpolated w ≈ +1 → green, ≈ -1 → blue,
                // anything in between → red intensity proportional to drift.
                if (!debugNormalMapped) {
                    debugColor = make_float3(0.0f, 0.0f, 0.0f);
                } else {
                    float w = debugHandedness;
                    float drift = fminf(fabsf(fabsf(w) - 1.0f), 1.0f);
                    if (drift < 0.05f) {
                        debugColor = (w >= 0.0f)
                            ? make_float3(0.0f, 1.0f, 0.0f)
                            : make_float3(0.0f, 0.0f, 1.0f);
                    } else {
                        // Red channel scales with drift; preserves sign hint
                        // as dim green/blue so seams remain readable.
                        float base = 0.15f;
                        debugColor = make_float3(
                            drift,
                            (w >= 0.0f) ? base : 0.0f,
                            (w <  0.0f) ? base : 0.0f);
                    }
                }
            } else if (scene.debugNormalViz == 3) {
                // Back-face-after-perturbation flag. Now that we flip BEFORE
                // perturbation, the geometric N is always front-facing at
                // this point; any `dot(N_perturbed, rayDir) > 0` here means
                // the normal map pushed the shading frame *back* through the
                // horizon — typically a bad tangent-space normal (ts.z < 0)
                // or extreme T/B drift. Red = problematic, green = fine,
                // grey = no normal map on this pixel.
                if (!debugNormalMapped) {
                    debugColor = make_float3(0.1f, 0.1f, 0.1f);
                } else {
                    float d = dot(debugNPreFlip, debugRayDir);
                    debugColor = (d > 0.0f)
                        ? make_float3(1.0f, 0.0f, 0.0f)
                        : make_float3(0.0f, 0.6f, 0.0f);
                }
            }
            radiance = debugColor;
            // Short-circuit: skip all remaining bounces for this path; the
            // spp-loop guard below skips further samples too (they would all
            // produce the exact same debug colour).
            if (!gbufferWritten) {
                if (gbuffer.viewZ) surf2Dwrite<float>(1.0e6f, gbuffer.viewZ, x * 4, y);
                gbufferWritten = true;
            }
            break;
        }

        // Write aux buffers from the first sample that produces a primary hit.
        // `firstBounce` guards within a path; `!gbufferWritten` guards across
        // the samples-per-pixel loop so we don't overwrite on subsequent spp.
        if (firstBounce) {
            if (!gbufferWritten) {
                float  viewZprim, clipCurrZ;
                float2 mvPx;
                computePrimaryReproject(camera, hit.position, hit.position,
                                        width, height,
                                        viewZprim, mvPx, clipCurrZ);

                if (auxBuffers.d_linearDepth)   auxBuffers.d_linearDepth[pixelIdx]   = viewZprim;
                if (auxBuffers.d_albedo)        auxBuffers.d_albedo[pixelIdx]        = albedo;
                if (auxBuffers.d_normal)        auxBuffers.d_normal[pixelIdx]        = N;
                if (auxBuffers.d_motionVectors) auxBuffers.d_motionVectors[pixelIdx] = mvPx;

                // DLSSOnly: also write to Vulkan-shared surfaces.
                writePrimaryGBufferSurfaces(
                    gbuffer.viewZ, gbuffer.motionVectors, gbuffer.ndcDepth,
                    x, y, viewZprim, mvPx, clipCurrZ);

                gbufferWritten = true;
            }
            firstBounce = false;
        }

        // ── Glass / transmissive material ───────────────────────
        if (mat.transmission > 0.0f) {
            GlassBounce gb = sampleGlassBounce(
                ray.direction, hit.position, N,
                hit.frontFace, mat.ior, albedo, rng);
            throughput     = throughput * gb.throughputMul;
            ray.origin     = gb.newOrigin;
            ray.direction  = gb.newDir;
            ray.tmin       = 0.001f;
            ray.tmax       = 1e30f;
            lastBounceDelta = true;
            prevSurfacePos  = hit.position;
            prevBsdfPdf     = 1.0f;
            havePrevSurface = true;

            // Flip N outward for aux buffers (denoiser expects outward normal).
            if (dot(N, ray.direction) > 0) N = -N;
            // Glass Russian roulette: terminate after many bounces to prevent
            // infinite TIR loops; do NOT boost throughput (delta BSDF doesn't
            // lose energy so boosting causes fireflies).
            if (bounce >= 6) {
                if (pcg32_float(rng) > 0.9f) break;
            }
            continue; // Skip NEE and opaque BRDF — glass is a delta BSDF
        }

        bool isEmissive = mat.emissionStrength > 0.0f &&
                          (emissiveColor.x > 0.0f || emissiveColor.y > 0.0f || emissiveColor.z > 0.0f);
        if (isEmissive) {
            float3 Le = emissiveColor * mat.emissionStrength;
            float weight = 1.0f;

            // MIS the emissive hit against the light-sampling strategy. pTri
            // must match the strategy used in NEE: if we sampled via the
            // light BVH at prevSurfacePos, the MIS inverse probability is the
            // BVH PDF of reaching `areaLightIndex` from there. Texture-emitter
            // triangles are registered as area lights too, so MIS applies
            // uniformly.
            if (bounce > 0 && havePrevSurface && !lastBounceDelta && scene.d_triangleAreaLightIndex) {
                int areaLightIndex = scene.d_triangleAreaLightIndex[(uint32_t)hit.primitiveIndex];
                if (areaLightIndex >= 0 && scene.d_areaLights && scene.areaLightCount > 0) {
                    GPUAreaLight light = scene.d_areaLights[areaLightIndex];
                    float pTri;
                    if (scene.d_lightBVHNodes && scene.d_lightIndexToSlot) {
                        uint32_t slot = scene.d_lightIndexToSlot[(uint32_t)areaLightIndex];
                        pTri = lightBVH_pdf(scene.d_lightBVHNodes,
                                            scene.lightBVHRootIndex,
                                            prevSurfacePos, slot);
                    } else {
                        pTri = light.weight / fmaxf(scene.areaLightTotalWeight, 1e-7f);
                    }
                    weight = computeEmissiveMISWeight(
                        light, hit.position, prevSurfacePos, prevBsdfPdf, pTri);
                }
            }

            radiance += throughput * Le * weight;

            // For textured emissives (e.g. light bulbs): continue the path so the
            // surface also reflects light and shows specular highlights / depth.
            // For pure area lights (uniform emission, no texture): terminate as before.
            if (mat.emissiveTex != 0) {
                // fall through to BRDF sampling below
            } else {
                break;
            }
        }

        // NEE callables — defined once per bounce, used by every light type
        // below. `traceShadow` adapts the CUDA SAH-BVH; mono kernels always
        // use the full Cook-Torrance mixture (primaryLobeOverride=false).
        float3 V = -ray.direction;
        auto traceShadow = [&](float3 o, float3 d, float dist) {
            return cudaTraceTransmissiveShadow(scene, o, d, dist);
        };
        auto neeBrdf = [&](float3 Ld, float NdotL) {
            return evalNEEBrdf(mat, N, V, Ld, albedo, /*primaryLobeOverride=*/false, 0);
        };
        auto neePdf = [&](float3 Ld, float NdotL) {
            return evalNEEBrdfPdf(mat, N, V, Ld, albedo, /*primaryLobeOverride=*/false, 0);
        };

        // Direct lighting from emissive triangle lights (next-event estimation).
        if (scene.d_areaLights && scene.areaLightCount > 0 &&
            scene.d_areaLightCDF && scene.areaLightTotalWeight > 0.0f) {
            uint32_t lightIndex = 0;
            float    pSelect  = 0.0f;       // unused when restirActive
            float    b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
            float    restirW = 0.0f;
            // ReSTIR is applied at the primary hit (bounce 0) only; bounces
            // ≥1 keep the existing light-BVH / CDF sampling pipeline because
            // the reservoir buffer is only populated for camera rays.
            bool restirActive = (scene.restirEnabled != 0) &&
                                (scene.d_restirReservoirs != nullptr) &&
                                (bounce == 0) && (s == 0);
            bool restirSkip = false;
            if (restirActive) {
                const ReSTIRReservoir* res =
                    reinterpret_cast<const ReSTIRReservoir*>(scene.d_restirReservoirs);
                ReSTIRReservoir r = res[pixelIdx];
                if (r.lightIndex == 0xFFFFFFFFu || r.W <= 0.0f || r.pHat <= 0.0f) {
                    // Empty reservoir (e.g. surface-culled) → skip NEE at this
                    // bounce entirely. Indirect light via BSDF sampling still
                    // works, so the image is unbiased on average.
                    restirSkip = true;
                } else {
                    lightIndex = r.lightIndex;
                    b1 = r.baryB1;
                    b2 = r.baryB2;
                    b0 = 1.0f - b1 - b2;
                    restirW = r.W;
                }
            } else {
                bool haveLight = false;
                if (scene.d_lightBVHNodes && scene.d_lightOrderedIndices) {
                    uint32_t slot = 0;
                    float    pdf  = 0.0f;
                    if (lightBVH_sample(scene.d_lightBVHNodes,
                                        scene.lightBVHRootIndex,
                                        hit.position, pcg32_float(rng),
                                        slot, pdf) && pdf > 0.0f) {
                        lightIndex = scene.d_lightOrderedIndices[slot];
                        pSelect    = pdf;
                        haveLight  = true;
                    }
                }
                if (!haveLight) {
                    lightIndex = sampleAreaLightIndex(
                        scene.d_areaLightCDF, scene.areaLightCount,
                        pcg32_float(rng));
                    pSelect = scene.d_areaLights[lightIndex].weight /
                              fmaxf(scene.areaLightTotalWeight, 1e-7f);
                }

                float r1 = pcg32_float(rng);
                float r2 = pcg32_float(rng);
                float su = sqrtf(r1);
                b0 = 1.0f - su;
                b1 = su * (1.0f - r2);
                b2 = su * r2;
            }

            if (!restirSkip) {
            GPUAreaLight light = scene.d_areaLights[lightIndex];
            // Mono ReSTIR-DI: cap=10 firefly clamp on the f*W estimator (a
            // near-grazing reservoir sample produces a single-frame ~50-lum
            // spike that survives the accumulator otherwise — M7 flash-and-decay).
            // Mono non-ReSTIR: no per-contribution clamp; relies on the
            // end-of-loop luminance clamp.
            float fireflyClamp = restirActive ? 10.0f : 0.0f;
            radiance += evalAreaLightNEEContribution(
                throughput, hit.position, N, light, b0, b1, b2, pSelect,
                restirActive, restirW,
                scene.medium, rng, traceShadow, neeBrdf, neePdf, fireflyClamp);
            } // end !restirSkip
        }

        // Point lights are delta emitters: BSDF-sampling can never hit them,
        // so they are always sampled independently (no MIS, no area-light
        // exclusivity). Bistro puts its main illumination on 4 point lights
        // in addition to emissive mesh geometry — gating this branch behind
        // "no area lights" would drop those entirely.
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
        // PT takes precedence (its postfix already contains GI's 1-bounce NEE
        // plus k more bounces' worth of light transport). Either branch adds
        // the pre-computed indirect estimate and skips continuation bounces;
        // the direct lighting at the primary hit was already added above.
        // Restricted to bounce==0 because the PT/GI buffer is populated for
        // camera rays only — higher bounces fall through to plain BSDF sampling.
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

        // BRDF sampling: Fresnel-weighted blend between diffuse and specular.
        // V was hoisted earlier for NEE.
        float specProb = materialSpecProb(mat, N, V, albedo);

        float3 newDir;

        if (pcg32_float(rng) < specProb) {
            // GGX importance sampling
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
            // Cook-Torrance specular lobe is NOT a delta — MIS is valid.
            lastBounceDelta = false;
        } else {
            // Cosine-weighted hemisphere sampling (diffuse)
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

        // Compute full mixture PDF for the sampled direction
        float pdf = materialMixturePdf(mat, N, V, newDir, specProb);
        if (pdf < 1e-7f) break;

        // Evaluate BRDF
        float3 brdf = materialBsdfEvaluate(mat, N, V, newDir, albedo);

        throughput = throughput * brdf * (NdotL_new / (pdf + 1e-7f));

        prevSurfacePos = hit.position;
            prevBsdfPdf = pdf;
        havePrevSurface = true;

        // Russian roulette
        if (bounce >= 2) {
            float lum = 0.2126f * throughput.x + 0.7152f * throughput.y + 0.0722f * throughput.z;
            float p = fminf(fmaxf(lum, 0.05f), 0.95f);
            if (pcg32_float(rng) >= p) break;
            throughput = throughput * (1.0f / p);
        }

        // Next ray
        ray.origin    = hit.position + N * 0.001f;
        ray.direction = newDir;
        ray.tmin      = 0.001f;
        ray.tmax      = 1e30f;
    }

        monoAccumulateSppSample(radiance, radianceSum);
    } // end spp loop

    monoFinalizePixel(radianceSum,
                      d_accumBuffer, d_outputBuffer, gbuffer.hdrColor,
                      pixelIdx, x, y, sampleIndex, samplesPerPixel);
}

void launchPathTraceKernel(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    float4* d_accumBuffer,
    float4* d_outputBuffer,
    AuxBufferPtrs auxBuffers,
    uint32_t width,
    uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerPixel,
    PrimaryHitSurfaces gbufferSurfaces)
{
    if (samplesPerPixel < 1) samplesPerPixel = 1;
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    pathTraceKernel<<<grid, block>>>(
        scene, camera, d_accumBuffer, d_outputBuffer, auxBuffers,
        width, height, sampleIndex, enableEnvironment, maxBounces, samplesPerPixel,
        gbufferSurfaces);
    CUDA_CHECK(cudaGetLastError());
}
