#include "render/PathTraceKernel.h"

#ifdef PATHTRACER_NRD_DLSS_ENABLED

#include "render/PathTraceHelpers.cuh"
#include "gpu/NRDHelpers.cuh"
#include "accel/BVH.h"
#include "gpu/Random.h"
#include "gpu/BRDF.h"
#include "util/CudaCheck.h"

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

// Primary-hit lobe-only BRDF evaluators. These are the diffuse/specular halves
// of `bsdfEvaluate` in PathTraceHelpers.cuh — keep them in sync.
__device__ inline float3 bsdfDiffuseLobe(
    const float3& N, const float3& V, const float3& L,
    const float3& albedo, float roughness, float metallic)
{
    (void)roughness;
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0,0,0);
    float3 H = normalize(V + L);
    float LdotH = fmaxf(dot(L, H), 0.0f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float3 F  = fresnelSchlick_local(LdotH, F0);
    float3 kd = (make_float3(1,1,1) - F) * (1.0f - metallic);
    return kd * albedo * (1.0f / M_PI_F);
}

__device__ inline float3 bsdfSpecularLobe(
    const float3& N, const float3& V, const float3& L,
    const float3& albedo, float roughness, float metallic)
{
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0,0,0);
    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float LdotH = fmaxf(dot(L, H), 0.0f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float3 F  = fresnelSchlick_local(LdotH, F0);
    float D_val = ggxD_local(NdotH, roughness);
    float alpha = roughness * roughness;
    float G_val = smithG1_GGX(NdotL, alpha) * smithG1_GGX(NdotV, alpha);
    return F * (D_val * G_val / (4.0f * NdotL * NdotV + 1e-7f));
}

// Per-contribution firefly clamp. RELAX is very sensitive to single-sample
// spikes: one 100x outlier survives the temporal filter for many frames and
// shows up as a shimmering bright speck (water-ripple look). We clamp each
// NEE / emissive contribution by luminance before adding it to the running
// path radiance, rather than only clamping the sum once at the end.
__device__ inline float3 clampFirefly(float3 c, float maxLum) {
    float lum = 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
    if (lum > maxLum && lum > 1e-7f) c = c * (maxLum / lum);
    return c;
}

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

    // G-buffer captured from the first sample that produces a primary opaque
    // hit. NRD only consumes one g-buffer per pixel, not an average.
    bool   gbufferWritten = false;
    float3 outPrimaryAlbedo   = make_float3(0, 0, 0);
    float3 outPrimaryNormal   = make_float3(0, 1, 0);
    float  outPrimaryRoughness = 1.0f;
    float  outPrimaryViewZ     = 0.0f;
    float2 outPrimaryMvPx      = make_float2(0.0f, 0.0f);

    if (samplesPerPixel < 1) samplesPerPixel = 1;

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
        // Unique RNG subseed per (pixel, frame, sample-in-frame).
        uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                                  sampleIndex * 0x85EBCA6Bu + s);

    float jx = pcg32_float(rng) - 0.5f;
    float jy = pcg32_float(rng) - 0.5f;
    jx += camera.jitterOffset.x;
    jy += camera.jitterOffset.y;

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
    int   pickedBucket = 0;       // 0 = diffuse, 1 = specular
    float bucketHitDist = 0.0f;    // world-space distance to first indirect surface
    bool  bucketHitDistSet = false;

    bool firstBounce = true;
    bool lastBounceSpecular = false;
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

        if (!didHit) {
            if (enableEnvironment) {
                float3 envColor = sampleEnvironment(ray.direction, scene.envMapTex);
                float envLum = 0.2126f*envColor.x + 0.7152f*envColor.y + 0.0722f*envColor.z;
                if (envLum > 20.0f) envColor = envColor * (20.0f / envLum);
                pathRadiance += clampFirefly(throughput * envColor, 10.0f);
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

        float3 albedo = mat.albedo;
        if (mat.albedoTex != 0) {
            float4 tc = tex2D<float4>(mat.albedoTex, texUV.x, texUV.y);
            albedo = make_float3(tc.x, tc.y, tc.z);
        }
        if (mat.metallicRoughTex != 0) {
            float4 mr = tex2D<float4>(mat.metallicRoughTex, texUV.x, texUV.y);
            mat.roughness = mat.roughness * mr.y;
            mat.metallic = mat.metallic * mr.z;
        }
        mat.roughness = fmaxf(mat.roughness, 0.045f);
        mat.metallic = clampf(mat.metallic, 0.0f, 1.0f);

        float3 emissiveColor = mat.emission;
        if (mat.emissiveTex != 0) {
            float4 et = tex2D<float4>(mat.emissiveTex, texUV.x, texUV.y);
            emissiveColor = make_float3(et.x, et.y, et.z);
        }

        float3 N = hit.shadingNormal;
        if (scene.d_normals) {
            N = normalize(scene.d_normals[i0] * baryW + scene.d_normals[i1] * baryU + scene.d_normals[i2] * baryV);
        }
        if (mat.transmission <= 0.0f) {
            if (dot(N, ray.direction) > 0) N = -N;
        }

        // Primary-hit g-buffer capture + bucket classification.
        if (firstBounce) {
            primaryAlbedo = albedo;
            primaryNormal = N;
            primaryRoughness = mat.roughness;
            primaryViewZ = nrd_helpers::computeViewZ(hit.position, camera.position, camera.forward);
            primaryMvPx = nrd_helpers::computeMotionVectorPx(
                hit.position, camera.viewProjMatrix, camera.prevViewProjMatrix, width, height);

            float3 V = -ray.direction;
            float specProb = computeSpecProb(N, V, albedo, mat.metallic);
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
            bool entering = hit.frontFace;
            float3 Nglass = entering ? N : -N;
            if (dot(Nglass, ray.direction) > 0.0f) Nglass = -Nglass;
            float eta = (entering ? 1.0f : mat.ior) / (entering ? mat.ior : 1.0f);
            float cosI = fmaxf(dot(-ray.direction, Nglass), 0.0f);
            float Fr = fresnelDielectric(cosI, eta);
            float3 newDir;
            if (pcg32_float(rng) < Fr) {
                newDir = normalize(ray.direction - Nglass * (2.0f * dot(ray.direction, Nglass)));
            } else if (!refractDir(ray.direction, Nglass, eta, newDir)) {
                newDir = normalize(ray.direction - Nglass * (2.0f * dot(ray.direction, Nglass)));
            }
            if (!entering) {
                float lum = 0.2126f*albedo.x + 0.7152f*albedo.y + 0.0722f*albedo.z;
                if (lum < 0.9f) throughput = throughput * albedo;
            }
            float3 off = (dot(newDir, Nglass) > 0.0f) ? Nglass : -Nglass;
            ray.origin = hit.position + off * 0.002f;
            ray.direction = newDir;
            ray.tmin = 0.001f; ray.tmax = 1e30f;
            lastBounceSpecular = true;
            prevSurfacePos = hit.position; prevBsdfPdf = 1.0f; havePrevSurface = true;
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
            if (bounce > 0 && havePrevSurface && !lastBounceSpecular && scene.d_triangleAreaLightIndex) {
                int ali = scene.d_triangleAreaLightIndex[(uint32_t)hit.primitiveIndex];
                if (ali >= 0 && scene.d_areaLights && scene.areaLightCount > 0) {
                    GPUAreaLight light = scene.d_areaLights[ali];
                    float3 toL = hit.position - prevSurfacePos;
                    float d2 = fmaxf(dot(toL, toL), 1e-6f);
                    float3 wi = normalize(toL);
                    float lNdot = fmaxf(dot(light.normal, -wi), 0.0f);
                    if (lNdot > 0.0f) {
                        float pTri = light.weight / fmaxf(scene.areaLightTotalWeight, 1e-7f);
                        float pArea = pTri / fmaxf(light.area, 1e-7f);
                        float pLight = pArea * d2 / fmaxf(lNdot, 1e-7f);
                        weight = powerHeuristic(prevBsdfPdf, pLight);
                    }
                }
            }
            if (bounce == 0) {
                emissiveContrib = Le * weight;    // Primary emissive — separate image.
            } else {
                pathRadiance += clampFirefly(throughput * Le * weight, 10.0f);
            }
            if (mat.emissiveTex == 0) break;
        }

        // NEE area lights.
        if (scene.d_areaLights && scene.areaLightCount > 0 &&
            scene.d_areaLightCDF && scene.areaLightTotalWeight > 0.0f)
        {
            uint32_t li = sampleAreaLightIndex(scene.d_areaLightCDF, scene.areaLightCount, pcg32_float(rng));
            GPUAreaLight light = scene.d_areaLights[li];
            float r1 = pcg32_float(rng), r2 = pcg32_float(rng);
            float su = sqrtf(r1);
            float3 lp = light.v0 * (1.0f - su) + (light.v0 + light.e1) * (su * (1.0f - r2)) + (light.v0 + light.e2) * (su * r2);
            float3 toL = lp - hit.position;
            float d2 = fmaxf(dot(toL, toL), 1e-6f);
            float d = sqrtf(d2);
            float3 Ld = toL * (1.0f / d);
            float NdotL = fmaxf(dot(N, Ld), 0.0f);
            float lNdot = fmaxf(dot(light.normal, -Ld), 0.0f);
            if (NdotL > 0.0f && lNdot > 0.0f) {
                bool occluded = false;
                float3 st = make_float3(1,1,1);
                if (scene.d_bvhNodes && scene.totalTriangles > 0) {
                    Ray sr;
                    sr.origin = hit.position + N * 0.001f;
                    sr.direction = Ld;
                    sr.tmin = 0.001f; sr.tmax = fmaxf(d - 0.002f, 0.001f);
                    for (int s = 0; s < 8; s++) {
                        HitRecord sh; sh.t = sr.tmax;
                        if (!bvh_closestHit(sr, scene.d_bvhNodes, scene.bvhRootIndex,
                                            scene.d_positions, scene.d_indices, scene.d_materialIndices, sh)) break;
                        GPUMaterial sm;
                        if (sh.materialIndex >= 0 && (uint32_t)sh.materialIndex < scene.materialCount)
                            sm = scene.d_materials[sh.materialIndex];
                        else { occluded = true; break; }
                        if (sm.transmission > 0.0f) {
                            float salum = 0.2126f*sm.albedo.x + 0.7152f*sm.albedo.y + 0.0722f*sm.albedo.z;
                            if (salum < 0.9f) st = st * sm.albedo;
                            sr.origin = sh.position + Ld * 0.002f;
                            sr.tmax = fmaxf(d - length(sr.origin - (hit.position + N*0.001f)) - 0.002f, 0.001f);
                        } else { occluded = true; break; }
                    }
                }
                float slum = 0.2126f*st.x + 0.7152f*st.y + 0.0722f*st.z;
                if (!occluded && slum > 1e-6f) {
                    float pTri = light.weight / scene.areaLightTotalWeight;
                    float pArea = pTri / fmaxf(light.area, 1e-7f);
                    float pdfOmega = pArea * d2 / fmaxf(lNdot, 1e-7f);
                    float3 V = -ray.direction;
                    float3 brdf;
                    float pdfBs;
                    if (primaryLobeOverride) {
                        if (pickedBucket == 0) {
                            brdf  = bsdfDiffuseLobe(N, V, Ld, albedo, mat.roughness, mat.metallic);
                            pdfBs = bsdfDiffusePdf(NdotL);
                        } else {
                            brdf  = bsdfSpecularLobe(N, V, Ld, albedo, mat.roughness, mat.metallic);
                            pdfBs = bsdfSpecularPdf(N, V, Ld, mat.roughness);
                        }
                    } else {
                        brdf = bsdfEvaluate(N, V, Ld, albedo, mat.roughness, mat.metallic);
                        float spProb = computeSpecProb(N, V, albedo, mat.metallic);
                        pdfBs = bsdfMixturePdf(N, V, Ld, mat.roughness, spProb);
                    }
                    float w = powerHeuristic(pdfOmega, pdfBs);
                    float3 neeContrib = throughput * st * brdf * light.emission * (NdotL / fmaxf(pdfOmega, 1e-7f)) * w;
                    pathRadiance += clampFirefly(neeContrib, 10.0f);
                }
            }
        } else if (scene.d_pointLights && scene.pointLightCount > 0) {
            float3 V = -ray.direction;
            float3 direct = make_float3(0,0,0);
            for (uint32_t li = 0; li < scene.pointLightCount; li++) {
                GPUPointLight light = scene.d_pointLights[li];
                float3 toL = light.position - hit.position;
                float d2 = fmaxf(dot(toL, toL), 1e-6f);
                float d = sqrtf(d2);
                float3 Ld = toL * (1.0f / d);
                float NdotL = fmaxf(dot(N, Ld), 0.0f);
                if (NdotL <= 0.0f) continue;
                bool occ = false;
                float3 st = make_float3(1,1,1);
                if (scene.d_bvhNodes && scene.totalTriangles > 0) {
                    Ray sr; sr.origin = hit.position + N * 0.001f; sr.direction = Ld;
                    sr.tmin = 0.001f; sr.tmax = fmaxf(d - 0.002f, 0.001f);
                    for (int s = 0; s < 8; s++) {
                        HitRecord sh; sh.t = sr.tmax;
                        if (!bvh_closestHit(sr, scene.d_bvhNodes, scene.bvhRootIndex,
                                            scene.d_positions, scene.d_indices, scene.d_materialIndices, sh)) break;
                        GPUMaterial sm;
                        if (sh.materialIndex >= 0 && (uint32_t)sh.materialIndex < scene.materialCount)
                            sm = scene.d_materials[sh.materialIndex];
                        else { occ = true; break; }
                        if (sm.transmission > 0.0f) {
                            float sl = 0.2126f*sm.albedo.x + 0.7152f*sm.albedo.y + 0.0722f*sm.albedo.z;
                            if (sl < 0.9f) st = st * sm.albedo;
                            sr.origin = sh.position + Ld * 0.002f;
                            sr.tmax = fmaxf(d - length(sr.origin - (hit.position + N*0.001f)) - 0.002f, 0.001f);
                        } else { occ = true; break; }
                    }
                }
                float slum = 0.2126f*st.x + 0.7152f*st.y + 0.0722f*st.z;
                if (occ || slum < 1e-6f) continue;
                float attenDen = light.constantAttenuation + light.linearAttenuation*d + light.quadraticAttenuation*d2;
                float atten = 1.0f / fmaxf(attenDen, 1e-4f);
                float3 Li = light.color * (light.intensity * atten);
                float3 brdf;
                if (primaryLobeOverride) {
                    brdf = (pickedBucket == 0)
                        ? bsdfDiffuseLobe(N, V, Ld, albedo, mat.roughness, mat.metallic)
                        : bsdfSpecularLobe(N, V, Ld, albedo, mat.roughness, mat.metallic);
                } else {
                    brdf = bsdfEvaluate(N, V, Ld, albedo, mat.roughness, mat.metallic);
                }
                direct += clampFirefly(brdf * st * Li * NdotL, 10.0f);
            }
            pathRadiance += throughput * direct;
        }

        // BRDF sampling for the next bounce. At the primary hit the lobe is
        // forced to match `pickedBucket`; at subsequent hits we use the full
        // mixture since the bucket is already locked in.
        float3 V = -ray.direction;
        float specProb = computeSpecProb(N, V, albedo, mat.metallic);
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
            lastBounceSpecular = true;
        } else {
            float u1 = pcg32_float(rng), u2 = pcg32_float(rng);
            float dummy;
            float3 lD = sampleCosineHemisphere(u1, u2, dummy);
            float3 T, B; buildONB(N, T, B);
            newDir = localToWorld(lD, T, N, B);
            lastBounceSpecular = false;
        }
        float NdotLn = dot(N, newDir);
        if (NdotLn < 1e-6f) break;
        float pdf;
        float3 brdf;
        if (primaryLobeOverride) {
            if (pickedBucket == 0) {
                pdf  = bsdfDiffusePdf(NdotLn);
                brdf = bsdfDiffuseLobe(N, V, newDir, albedo, mat.roughness, mat.metallic);
            } else {
                pdf  = bsdfSpecularPdf(N, V, newDir, mat.roughness);
                brdf = bsdfSpecularLobe(N, V, newDir, albedo, mat.roughness, mat.metallic);
            }
        } else {
            pdf  = bsdfMixturePdf(N, V, newDir, mat.roughness, specProb);
            brdf = bsdfEvaluate(N, V, newDir, albedo, mat.roughness, mat.metallic);
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
        }
        // G-buffer: first sample that produced a primary hit wins. Averaging
        // normals / viewZ across samples would soften silhouettes and break
        // NRD's disocclusion test, so we don't.
        if (!gbufferWritten && haveGbuffer) {
            outPrimaryAlbedo    = primaryAlbedo;
            outPrimaryNormal    = primaryNormal;
            outPrimaryRoughness = primaryRoughness;
            outPrimaryViewZ     = primaryViewZ;
            outPrimaryMvPx      = primaryMvPx;
            gbufferWritten = true;
        }
    } // end spp loop

    // Average per-pixel radiance over the samples taken.
    float invSpp = 1.0f / (float)samplesPerPixel;
    float3 demodDiffAvg = demodDiffSum * invSpp;
    float3 demodSpecAvg = demodSpecSum * invSpp;
    float3 emissiveAvg  = emissiveSum  * invSpp;
    // HitDist: average only over samples that actually filled the bucket, so
    // pixels where one sample went diffuse and the others specular don't bias
    // the diff-bucket hitT toward zero.
    float diffHitAvg = diffHitCount > 0 ? (diffHitSum / (float)diffHitCount) : 0.0f;
    float specHitAvg = specHitCount > 0 ? (specHitSum / (float)specHitCount) : 0.0f;

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
        a4.w = 255;
        surf2Dwrite<uchar4>(a4, surfaces.albedo, x * 4, y);
    }
    if (surfaces.emissive) {
        ushort4 p = packHalf4(emTexel);
        surf2Dwrite<ushort4>(p, surfaces.emissive, x * 8, y); // RGBA16F = 8B
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
