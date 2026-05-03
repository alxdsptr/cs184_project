#include "render/ReSTIRGI.h"
#include "render/ReSTIRGIDevice.cuh"
#include "core/Math.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "gpu/Sampling.h"
#include "accel/BVH.h"
#include "accel/LightBVHSample.h"
#include "util/CudaCheck.h"

#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────
// ReSTIR GI — primary-hit indirect lighting via reservoir resampling.
//
// Pipeline (one frame):
//   1. Initial candidates : at every pixel, build the visible-point surface,
//      sample one BSDF direction, trace it with the CUDA BVH, and capture
//      the path-vertex (sample point, normal, outgoing radiance Lo). Lo is
//      computed as `emission(x_s) + NEE(x_s)` so we resample on a
//      visibility-tested 1-bounce indirect estimate.
//   2. Temporal reuse     : combine prev-frame reservoir at the reprojected
//      pixel, with normal/depth gates and a Bitterli-style M cap.
//   3. Spatial reuse      : combine k disk-sampled neighbor reservoirs with
//      the proper Jacobian for change-of-visible-point.
//   4. Shade              : materialize the reservoir into a per-pixel
//      indirect-radiance buffer the path tracer adds at primary-hit shading.
//
// Visibility is NOT re-tested in spatial / shading — common simplification
// that introduces a small bias for long-distance reuse but is typically
// invisible against the variance reduction.
// ─────────────────────────────────────────────────────────────────────────

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

namespace {

// Identical math to the main kernel's generateRay (we can't share that
// definition because PathTraceKernel.cu's helpers are static / TU-local).
__device__ inline Ray giGenerateRay(
    uint32_t x, uint32_t y, uint32_t width, uint32_t height,
    const CameraParams& cam, float jitterX, float jitterY)
{
    float u = ((float)x + 0.5f + jitterX) / (float)width;
    float v = ((float)y + 0.5f + jitterY) / (float)height;
    float ndcX = 2.0f * u - 1.0f;
    float ndcY = 1.0f - 2.0f * v;
    float tanHalf = tanf(cam.fovYRadians * 0.5f);
    float px = ndcX * cam.aspectRatio * tanHalf;
    float py = ndcY * tanHalf;
    float3 dir = normalize(cam.forward + cam.right * px + cam.up * py);
    Ray ray;
    ray.origin    = cam.position;
    ray.direction = dir;
    ray.tmin      = 0.001f;
    ray.tmax      = 1e30f;
    return ray;
}

// Cheap procedural sky fallback — only used when no env map is bound. Same
// formula as the main kernel.
__device__ inline float3 giProceduralSky(float3 dir) {
    float t = 0.5f * (dir.y + 1.0f);
    float3 skyTop = make_float3(0.5f, 0.7f, 1.0f);
    float3 skyBot = make_float3(1.0f, 1.0f, 1.0f);
    return lerp(skyBot, skyTop, t) * 0.8f;
}

__device__ inline float3 giSampleEnvironment(
    float3 dir, cudaTextureObject_t envMap)
{
    if (envMap != 0) {
        float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
        float phi   = atan2f(dir.z, dir.x);
        float u = (phi + M_PI_F) * (0.5f / M_PI_F);
        float v = theta / M_PI_F;
        float4 texel = tex2D<float4>(envMap, u, v);
        return make_float3(texel.x, texel.y, texel.z);
    }
    return giProceduralSky(dir);
}

// Material-aware specular-probability, evaluation, and PDF — matched to the
// main path tracer's helpers (cf. PathTraceHelpers.cuh). Defined here as
// __device__ to keep ReSTIR GI fully self-contained.
__device__ inline float giComputeSpecProb(
    const float3& N, const float3& V, const float3& albedo, float metallic)
{
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float t = 1.0f - fminf(fmaxf(NdotV, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    float3 F = F0 + (make_float3(1,1,1) - F0) * t5;
    float specW  = restirLuminance(F);
    float3 kd    = (make_float3(1,1,1) - F) * (1.0f - metallic);
    float diffW  = restirLuminance(kd * albedo);
    float p = specW / fmaxf(specW + diffW, 1e-7f);
    return fminf(fmaxf(p, 0.1f), 0.9f);
}

__device__ inline float giDiffusePdf(float NdotL) {
    return fmaxf(NdotL, 0.0f) * (1.0f / M_PI_F);
}

__device__ inline float giSpecularPdf(
    const float3& N, const float3& V, const float3& L, float roughness)
{
    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float VdotH = fmaxf(dot(V, H), 0.0f);
    if (NdotH <= 0.0f || VdotH <= 0.0f) return 0.0f;
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    float D_val = a2 / (M_PI_F * denom * denom + 1e-14f);
    return D_val * NdotH / (4.0f * VdotH + 1e-7f);
}

__device__ inline float giMixturePdf(
    bool pureDiffuse,
    const float3& N, const float3& V, const float3& L,
    float roughness, float specProb)
{
    float diffPdf = giDiffusePdf(dot(N, L));
    if (pureDiffuse) return diffPdf;
    float specPdf = giSpecularPdf(N, V, L, roughness);
    return specProb * specPdf + (1.0f - specProb) * diffPdf;
}

// Sample a BSDF direction at the visible point. Returns (newDir, pdf).
__device__ inline bool giSampleBsdfDir(
    const ReSTIRSurface& s, uint32_t& rng,
    float3& outDir, float& outPdf)
{
    bool pureDiffuse = (s.pureDiffuse != 0u);
    float specProb = pureDiffuse ? 0.0f : s.specProb;
    float u = pcg32_float(rng);
    float3 dir;
    if (!pureDiffuse && u < specProb) {
        // GGX importance sample.
        float a = s.roughness * s.roughness;
        float u1 = pcg32_float(rng);
        float u2 = pcg32_float(rng);
        float cosTheta = sqrtf((1.0f - u1) / (1.0f + (a*a - 1.0f) * u1 + 1e-7f));
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = 2.0f * M_PI_F * u2;
        float3 localH = make_float3(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi));
        float3 T, B;
        buildONB(s.normal, T, B);
        float3 H = localToWorld(localH, T, s.normal, B);
        // Reflect viewDir around H (V points AWAY from the surface; reflect
        // the *incoming* direction = -V to get the outgoing scatter dir).
        float3 inDir = -s.viewDir;
        dir = inDir - H * (2.0f * dot(inDir, H));
        dir = normalize(dir);
    } else {
        float u1 = pcg32_float(rng);
        float u2 = pcg32_float(rng);
        float dummy;
        float3 local = sampleCosineHemisphere(u1, u2, dummy);
        float3 T, B;
        buildONB(s.normal, T, B);
        dir = localToWorld(local, T, s.normal, B);
    }
    if (dot(s.normal, dir) <= 1e-6f) return false;
    outDir = dir;
    outPdf = giMixturePdf(pureDiffuse, s.normal, s.viewDir, dir,
                          s.roughness, specProb);
    return outPdf > 1e-7f;
}

// Sample one area-light contribution at a sample point (one NEE shadow ray)
// to seed Lo with direct lighting at the indirect bounce. Skipped when no
// area lights or no light BVH is built.
__device__ inline float3 giDirectLightingAtSample(
    const DeviceSceneData& scene,
    const float3& pos, const float3& normal,
    const float3& albedo, float roughness, float metallic, bool pureDiffuse,
    const float3& viewDir,
    uint32_t& rng)
{
    if (!scene.d_areaLights || scene.areaLightCount == 0 ||
        !scene.d_lightBVHNodes) return make_float3(0, 0, 0);

    uint32_t slot = 0;
    float    pSelect = 0.0f;
    if (!lightBVH_sample(scene.d_lightBVHNodes, scene.lightBVHRootIndex,
                         pos, pcg32_float(rng), slot, pSelect) ||
        !(pSelect > 0.0f))
        return make_float3(0, 0, 0);
    uint32_t lightIdx = scene.d_lightOrderedIndices[slot];
    GPUAreaLight light = scene.d_areaLights[lightIdx];

    float r1 = pcg32_float(rng);
    float r2 = pcg32_float(rng);
    float su = sqrtf(r1);
    float b0 = 1.0f - su;
    float b1 = su * (1.0f - r2);
    float b2 = su * r2;
    float3 lp = light.v0 * b0 + (light.v0 + light.e1) * b1 + (light.v0 + light.e2) * b2;
    float3 toL = lp - pos;
    float  d2  = fmaxf(dot(toL, toL), 1e-6f);
    float  d   = sqrtf(d2);
    float3 L   = toL * (1.0f / d);
    float NdotL = fmaxf(dot(normal, L), 0.0f);
    float lightCos = fmaxf(dot(light.normal, -L), 0.0f);
    if (NdotL <= 0.0f || lightCos <= 0.0f) return make_float3(0, 0, 0);

    // Shadow ray (opaque-only, no glass transparency tracking — keeps the
    // GI sample cheap; shadow inaccuracy at the indirect bounce is invisible
    // once accumulated). bvh_anyHit pre-shrinks tmax by 1e-4 so we won't
    // self-intersect the emitter triangle at distance `d`.
    if (scene.d_bvhNodes && scene.totalTriangles > 0) {
        float3 shadowOrigin = pos + normal * 0.001f;
        if (bvh_anyHit(shadowOrigin, lp,
                       scene.d_bvhNodes, scene.bvhRootIndex,
                       scene.d_positions, scene.d_indices))
            return make_float3(0, 0, 0);
    }

    // Inline BRDF eval — same Cook-Torrance + diffuse mixture as
    // restirEvalBrdf, but constructed from raw material params (the sample
    // point isn't a ReSTIRSurface).
    float3 Le;
    if (light.emissiveTex == 0) {
        Le = light.emission;
    } else {
        float texU = light.uv0.x * b0 + light.uv1.x * b1 + light.uv2.x * b2;
        float texV = light.uv0.y * b0 + light.uv1.y * b1 + light.uv2.y * b2;
        float4 et = tex2D<float4>(light.emissiveTex, texU, texV);
        Le = make_float3(et.x, et.y, et.z) * light.emission;
    }

    float3 brdf;
    if (pureDiffuse) {
        brdf = albedo * (1.0f / M_PI_F);
    } else {
        // Build a temp ReSTIRSurface to call restirEvalBrdf — keeps the BRDF
        // model identical to what we use for pHat evaluation.
        ReSTIRSurface tmp{};
        tmp.position = pos;
        tmp.normal   = normal;
        tmp.albedo   = albedo;
        tmp.roughness = fmaxf(roughness, 0.04f);
        tmp.metallic  = metallic;
        tmp.viewDir   = viewDir;
        tmp.pureDiffuse = 0u;
        brdf = restirEvalBrdf(tmp, L);
    }

    float pTri  = pSelect;
    float pArea = pTri / fmaxf(light.area, 1e-7f);
    float pdfOmega = pArea * d2 / fmaxf(lightCos, 1e-7f);
    return brdf * Le * (NdotL / fmaxf(pdfOmega, 1e-7f));
}

} // anonymous namespace

// ── Kernel 1: initial candidate generation ────────────────────────────
// Build the visible-point surface; trace one BSDF-sampled bounce; capture
// (sample point, sample normal, sample outgoing radiance); RIS into a
// fresh reservoir. This is essentially RIS with M=1 candidate — we lean on
// temporal+spatial reuse for the "many samples" effect.
__global__ void kReSTIRGI_InitCandidates(
    DeviceSceneData scene,
    CameraParams    camera,
    GIReservoir*    outReservoirs,
    ReSTIRSurface*  outSurfaces,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    int      enableEnvironment)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    // Independent RNG stream from the DI pass.
    uint32_t rng = pcg32_seed(pixelIdx * 0x517CC1B7u + sampleIndex,
                              sampleIndex * 0xCAFEF00Du + 0x67u);

    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;
    Ray ray = giGenerateRay(x, y, width, height, camera, jx, jy);

    GIReservoir r; giReservoirReset(r);
    ReSTIRSurface surf{};
    surf.valid = 0.0f;

    HitRecord hit; hit.t = 1e30f;
    bool didHit = false;
    if (scene.d_bvhNodes && scene.totalTriangles > 0) {
        didHit = bvh_closestHit(ray, scene.d_bvhNodes, scene.bvhRootIndex,
                                scene.d_positions, scene.d_indices,
                                scene.d_materialIndices, hit);
    }
    if (!didHit || hit.materialIndex < 0 ||
        (uint32_t)hit.materialIndex >= scene.materialCount)
    {
        outReservoirs[pixelIdx] = r;
        outSurfaces[pixelIdx]   = surf;
        return;
    }

    GPUMaterial mat = scene.d_materials[hit.materialIndex];
    uint32_t i0 = scene.d_indices[hit.primitiveIndex * 3 + 0];
    uint32_t i1 = scene.d_indices[hit.primitiveIndex * 3 + 1];
    uint32_t i2 = scene.d_indices[hit.primitiveIndex * 3 + 2];
    float b1 = hit.uv.x, b2 = hit.uv.y;
    float b0 = 1.0f - b1 - b2;
    float3 N = normalize(scene.d_normals[i0] * b0 +
                         scene.d_normals[i1] * b1 +
                         scene.d_normals[i2] * b2);
    if (dot(N, ray.direction) > 0.0f) N = -N;

    float2 uv = scene.d_uvs[i0] * b0 + scene.d_uvs[i1] * b1 + scene.d_uvs[i2] * b2;
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

    surf.position    = hit.position;
    surf.normal      = N;
    surf.albedo      = albedo;
    surf.roughness   = fmaxf(mat.roughness, 0.04f);
    surf.metallic    = mat.metallic;
    surf.pureDiffuse = mat.pureDiffuse ? 1u : 0u;
    surf.viewDir     = -ray.direction;
    surf.valid       = 1.0f;
    surf.specProb    = giComputeSpecProb(N, surf.viewDir, albedo, mat.metallic);

    // Reprojection coordinate for next-frame temporal reuse.
    float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hit.position);
    surf.prevPixel  = make_float2((clipPrev.x + 1.0f) * 0.5f * width,
                                   (1.0f - clipPrev.y) * 0.5f * height);

    // Sample a BSDF direction.
    float3 wi;
    float  pdfBsdf = 0.0f;
    if (!giSampleBsdfDir(surf, rng, wi, pdfBsdf)) {
        outReservoirs[pixelIdx] = r;
        outSurfaces[pixelIdx]   = surf;
        return;
    }

    // Trace the indirect ray.
    Ray sec;
    sec.origin    = hit.position + N * 0.001f;
    sec.direction = wi;
    sec.tmin      = 0.001f;
    sec.tmax      = 1e30f;
    HitRecord hit2; hit2.t = 1e30f;
    bool didHit2 = false;
    if (scene.d_bvhNodes && scene.totalTriangles > 0) {
        didHit2 = bvh_closestHit(sec, scene.d_bvhNodes, scene.bvhRootIndex,
                                 scene.d_positions, scene.d_indices,
                                 scene.d_materialIndices, hit2);
    }

    bool   hasSample = false;
    bool   isEnvSample = false;
    float3 samplePos    = make_float3(0, 0, 0);
    float3 sampleNormal = make_float3(0, 1, 0);
    float3 Lo           = make_float3(0, 0, 0);

    if (!didHit2) {
        if (enableEnvironment) {
            float3 envColor = giSampleEnvironment(wi, scene.envMapTex);
            float envLum = restirLuminance(envColor);
            const float clampLum = 100.0f;
            if (envLum > clampLum) envColor = envColor * (clampLum / envLum);
            isEnvSample = true;
            samplePos   = wi;          // direction
            sampleNormal = -wi;        // unused but keep something sensible
            Lo = envColor;
            hasSample = (envLum > 0.0f);
        }
    } else if (hit2.materialIndex >= 0 &&
               (uint32_t)hit2.materialIndex < scene.materialCount) {
        GPUMaterial mat2 = scene.d_materials[hit2.materialIndex];
        uint32_t j0 = scene.d_indices[hit2.primitiveIndex * 3 + 0];
        uint32_t j1 = scene.d_indices[hit2.primitiveIndex * 3 + 1];
        uint32_t j2 = scene.d_indices[hit2.primitiveIndex * 3 + 2];
        float c1 = hit2.uv.x, c2 = hit2.uv.y;
        float c0 = 1.0f - c1 - c2;
        float3 N2 = normalize(scene.d_normals[j0] * c0 +
                              scene.d_normals[j1] * c1 +
                              scene.d_normals[j2] * c2);
        if (dot(N2, wi) > 0.0f) N2 = -N2;
        float2 uv2 = scene.d_uvs[j0] * c0 + scene.d_uvs[j1] * c1 + scene.d_uvs[j2] * c2;
        float3 albedo2 = mat2.albedo;
        if (mat2.albedoTex != 0) {
            float4 t = tex2D<float4>(mat2.albedoTex, uv2.x, uv2.y);
            albedo2 = albedo2 * make_float3(t.x, t.y, t.z);
        }
        if (mat2.metallicRoughTex != 0) {
            float4 mrT = tex2D<float4>(mat2.metallicRoughTex, uv2.x, uv2.y);
            mat2.roughness *= mrT.y;
            mat2.metallic  *= mrT.z;
        }
        // Outgoing radiance toward the visible point = emission + 1-bounce NEE.
        float3 emis = mat2.emission * mat2.emissionStrength;
        if (mat2.emissiveTex != 0) {
            float4 et = tex2D<float4>(mat2.emissiveTex, uv2.x, uv2.y);
            emis = make_float3(et.x, et.y, et.z) * mat2.emissionStrength;
        }
        float3 viewDir2 = -wi;  // toward the visible point
        float3 direct = giDirectLightingAtSample(
            scene, hit2.position, N2, albedo2,
            fmaxf(mat2.roughness, 0.04f), mat2.metallic,
            mat2.pureDiffuse != 0, viewDir2, rng);

        Lo = emis + direct;
        samplePos    = hit2.position;
        sampleNormal = N2;
        isEnvSample  = false;
        hasSample = (restirLuminance(Lo) > 0.0f);
    }

    if (hasSample) {
        // Build a temporary reservoir to evaluate pHat through the shared
        // helper (needs a populated GIReservoir to compute brdf+geom).
        GIReservoir cand{};
        cand.visiblePos     = surf.position;
        cand.visibleNormal  = surf.normal;
        cand.samplePos      = samplePos;
        cand.sampleNormal   = sampleNormal;
        cand.sampleRadiance = Lo;
        cand.isEnv          = isEnvSample ? 1u : 0u;
        cand.valid          = 1u;
        float3 wiOut;
        float pHat = giEvalTargetPdf(surf, cand, wiOut);
        // Source pdf is the BSDF pdf in solid-angle measure at the visible
        // point; pHat is also in solid-angle measure → ratio is dimensionally
        // consistent.
        float wCand = (pdfBsdf > 0.0f) ? (pHat / pdfBsdf) : 0.0f;
        float wSum = 0.0f;
        giReservoirUpdate(r, wSum,
                          surf.position, surf.normal,
                          isEnvSample, samplePos, sampleNormal, Lo,
                          pHat, wCand, pcg32_float(rng));
        giReservoirFinalize(r, wSum);
    }

    outReservoirs[pixelIdx] = r;
    outSurfaces[pixelIdx]   = surf;
}

// ── Kernel 2: temporal reuse ──────────────────────────────────────────
__global__ void kReSTIRGI_Temporal(
    DeviceSceneData scene,
    GIReservoir*       curr,
    const GIReservoir* prev,
    const ReSTIRSurface* surfCurr,
    const ReSTIRSurface* surfPrev,
    uint32_t width, uint32_t height,
    uint32_t prevWidth, uint32_t prevHeight,
    uint32_t sampleIndex,
    uint32_t mCap)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    GIReservoir r = curr[pixelIdx];
    ReSTIRSurface s = surfCurr[pixelIdx];
    if (s.valid < 0.5f) return;

    int px = (int)floorf(s.prevPixel.x);
    int py = (int)floorf(s.prevPixel.y);
    if (px < 0 || py < 0 || px >= (int)prevWidth || py >= (int)prevHeight) return;
    uint32_t prevIdx = (uint32_t)py * prevWidth + (uint32_t)px;

    ReSTIRSurface sp = surfPrev[prevIdx];
    if (sp.valid < 0.5f) return;
    if (dot(s.normal, sp.normal) < 0.9f) return;
    float drift = length(s.position - sp.position);
    if (drift > 0.1f * fmaxf(length(s.position), 1.0f)) return;

    GIReservoir pr = prev[prevIdx];
    if (!pr.valid) return;
    if (pr.M > (float)mCap) {
        // Cap M to bound temporal correlation. wSum stays consistent because
        // W = wSum/(M*pHat) — scaling M without touching W effectively scales
        // wSum by the same factor, exactly the "reduce influence" effect.
        pr.M = (float)mCap;
    }

    uint32_t rng = pcg32_seed(pixelIdx * 0x12345678u + sampleIndex,
                              sampleIndex * 0x9E3779B1u + 0xA5u);

    float wSum = r.pHat * r.M * r.W;
    giReservoirCombine(r, wSum, s, pr, pcg32_float(rng));
    giReservoirFinalize(r, wSum);

    curr[pixelIdx] = r;
    (void)scene;
}

// ── Kernel 3: spatial reuse ───────────────────────────────────────────
__global__ void kReSTIRGI_Spatial(
    DeviceSceneData scene,
    const GIReservoir* inRes,
    GIReservoir*       outRes,
    const ReSTIRSurface* surf,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    uint32_t numNeighbors,
    float    radiusPixels,
    uint32_t mCap)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    GIReservoir r = inRes[pixelIdx];
    ReSTIRSurface s = surf[pixelIdx];
    if (s.valid < 0.5f) { outRes[pixelIdx] = r; return; }

    uint32_t rng = pcg32_seed(pixelIdx * 0xDEADBEEFu + sampleIndex,
                              sampleIndex * 0x85EBCA77u + 0xC1u);

    float wSum = r.pHat * r.M * r.W;

    for (uint32_t i = 0; i < numNeighbors; i++) {
        float u1 = pcg32_float(rng);
        float u2 = pcg32_float(rng);
        float rr = sqrtf(u1) * radiusPixels;
        float th = 2.0f * M_PI_F * u2;
        int nx = (int)x + (int)(rr * cosf(th));
        int ny = (int)y + (int)(rr * sinf(th));
        if (nx < 0 || ny < 0 || nx >= (int)width || ny >= (int)height) continue;
        uint32_t nIdx = (uint32_t)ny * width + (uint32_t)nx;
        if (nIdx == pixelIdx) continue;

        ReSTIRSurface ns = surf[nIdx];
        if (ns.valid < 0.5f) continue;
        if (dot(s.normal, ns.normal) < 0.9f) continue;
        float dz = length(s.position - ns.position);
        if (dz > 0.1f * fmaxf(length(s.position), 1.0f)) continue;

        GIReservoir nr = inRes[nIdx];
        if (!nr.valid) continue;
        if (nr.M > (float)mCap) nr.M = (float)mCap;

        giReservoirCombine(r, wSum, s, nr, pcg32_float(rng));
    }
    giReservoirFinalize(r, wSum);
    outRes[pixelIdx] = r;
    (void)scene;
}

// ── Kernel 4: shade ────────────────────────────────────────────────────
// Resolve each reservoir into the per-pixel indirect-radiance value the
// path tracer adds at primary-hit shading. Estimator at q is:
//   L_indirect = f_r(q, V, wi) * Lo * cos(θ_q) * W
// where pHat is implicit in W (we don't divide by it because the integrand
// IS what the path tracer would have computed).
__global__ void kReSTIRGI_Shade(
    DeviceSceneData scene,
    const GIReservoir*  inRes,
    const ReSTIRSurface* surf,
    float3*              outIndirect,
    uint32_t width, uint32_t height)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    float3 L = make_float3(0, 0, 0);
    ReSTIRSurface s = surf[pixelIdx];
    GIReservoir r   = inRes[pixelIdx];
    if (s.valid >= 0.5f && r.valid && r.W > 0.0f) {
        float3 wi;
        float r2 = 0, cosQ = 0, cosS = 0;
        if (giConnect(s.position, s.normal, r, wi, r2, cosQ, cosS)) {
            float3 brdf = restirEvalBrdf(s, wi);
            // Solid-angle-measure estimator: integrand * W where integrand is
            // f_r * Lo * cos(θ_q). For env hits the integrand has no 1/r^2
            // but giConnect set r2=1 and cosS=1, so the form is the same.
            L = brdf * r.sampleRadiance * (cosQ * r.W);
        }
        // Firefly clamp — long-distance reuse can occasionally produce bright
        // outliers that take many frames to wash out. 50.0 luminance is the
        // standard "be aggressive but don't kill physics" cap.
        float lum = restirLuminance(L);
        const float clampMax = 50.0f;
        if (lum > clampMax) L = L * (clampMax / lum);
        if (isnan(L.x) || isnan(L.y) || isnan(L.z) ||
            isinf(L.x) || isinf(L.y) || isinf(L.z)) L = make_float3(0,0,0);
    }
    outIndirect[pixelIdx] = L;
    (void)scene;
}

// ── Host launchers ──────────────────────────────────────────────────────
static inline dim3 makeGrid(uint32_t w, uint32_t h, dim3 block) {
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

void launchReSTIRGIInitialCandidates(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    GIBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    bool                   enableEnvironment,
    uint32_t               /*temporalMCap*/)
{
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRGI_InitCandidates<<<grid, block>>>(
        scene, camera,
        buffers.d_reservoirsCurr, buffers.d_surfaceCurr,
        width, height, sampleIndex, enableEnvironment ? 1 : 0);
}

void launchReSTIRGITemporalReuse(
    const DeviceSceneData& scene,
    GIBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               temporalMCap)
{
    if (!buffers.historyValid) return;
    if (buffers.prevWidth == 0 || buffers.prevHeight == 0) return;
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRGI_Temporal<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_reservoirsPrev,
        buffers.d_surfaceCurr,    buffers.d_surfacePrev,
        width, height, buffers.prevWidth, buffers.prevHeight,
        sampleIndex, temporalMCap);
}

void launchReSTIRGISpatialReuse(
    const DeviceSceneData& scene,
    GIBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               numNeighbors,
    float                  radiusPixels,
    uint32_t               spatialMCap)
{
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRGI_Spatial<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_reservoirsSpatial,
        buffers.d_surfaceCurr,
        width, height, sampleIndex,
        numNeighbors, radiusPixels, spatialMCap);
}

void launchReSTIRGIShade(
    const DeviceSceneData& scene,
    GIBuffers              buffers,
    uint32_t               width,
    uint32_t               height)
{
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRGI_Shade<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_surfaceCurr,
        buffers.d_indirectOut, width, height);
}

// ── ReSTIRGIContext implementation ────────────────────────────────────
static void allocGIReservoirs(GIReservoir** p, uint32_t count) {
    CUDA_CHECK(cudaMalloc(p, count * sizeof(GIReservoir)));
    CUDA_CHECK(cudaMemset(*p, 0, count * sizeof(GIReservoir)));
}
static void allocGISurfaces(ReSTIRSurface** p, uint32_t count) {
    CUDA_CHECK(cudaMalloc(p, count * sizeof(ReSTIRSurface)));
    CUDA_CHECK(cudaMemset(*p, 0, count * sizeof(ReSTIRSurface)));
}

void ReSTIRGIContext::init(uint32_t width, uint32_t height) {
    free();
    const uint32_t count = width * height;
    allocGIReservoirs(&m_buffers.d_reservoirsCurr,    count);
    allocGIReservoirs(&m_buffers.d_reservoirsPrev,    count);
    allocGIReservoirs(&m_buffers.d_reservoirsSpatial, count);
    allocGISurfaces(&m_buffers.d_surfaceCurr, count);
    allocGISurfaces(&m_buffers.d_surfacePrev, count);
    CUDA_CHECK(cudaMalloc(&m_buffers.d_indirectOut, count * sizeof(float3)));
    CUDA_CHECK(cudaMemset(m_buffers.d_indirectOut, 0, count * sizeof(float3)));
    m_buffers.width  = width;
    m_buffers.height = height;
    m_buffers.prevWidth  = width;
    m_buffers.prevHeight = height;
    m_buffers.historyValid = false;
}

void ReSTIRGIContext::resize(uint32_t width, uint32_t height) {
    if (width == m_buffers.width && height == m_buffers.height) return;
    init(width, height);
}

void ReSTIRGIContext::free() {
    if (m_buffers.d_reservoirsCurr)    cudaFree(m_buffers.d_reservoirsCurr);
    if (m_buffers.d_reservoirsPrev)    cudaFree(m_buffers.d_reservoirsPrev);
    if (m_buffers.d_reservoirsSpatial) cudaFree(m_buffers.d_reservoirsSpatial);
    if (m_buffers.d_surfaceCurr)       cudaFree(m_buffers.d_surfaceCurr);
    if (m_buffers.d_surfacePrev)       cudaFree(m_buffers.d_surfacePrev);
    if (m_buffers.d_indirectOut)       cudaFree(m_buffers.d_indirectOut);
    m_buffers = GIBuffers{};
}

void ReSTIRGIContext::swapHistory() {
    GIReservoir* tr = m_buffers.d_reservoirsCurr;
    m_buffers.d_reservoirsCurr = m_buffers.d_reservoirsPrev;
    m_buffers.d_reservoirsPrev = tr;
    ReSTIRSurface* ts = m_buffers.d_surfaceCurr;
    m_buffers.d_surfaceCurr = m_buffers.d_surfacePrev;
    m_buffers.d_surfacePrev = ts;
    m_buffers.prevWidth  = m_buffers.width;
    m_buffers.prevHeight = m_buffers.height;
    m_buffers.historyValid = true;
}

void ReSTIRGIContext::invalidateHistory() {
    m_buffers.historyValid = false;
}

bool ReSTIRGIContext::runFrame(
    const DeviceSceneData& scene, const CameraParams& camera,
    uint32_t width, uint32_t height, uint32_t sampleIndex,
    bool enableEnvironment)
{
    if (!m_enabled) return false;
    // Initial-candidates pass needs the CUDA BVH.
    if (!scene.d_bvhNodes || scene.totalTriangles == 0) return false;

    launchReSTIRGIInitialCandidates(
        scene, camera, m_buffers,
        width, height, sampleIndex, enableEnvironment, m_temporalMCap);
    launchReSTIRGITemporalReuse(
        scene, m_buffers, width, height, sampleIndex, m_temporalMCap);
    launchReSTIRGISpatialReuse(
        scene, m_buffers, width, height, sampleIndex,
        m_numNeighbors, m_spatialRadius, m_spatialMCap);
    // Spatial output lives in d_reservoirsSpatial — swap with curr so the
    // shade pass and (next frame's) temporal reuse see the post-spatial
    // result.
    GIReservoir* t = m_buffers.d_reservoirsCurr;
    m_buffers.d_reservoirsCurr    = m_buffers.d_reservoirsSpatial;
    m_buffers.d_reservoirsSpatial = t;

    launchReSTIRGIShade(scene, m_buffers, width, height);
    return true;
}
