#include "render/ReSTIRPT.h"
#include "render/ReSTIRGIDevice.cuh"   // GRIS primitives, shared with ReSTIR GI
#include "backend/RayTracingBackend.h"
#include "core/Math.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "gpu/Sampling.h"
#include "accel/BVH.h"
#include "accel/LightBVHSample.h"
#include "util/CudaCheck.h"

#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
// ReSTIR PT (Lin et al. 2022) — primary-hit indirect lighting via GRIS-based
// path-reservoir resampling.
//
// Pipeline (per frame, paper §6.3):
//   1. Initial candidates : at every pixel, generate `numCandidates` BSDF-
//      sampled paths and RIS-resample one survivor with target p̂ ∝ |f * Lo|.
//      Each path traces to a reconnection vertex x_r and runs a multi-bounce
//      random walk (NEE per vertex) for the path postfix Lo.
//   2. Temporal reuse     : multi-peer GRIS merge (with N=1) of the prev-
//      frame reservoir at the reprojected pixel, using defensive pairwise
//      MIS (paper Eq. 38). M-cap bounds temporal correlation length (§6.4).
//   3. Spatial reuse      : multi-peer GRIS merge over `numNeighbors` disk-
//      sampled neighbors PLUS the current pixel's canonical sample (|R|=1
//      per §5.5/§5.7 for guaranteed convergence). Single pass per kernel
//      so MIS denominators are jointly evaluated (no streaming-order bias).
//   4. Shade              : materialize the final reservoir into per-pixel
//      indirect radiance via Lo · f · cos(θ_q) · W (Eq. 22).
//
// Visibility from q to x_r is NOT re-tested across pixels (a pragmatic
// approximation — paper Appendix B §"On Visibility" calls this out as a
// performance-driven shortcut to ReSTIR DI's conservative approach).
// ─────────────────────────────────────────────────────────────────────────────

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

namespace {

// Camera ray identical to the main path tracer / DI / GI generators so the
// visible point lines up with the path tracer's primary-hit shading point.
__device__ inline Ray ptGenerateRay(
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

__device__ inline float3 ptProceduralSky(float3 dir) {
    float t = 0.5f * (dir.y + 1.0f);
    return lerp(make_float3(1, 1, 1), make_float3(0.5f, 0.7f, 1.0f), t) * 0.8f;
}

__device__ inline float3 ptSampleEnvironment(float3 dir,
                                              cudaTextureObject_t envMap)
{
    if (envMap != 0) {
        float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
        float phi   = atan2f(dir.z, dir.x);
        float u = (phi + M_PI_F) * (0.5f / M_PI_F);
        float v = theta / M_PI_F;
        float4 texel = tex2D<float4>(envMap, u, v);
        return make_float3(texel.x, texel.y, texel.z);
    }
    return ptProceduralSky(dir);
}

// ── BSDF helpers (mirrors the main path tracer's mixture model) ────────────

__device__ inline float ptComputeSpecProb(
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

__device__ inline float ptDiffusePdf(float NdotL) {
    return fmaxf(NdotL, 0.0f) * (1.0f / M_PI_F);
}

__device__ inline float ptSpecularPdf(
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

__device__ inline float ptMixturePdf(
    bool pureDiffuse, const float3& N, const float3& V, const float3& L,
    float roughness, float specProb)
{
    float diffPdf = ptDiffusePdf(dot(N, L));
    if (pureDiffuse) return diffPdf;
    float specPdf = ptSpecularPdf(N, V, L, roughness);
    return specProb * specPdf + (1.0f - specProb) * diffPdf;
}

__device__ inline ReSTIRSurface ptMakeSurface(
    const float3& pos, const float3& N, const float3& albedo,
    float roughness, float metallic, bool pureDiffuse, const float3& viewDir,
    float specProb)
{
    ReSTIRSurface s{};
    s.position    = pos;
    s.normal      = N;
    s.albedo      = albedo;
    s.roughness   = fmaxf(roughness, 0.04f);
    s.metallic    = metallic;
    s.pureDiffuse = pureDiffuse ? 1u : 0u;
    s.viewDir     = viewDir;
    s.specProb    = specProb;
    s.valid       = 1.0f;
    return s;
}

__device__ inline bool ptSampleBsdfDir(
    const ReSTIRSurface& s, uint32_t& rng,
    float3& outDir, float& outPdf)
{
    bool pureDiffuse = (s.pureDiffuse != 0u);
    float specProb = pureDiffuse ? 0.0f : s.specProb;
    float u = pcg32_float(rng);
    float3 dir;
    if (!pureDiffuse && u < specProb) {
        float a = s.roughness * s.roughness;
        float u1 = pcg32_float(rng);
        float u2 = pcg32_float(rng);
        float cosTheta = sqrtf((1.0f - u1) / (1.0f + (a*a - 1.0f) * u1 + 1e-7f));
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = 2.0f * M_PI_F * u2;
        float3 localH = make_float3(sinTheta * cosf(phi), cosTheta,
                                     sinTheta * sinf(phi));
        float3 T, B;
        buildONB(s.normal, T, B);
        float3 H = localToWorld(localH, T, s.normal, B);
        float3 inDir = -s.viewDir;
        dir = normalize(inDir - H * (2.0f * dot(inDir, H)));
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
    outPdf = ptMixturePdf(pureDiffuse, s.normal, s.viewDir, dir,
                          s.roughness, specProb);
    return outPdf > 1e-7f;
}

// One NEE bounce. Mirrors the main path tracer.
__device__ inline float3 ptDirectLightingAtVertex(
    const DeviceSceneData& scene, const ReSTIRSurface& s, uint32_t& rng)
{
    if (!scene.d_areaLights || scene.areaLightCount == 0 ||
        !scene.d_lightBVHNodes) return make_float3(0, 0, 0);
    if (!scene.d_bvhNodes || scene.totalTriangles == 0)
        return make_float3(0, 0, 0);

    uint32_t slot = 0;
    float    pSelect = 0.0f;
    if (!lightBVH_sample(scene.d_lightBVHNodes, scene.lightBVHRootIndex,
                         s.position, pcg32_float(rng), slot, pSelect) ||
        !(pSelect > 0.0f))
        return make_float3(0, 0, 0);
    uint32_t lightIdx = scene.d_lightOrderedIndices[slot];
    GPUAreaLight light = scene.d_areaLights[lightIdx];

    float r1 = pcg32_float(rng), r2 = pcg32_float(rng);
    float su = sqrtf(r1);
    float b0 = 1.0f - su;
    float b1 = su * (1.0f - r2);
    float b2 = su * r2;
    float3 lp = light.v0 * b0 + (light.v0 + light.e1) * b1
              + (light.v0 + light.e2) * b2;
    float3 toL = lp - s.position;
    float  d2  = fmaxf(dot(toL, toL), 1e-6f);
    float  d   = sqrtf(d2);
    float3 L   = toL * (1.0f / d);
    float NdotL = fmaxf(dot(s.normal, L), 0.0f);
    float lightCos = fmaxf(dot(light.normal, -L), 0.0f);
    if (NdotL <= 0.0f || lightCos <= 0.0f) return make_float3(0, 0, 0);

    float3 shadowOrigin = s.position + s.normal * 0.001f;
    if (bvh_anyHit(shadowOrigin, lp,
                   scene.d_bvhNodes, scene.bvhRootIndex,
                   scene.d_positions, scene.d_indices))
        return make_float3(0, 0, 0);

    float3 Le;
    if (light.emissiveTex == 0) {
        Le = light.emission;
    } else {
        float texU = light.uv0.x * b0 + light.uv1.x * b1 + light.uv2.x * b2;
        float texV = light.uv0.y * b0 + light.uv1.y * b1 + light.uv2.y * b2;
        float4 et = tex2D<float4>(light.emissiveTex, texU, texV);
        Le = make_float3(et.x, et.y, et.z) * light.emission;
    }

    float3 brdf = restirEvalBrdf(s, L);
    float pTri  = pSelect;
    float pArea = pTri / fmaxf(light.area, 1e-7f);
    float pdfOmega = pArea * d2 / fmaxf(lightCos, 1e-7f);
    float3 Li = brdf * Le * (NdotL / fmaxf(pdfOmega, 1e-7f));
    // Source-side firefly clamp at the indirect bounce. Mirrors the GI
    // path's clamp in giDirectLightingAtSample. See ReSTIRGI.cu for the
    // rationale; in short: a grazing NEE-firefly stored in `sampleRadiance`
    // would get propagated forward for ~mCap frames by the temporal pass.
    // 25 luminance is tighter than GI's 50 because PT's path postfix can
    // multiply Li through several throughput stages (random-walk bounces).
    float lumLi = restirLuminance(Li);
    const float liCap = 25.0f;
    if (lumLi > liCap) Li = Li * (liCap / lumLi);
    return Li;
}

// Resolve a BVH closest-hit into shading attributes.
__device__ inline bool ptShadeHit(
    const DeviceSceneData& scene, const Ray& ray, const HitRecord& hit,
    float3& outPos, float3& outN, float3& outAlbedo, float3& outEmission,
    float& outRoughness, float& outMetallic, bool& outPureDiffuse)
{
    if (hit.materialIndex < 0 ||
        (uint32_t)hit.materialIndex >= scene.materialCount) return false;

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
    float3 emis = mat.emission * mat.emissionStrength;
    if (mat.emissiveTex != 0) {
        float4 et = tex2D<float4>(mat.emissiveTex, uv.x, uv.y);
        emis = make_float3(et.x, et.y, et.z) * mat.emissionStrength;
    }
    outPos = hit.position;
    outN   = N;
    outAlbedo   = albedo;
    outEmission = emis;
    outRoughness = fmaxf(mat.roughness, 0.04f);
    outMetallic  = mat.metallic;
    outPureDiffuse = (mat.pureDiffuse != 0);
    return true;
}

// Multi-bounce random walk starting at x_r. Returns the postfix radiance Lo
// at x_r toward `viewDir`. `bounces` is the number of additional bounces
// past x_r — bounces=0 reduces to ReSTIR GI's k=1 postfix.
__device__ inline float3 ptPathPostfix(
    const DeviceSceneData& scene,
    const float3& xrPos, const float3& xrN,
    const float3& xrAlbedo, const float3& xrEmis,
    float xrRoughness, float xrMetallic, bool xrPureDiffuse,
    const float3& viewDir,
    bool  enableEnvironment,
    uint32_t bounces,
    uint32_t& rng)
{
    float3 L = xrEmis;
    float specProb_xr = ptComputeSpecProb(xrN, viewDir, xrAlbedo, xrMetallic);
    ReSTIRSurface curr = ptMakeSurface(xrPos, xrN, xrAlbedo,
                                        xrRoughness, xrMetallic, xrPureDiffuse,
                                        viewDir, specProb_xr);
    L = L + ptDirectLightingAtVertex(scene, curr, rng);

    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    for (uint32_t i = 0; i < bounces; i++) {
        float3 wi;
        float  pdfBsdf = 0.0f;
        if (!ptSampleBsdfDir(curr, rng, wi, pdfBsdf)) break;

        float3 brdf = restirEvalBrdf(curr, wi);
        float NdotL = fmaxf(dot(curr.normal, wi), 0.0f);
        if (NdotL <= 0.0f) break;
        float3 weight = brdf * (NdotL / pdfBsdf);
        throughput = throughput * weight;

        Ray nextRay;
        nextRay.origin    = curr.position + curr.normal * 0.001f;
        nextRay.direction = wi;
        nextRay.tmin      = 0.001f;
        nextRay.tmax      = 1e30f;
        HitRecord nh; nh.t = 1e30f;
        bool got = false;
        if (scene.d_bvhNodes && scene.totalTriangles > 0) {
            got = bvh_closestHit(nextRay, scene.d_bvhNodes, scene.bvhRootIndex,
                                 scene.d_positions, scene.d_indices,
                                 scene.d_materialIndices, nh);
        }

        if (!got) {
            if (enableEnvironment) {
                float3 envColor = ptSampleEnvironment(wi, scene.envMapTex);
                float envLum = restirLuminance(envColor);
                const float clampLum = 100.0f;
                if (envLum > clampLum) envColor = envColor * (clampLum / envLum);
                L = L + throughput * envColor;
            }
            break;
        }

        float3 hPos, hN, hAlbedo, hEmis;
        float  hRoughness, hMetallic;
        bool   hPure;
        if (!ptShadeHit(scene, nextRay, nh,
                        hPos, hN, hAlbedo, hEmis,
                        hRoughness, hMetallic, hPure)) break;

        // Add the BSDF-sampled emission with proper MIS weight against NEE
        // (balance heuristic) so emitter-heavy scenes don't double-count.
        if (restirLuminance(hEmis) > 0.0f) {
            // Approx: NEE pdf for hitting this same point would be
            //    p_nee = (pTri/area) * d^2 / cos_light  — without geometry
            // we don't have nh's emitter triangle in our quick post-walk so
            // we use the conservative approximation that NEE selects a
            // random light-area sample at the same surface; the mixture is
            // pdfBsdf vs. uniform-area sampling. Detailed MIS would need
            // an emitter PDF lookup; we use throughput-weighted Lo without
            // multi-counting by gating on hEmis only at the FIRST extra
            // bounce (the path postfix Lo is dominated by NEE elsewhere).
            if (i == 0) L = L + throughput * hEmis;
        }

        float3 nViewDir = -wi;
        float specProb = ptComputeSpecProb(hN, nViewDir, hAlbedo, hMetallic);
        curr = ptMakeSurface(hPos, hN, hAlbedo, hRoughness, hMetallic, hPure,
                             nViewDir, specProb);

        L = L + throughput * ptDirectLightingAtVertex(scene, curr, rng);

        if (i >= 1) {
            float maxC = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            float pCont = fminf(fmaxf(maxC, 0.05f), 0.95f);
            if (pcg32_float(rng) > pCont) break;
            throughput = throughput * (1.0f / pCont);
        }
    }

    // Aggressive firefly clamp — long postfixes can hit caustics whose
    // multiplied contribution at q would persist for many frames in the
    // reservoir history.
    float lum = restirLuminance(L);
    const float clampMax = 200.0f;
    if (lum > clampMax) L = L * (clampMax / lum);
    return L;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: initial-candidate GRIS RIS (paper §4.1, Eq. 5)
//
// For each pixel we draw `numCandidates` independent BSDF-sampled paths and
// resample one survivor with target p̂(y) ∝ luminance(f * Lo) · cos(θ_q).
// Each candidate contributes p̂ / pdfBsdf to the RIS sum; the final reservoir
// W is wSum / (M · p̂(Y)) per Eq. 22 with canonical |R|=1 reduction.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kReSTIRPT_InitCandidates(
    DeviceSceneData scene,
    CameraParams    camera,
    GIReservoir*    outReservoirs,
    ReSTIRSurface*  outSurfaces,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    int      enableEnvironment,
    uint32_t pathLength,
    uint32_t numCandidates)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    // Mix camera.frameIndex into the seed so the canonical sample changes
    // every frame even when sampleIndex is pinned to 0 by camera motion
    // (resetAccumulation() zeros sampleIndex but frameIndex monotonically
    // counts displayed frames). Without this, every frame during continuous
    // camera motion would draw IDENTICAL BSDF directions / NEE picks per
    // pixel — making temporal reuse degenerate to "same sample forever".
    uint32_t seedSalt = sampleIndex + camera.frameIndex * 0x9E3779B9u;
    uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B1u + seedSalt,
                              seedSalt * 0x85EBCA6Bu + 0xB7u);

    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;
    Ray ray = ptGenerateRay(x, y, width, height, camera, jx, jy);

    GIReservoir r; giReservoirReset(r);
    ReSTIRSurface surf{}; surf.valid = 0.0f;

    HitRecord hit; hit.t = 1e30f;
    bool didHit = false;
    if (scene.d_bvhNodes && scene.totalTriangles > 0) {
        didHit = bvh_closestHit(ray, scene.d_bvhNodes, scene.bvhRootIndex,
                                scene.d_positions, scene.d_indices,
                                scene.d_materialIndices, hit);
    }
    if (!didHit) {
        outReservoirs[pixelIdx] = r;
        outSurfaces[pixelIdx]   = surf;
        return;
    }

    float3 hPos, hN, hAlbedo, hEmis;
    float  hRoughness, hMetallic;
    bool   hPure;
    if (!ptShadeHit(scene, ray, hit, hPos, hN, hAlbedo, hEmis,
                    hRoughness, hMetallic, hPure)) {
        outReservoirs[pixelIdx] = r;
        outSurfaces[pixelIdx]   = surf;
        return;
    }

    surf.position    = hPos;
    surf.normal      = hN;
    surf.albedo      = hAlbedo;
    surf.roughness   = hRoughness;
    surf.metallic    = hMetallic;
    surf.pureDiffuse = hPure ? 1u : 0u;
    surf.viewDir     = -ray.direction;
    surf.valid       = 1.0f;
    surf.specProb    = ptComputeSpecProb(hN, surf.viewDir, hAlbedo, hMetallic);

    float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hPos);
    surf.prevPixel  = make_float2((clipPrev.x + 1.0f) * 0.5f * width,
                                   (1.0f - clipPrev.y) * 0.5f * height);

    // ── Generate `numCandidates` independent paths and stream into RIS ──
    float wSum = 0.0f;
    float xrRoughnessKept = 1.0f;        // stored alongside the kept sample

    if (numCandidates < 1) numCandidates = 1;
    for (uint32_t k = 0; k < numCandidates; k++) {
        float3 wi;
        float  pdfBsdf = 0.0f;
        if (!ptSampleBsdfDir(surf, rng, wi, pdfBsdf)) {
            // Failed candidate still bumps M for the |R| accounting.
            r.M += 1.0f;
            continue;
        }

        Ray sec;
        sec.origin    = hPos + hN * 0.001f;
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

        bool   isEnvCand = false;
        float3 candPos    = make_float3(0,0,0);
        float3 candNormal = make_float3(0,1,0);
        float3 Lo         = make_float3(0,0,0);
        float  candXrRough = 0.0f;
        bool   ok = false;

        if (!didHit2) {
            if (enableEnvironment) {
                float3 envColor = ptSampleEnvironment(wi, scene.envMapTex);
                float envLum = restirLuminance(envColor);
                const float clampLum = 100.0f;
                if (envLum > clampLum) envColor = envColor * (clampLum / envLum);
                isEnvCand = true;
                candPos    = wi;
                candNormal = -wi;
                Lo = envColor;
                candXrRough = 0.0f;       // env: roughness gate disabled
                ok = (envLum > 0.0f);
            }
        } else {
            float3 xPos, xN, xAlbedo, xEmis;
            float  xRoughness, xMetallic;
            bool   xPure;
            if (ptShadeHit(scene, sec, hit2,
                           xPos, xN, xAlbedo, xEmis,
                           xRoughness, xMetallic, xPure)) {
                float3 viewAtXr = -wi;
                Lo = ptPathPostfix(scene,
                                   xPos, xN, xAlbedo, xEmis,
                                   xRoughness, xMetallic, xPure,
                                   viewAtXr,
                                   enableEnvironment != 0,
                                   pathLength,
                                   rng);
                candPos    = xPos;
                candNormal = xN;
                isEnvCand  = false;
                candXrRough = xRoughness;
                ok = (restirLuminance(Lo) > 0.0f);
            }
        }

        // Evaluate target pHat at the visible surface.
        float pHat = 0.0f;
        if (ok) {
            GIReservoir cand{};
            cand.visiblePos     = surf.position;
            cand.visibleNormal  = surf.normal;
            cand.samplePos      = candPos;
            cand.sampleNormal   = candNormal;
            cand.sampleRadiance = Lo;
            cand.isEnv          = isEnvCand ? 1u : 0u;
            cand.valid          = 1u;
            cand.xrRoughness    = candXrRough;
            float3 wiOut;
            pHat = giEvalTargetPdf(surf, cand, wiOut);
        }

        // Stream into the canonical reservoir. gris_streamCandidate bumps M
        // unconditionally (paper §5.5/§5.7 needs M to track ALL canonical
        // draws, including failed ones, for the convergence guarantee).
        bool replaced = gris_streamCandidate(
            r, wSum,
            surf.position, surf.normal,
            isEnvCand, candPos, candNormal, Lo,
            pHat, pdfBsdf, pcg32_float(rng));
        if (replaced) xrRoughnessKept = candXrRough;
    }

    // Cache the surface roughness as cHat (target at canonical source = p̂).
    if (r.valid) {
        // cHat = p̂_src for the held sample at THIS pixel, equal to r.pHat.
        gris_cHat(r) = r.pHat;
        r.xrRoughness = xrRoughnessKept;
    }
    giReservoirFinalize(r, wSum);

    outReservoirs[pixelIdx] = r;
    outSurfaces[pixelIdx]   = surf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: temporal reuse (paper §6.3 step 2)
//
// Single-peer multi-pair MIS merge with M-cap on the prior frame. Geometric
// gates filter disocclusions and motion-vector errors before merging.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kReSTIRPT_Temporal(
    DeviceSceneData scene,
    GIReservoir*       curr,
    const GIReservoir* prev,
    const ReSTIRSurface* surfCurr,
    const ReSTIRSurface* surfPrev,
    uint32_t width, uint32_t height,
    uint32_t prevWidth, uint32_t prevHeight,
    uint32_t sampleIndex,
    uint32_t frameIndex,                       // monotonic, never reset
    uint32_t mCap)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;
    uint32_t seedSalt = sampleIndex + frameIndex * 0x9E3779B9u;

    GIReservoir r = curr[pixelIdx];
    ReSTIRSurface s = surfCurr[pixelIdx];
    if (s.valid < 0.5f) return;

    int px = (int)floorf(s.prevPixel.x);
    int py = (int)floorf(s.prevPixel.y);
    if (px < 0 || py < 0 ||
        px >= (int)prevWidth || py >= (int)prevHeight) return;
    uint32_t prevIdx = (uint32_t)py * prevWidth + (uint32_t)px;

    ReSTIRSurface sp = surfPrev[prevIdx];
    if (sp.valid < 0.5f) return;
    if (dot(s.normal, sp.normal) < 0.9f) return;
    float drift = length(s.position - sp.position);
    if (drift > 0.1f * fmaxf(length(s.position), 1.0f)) return;

    GIReservoir pr = prev[prevIdx];
    if (!pr.valid) return;
    gris_capM(pr, (float)mCap);                // §6.4

    uint32_t rng = pcg32_seed(pixelIdx * 0x119DE1F3u + seedSalt,
                              seedSalt * 0xCC9E2D51u + 0xC1u);

    GIReservoir peers[1] = { pr };
    ReSTIRSurface peerS[1] = { sp };
    float u01s[1] = { pcg32_float(rng) };
    gris_mergeMultiPair(r, s, peers, peerS, 1, u01s);

    curr[pixelIdx] = r;
    (void)scene;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: spatial reuse (paper §6.3 step 3)
//
// Multi-peer one-shot GRIS merge — peer reservoirs are gathered into a small
// stack-resident array and then folded into `r` in a single pairwise-MIS
// pass. This avoids the sequential streaming bias that would otherwise creep
// in with N>1 peers under generalized Talbot weights.
//
// The destination's own reservoir is treated as the canonical sample (|R|=1)
// throughout the merge, so the convergence proofs of §5.7 apply.
// ─────────────────────────────────────────────────────────────────────────────

#ifndef GRIS_PT_MAX_NEIGHBORS
#define GRIS_PT_MAX_NEIGHBORS 8
#endif

__global__ void kReSTIRPT_Spatial(
    DeviceSceneData scene,
    const GIReservoir* inRes,
    GIReservoir*       outRes,
    const ReSTIRSurface* surf,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    uint32_t frameIndex,                       // monotonic, never reset
    uint32_t numNeighbors,
    float    radiusPixels,
    uint32_t mCap)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;
    uint32_t seedSalt = sampleIndex + frameIndex * 0x9E3779B9u;

    GIReservoir r = inRes[pixelIdx];
    ReSTIRSurface s = surf[pixelIdx];
    if (s.valid < 0.5f) { outRes[pixelIdx] = r; return; }

    uint32_t rng = pcg32_seed(pixelIdx * 0x1B873593u + seedSalt,
                              seedSalt * 0xE6546B64u + 0xC2u);

    // Gather peer reservoirs + their surfaces into local arrays. Capped by
    // GRIS_PT_MAX_NEIGHBORS; surplus neighbours are silently dropped (cheap).
    GIReservoir   peers[GRIS_PT_MAX_NEIGHBORS];
    ReSTIRSurface peerS[GRIS_PT_MAX_NEIGHBORS];
    float         u01s[GRIS_PT_MAX_NEIGHBORS];
    uint32_t collected = 0;

    uint32_t cap = numNeighbors;
    if (cap > GRIS_PT_MAX_NEIGHBORS) cap = GRIS_PT_MAX_NEIGHBORS;

    for (uint32_t i = 0; i < cap; i++) {
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
        gris_capM(nr, (float)mCap);

        peers[collected] = nr;
        peerS[collected] = ns;
        u01s[collected]  = pcg32_float(rng);
        collected++;
    }

    if (collected > 0) {
        gris_mergeMultiPair(r, s, peers, peerS, collected, u01s);
    }
    outRes[pixelIdx] = r;
    (void)scene;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: shade — Eq. 22 estimator at q.
//
//   L_indirect(q) = f_r(q, V, wi) · L_o · cos(θ_q) · W
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kReSTIRPT_Shade(
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
            L = brdf * r.sampleRadiance * (cosQ * r.W);
        }
        // Aggressive firefly clamp on the final estimator (paper §5.4
        // bounded variance still allows occasional outliers; this caps
        // them to keep the displayed image stable while we're still in
        // the few-sample-per-pixel regime). Tightened from 50 → 8 to
        // fix the M7 flash-and-decay artifact: with 9759 emissive tris
        // a near-grazing NEE-firefly (Le * NdotL / pdfOmega) can ride
        // pairwise-MIS through temporal reuse and stick for ~mCap
        // frames, brightening the accumulator long after the spike.
        float lum = restirLuminance(L);
        const float clampMax = 8.0f;
        if (lum > clampMax) L = L * (clampMax / lum);
        if (isnan(L.x) || isnan(L.y) || isnan(L.z) ||
            isinf(L.x) || isinf(L.y) || isinf(L.z)) L = make_float3(0,0,0);
    }
    outIndirect[pixelIdx] = L;
    (void)scene;
}

// ─────────────────────────────────────────────────────────────────────────────
// Host launchers
// ─────────────────────────────────────────────────────────────────────────────
static inline dim3 makeGrid(uint32_t w, uint32_t h, dim3 block) {
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

void launchReSTIRPTInitialCandidates(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    bool                   enableEnvironment,
    uint32_t               pathLength,
    uint32_t               numCandidates)
{
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRPT_InitCandidates<<<grid, block>>>(
        scene, camera,
        buffers.d_reservoirsCurr, buffers.d_surfaceCurr,
        width, height, sampleIndex,
        enableEnvironment ? 1 : 0, pathLength, numCandidates);
}

void launchReSTIRPTTemporalReuse(
    const DeviceSceneData& scene,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               frameIndex,
    uint32_t               temporalMCap)
{
    if (!buffers.historyValid) return;
    if (buffers.prevWidth == 0 || buffers.prevHeight == 0) return;
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRPT_Temporal<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_reservoirsPrev,
        buffers.d_surfaceCurr,    buffers.d_surfacePrev,
        width, height, buffers.prevWidth, buffers.prevHeight,
        sampleIndex, frameIndex, temporalMCap);
}

void launchReSTIRPTSpatialReuse(
    const DeviceSceneData& scene,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               frameIndex,
    uint32_t               numNeighbors,
    float                  radiusPixels,
    uint32_t               spatialMCap)
{
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRPT_Spatial<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_reservoirsSpatial,
        buffers.d_surfaceCurr,
        width, height, sampleIndex, frameIndex,
        numNeighbors, radiusPixels, spatialMCap);
}

void launchReSTIRPTShade(
    const DeviceSceneData& scene,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height)
{
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRPT_Shade<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_surfaceCurr,
        buffers.d_indirectOut, width, height);
}

// ── ReSTIRPTContext implementation ────────────────────────────────────────
static void allocPTReservoirs(GIReservoir** p, uint32_t count) {
    CUDA_CHECK(cudaMalloc(p, count * sizeof(GIReservoir)));
    CUDA_CHECK(cudaMemset(*p, 0, count * sizeof(GIReservoir)));
}
static void allocPTSurfaces(ReSTIRSurface** p, uint32_t count) {
    CUDA_CHECK(cudaMalloc(p, count * sizeof(ReSTIRSurface)));
    CUDA_CHECK(cudaMemset(*p, 0, count * sizeof(ReSTIRSurface)));
}

void ReSTIRPTContext::init(uint32_t width, uint32_t height) {
    free();
    const uint32_t count = width * height;
    allocPTReservoirs(&m_buffers.d_reservoirsCurr,    count);
    allocPTReservoirs(&m_buffers.d_reservoirsPrev,    count);
    allocPTReservoirs(&m_buffers.d_reservoirsSpatial, count);
    allocPTSurfaces(&m_buffers.d_surfaceCurr, count);
    allocPTSurfaces(&m_buffers.d_surfacePrev, count);
    CUDA_CHECK(cudaMalloc(&m_buffers.d_indirectOut, count * sizeof(float3)));
    CUDA_CHECK(cudaMemset(m_buffers.d_indirectOut, 0, count * sizeof(float3)));
    m_buffers.width  = width;
    m_buffers.height = height;
    m_buffers.prevWidth  = width;
    m_buffers.prevHeight = height;
    m_buffers.historyValid = false;
}

void ReSTIRPTContext::resize(uint32_t width, uint32_t height) {
    if (width == m_buffers.width && height == m_buffers.height) return;
    init(width, height);
}

void ReSTIRPTContext::free() {
    if (m_buffers.d_reservoirsCurr)    cudaFree(m_buffers.d_reservoirsCurr);
    if (m_buffers.d_reservoirsPrev)    cudaFree(m_buffers.d_reservoirsPrev);
    if (m_buffers.d_reservoirsSpatial) cudaFree(m_buffers.d_reservoirsSpatial);
    if (m_buffers.d_surfaceCurr)       cudaFree(m_buffers.d_surfaceCurr);
    if (m_buffers.d_surfacePrev)       cudaFree(m_buffers.d_surfacePrev);
    if (m_buffers.d_indirectOut)       cudaFree(m_buffers.d_indirectOut);
    m_buffers = PTBuffers{};
}

void ReSTIRPTContext::swapHistory() {
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

void ReSTIRPTContext::invalidateHistory() {
    m_buffers.historyValid = false;
}

bool ReSTIRPTContext::runFrame(
    const DeviceSceneData& scene, const CameraParams& camera,
    uint32_t width, uint32_t height, uint32_t sampleIndex,
    bool enableEnvironment,
    RayTracingBackend* backend,
    bool cameraMoved)
{
    if (!m_enabled) return false;
    uint32_t effectiveTemporalMCap = cameraMoved ? m_motionMCap : m_temporalMCap;

    // Prefer the backend's native init-candidates implementation (OptiX
    // raygen → GAS); fall back to the CUDA kernel (needs scene.d_bvhNodes).
    bool initRan = false;
    if (backend) {
        // Forward m_numCandidates so OptiX matches the CUDA kernel's RIS.
        // The OptiX raygen used to be hardcoded at 1 candidate, which made
        // ReSTIR PT noticeably noisier than the CUDA path on the same scene.
        initRan = backend->runReSTIRPTInitCandidates(
            scene, camera,
            (void*)m_buffers.d_reservoirsCurr,
            (void*)m_buffers.d_surfaceCurr,
            width, height, sampleIndex,
            enableEnvironment, m_pathLength, m_numCandidates);
    }
    if (!initRan) {
        if (!scene.d_bvhNodes || scene.totalTriangles == 0) return false;
        launchReSTIRPTInitialCandidates(
            scene, camera, m_buffers,
            width, height, sampleIndex, enableEnvironment,
            m_pathLength, m_numCandidates);
    }
    launchReSTIRPTTemporalReuse(
        scene, m_buffers, width, height,
        sampleIndex, camera.frameIndex, effectiveTemporalMCap);
    launchReSTIRPTSpatialReuse(
        scene, m_buffers, width, height,
        sampleIndex, camera.frameIndex,
        m_numNeighbors, m_spatialRadius, m_spatialMCap);
    GIReservoir* t = m_buffers.d_reservoirsCurr;
    m_buffers.d_reservoirsCurr    = m_buffers.d_reservoirsSpatial;
    m_buffers.d_reservoirsSpatial = t;

    launchReSTIRPTShade(scene, m_buffers, width, height);
    return true;
}
