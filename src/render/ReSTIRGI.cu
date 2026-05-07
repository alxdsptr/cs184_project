#include "render/ReSTIRGI.h"
#include "render/ReSTIRGIDevice.cuh"
#include "render/PathTraceHelpers.cuh"
#include "backend/RayTracingBackend.h"
#include "gpu/Random.h"
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

// Sample direct-lighting contributions at a sample point to seed Lo with
// direct lighting at the indirect bounce: one NEE shadow ray for area lights
// (via the light BVH) plus an unconditional sweep over directional lights.
// Point lights are intentionally omitted — the OptiX kernel disables them
// when area lights exist, and replicating that gate at indirect vertices is
// the same design choice. Directional lights run regardless of area lights.
__device__ inline float3 giDirectLightingAtSample(
    const DeviceSceneData& scene,
    const float3& pos, const float3& normal,
    const float3& albedo, float roughness, float metallic, bool pureDiffuse,
    const float3& viewDir,
    uint32_t& rng)
{
    if (!scene.d_bvhNodes || scene.totalTriangles == 0)
        return make_float3(0.0f, 0.0f, 0.0f);

    // Pack the raw shading attributes into a temporary ReSTIRSurface so the
    // shared BRDF eval (`restirEvalBrdf`) sees the same fields it does at
    // pHat / reservoir-merge time — keeps Lo and pHat using identical BRDF
    // weights.
    ReSTIRSurface tmp = ptMakeSurface(pos, normal, albedo,
                                       fmaxf(roughness, 0.04f), metallic,
                                       pureDiffuse, viewDir, /*specProb=*/0.0f);

    // Opaque-only shadow trace: glass transparency tracking is too expensive
    // for the indirect bounce, and shadow inaccuracy here is invisible once
    // the GI sample is accumulated through temporal + spatial reuse.
    auto traceShadow = [&](float3 origin, float3 dir, float dist) -> float3 {
        float t = (dist >= 1.0e29f) ? 1.0e10f : fmaxf(dist - 0.002f, 0.001f);
        float3 target = origin + dir * t;
        if (bvh_anyHit(origin, target, scene.d_bvhNodes, scene.bvhRootIndex,
                       scene.d_positions, scene.d_indices))
            return make_float3(0.0f, 0.0f, 0.0f);
        return make_float3(1.0f, 1.0f, 1.0f);
    };
    // GI's source-side cap is 50 (looser than PT's 25): GI's sample is just
    // one bounce, so there's less throughput compounding to amplify a grazing
    // firefly. PT's postfix random walk can multiply Li through several
    // throughput stages, which is why PT uses the tighter cap.
    float3 Li = restirAreaLightNEE(scene, tmp, rng, traceShadow, /*fireflyClamp=*/50.0f);
    Li = Li + restirDirectionalLightsNEE(scene, tmp, traceShadow);
    return Li;
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
    // Mix camera.frameIndex into the salt so the canonical sample changes
    // every displayed frame even when sampleIndex is pinned to 0 by camera
    // motion (resetAccumulation zeroes sampleIndex; frameIndex doesn't reset).
    uint32_t seedSalt = sampleIndex + camera.frameIndex * 0x9E3779B9u;
    uint32_t rng = pcg32_seed(pixelIdx * 0x517CC1B7u + seedSalt,
                              seedSalt * 0xCAFEF00Du + 0x67u);

    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;
    Ray ray = generateRay(x, y, width, height, camera, jx, jy);

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
    if (!didHit) {
        outReservoirs[pixelIdx] = r;
        outSurfaces[pixelIdx]   = surf;
        return;
    }
    ReSTIRHitDecode hPrim = restirDecodeHit(
        scene, (uint32_t)hit.primitiveIndex, hit.uv.x, hit.uv.y, ray.direction);
    if (!hPrim.valid) {
        outReservoirs[pixelIdx] = r;
        outSurfaces[pixelIdx]   = surf;
        return;
    }

    surf.position    = hit.position;          // BVH-fed, matches hPrim.pos to fp precision
    surf.normal      = hPrim.normal;
    surf.albedo      = hPrim.albedo;
    surf.roughness   = fmaxf(hPrim.mat.roughness, 0.04f);
    surf.metallic    = hPrim.mat.metallic;
    surf.pureDiffuse = hPrim.pureDiffuse ? 1u : 0u;
    surf.viewDir     = -ray.direction;
    surf.valid       = 1.0f;
    surf.specProb    = computeSpecProb(hPrim.normal, surf.viewDir, hPrim.albedo, hPrim.mat.metallic);

    // Reprojection coordinate for next-frame temporal reuse.
    float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hit.position);
    surf.prevPixel  = make_float2((clipPrev.x + 1.0f) * 0.5f * width,
                                   (1.0f - clipPrev.y) * 0.5f * height);

    // Sample a BSDF direction.
    float3 wi;
    float  pdfBsdf = 0.0f;
    if (!restirSampleBsdfDir(surf, rng, wi, pdfBsdf)) {
        outReservoirs[pixelIdx] = r;
        outSurfaces[pixelIdx]   = surf;
        return;
    }

    // Trace the indirect ray.
    Ray sec;
    sec.origin    = hit.position + hPrim.normal * 0.001f;
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
    float  xrRoughness  = 0.0f;        // §7.5 connectability gate uses this

    if (!didHit2) {
        if (enableEnvironment) {
            float3 envColor = sampleEnvironment(wi, scene.envMapTex);
            float envLum = luminance(envColor);
            const float clampLum = 100.0f;
            if (envLum > clampLum) envColor = envColor * (clampLum / envLum);
            isEnvSample = true;
            samplePos   = wi;          // direction
            sampleNormal = -wi;        // unused but keep something sensible
            Lo = envColor;
            hasSample = (envLum > 0.0f);
            xrRoughness = 0.0f;        // env: gate disabled
        }
    } else {
        ReSTIRHitDecode hSec = restirDecodeHit(
            scene, (uint32_t)hit2.primitiveIndex, hit2.uv.x, hit2.uv.y, wi);
        if (hSec.valid) {
            // Outgoing radiance toward the visible point = emission + 1-bounce NEE.
            float3 viewDir2 = -wi;  // toward the visible point
            float3 direct = giDirectLightingAtSample(
                scene, hit2.position, hSec.normal, hSec.albedo,
                fmaxf(hSec.mat.roughness, 0.04f), hSec.mat.metallic,
                hSec.pureDiffuse, viewDir2, rng);

            Lo = hSec.emission + direct;
            samplePos    = hit2.position;
            sampleNormal = hSec.normal;
            isEnvSample  = false;
            hasSample = (luminance(Lo) > 0.0f);
            xrRoughness = fmaxf(hSec.mat.roughness, 0.04f);
        }
    }

    // Always treat this pixel as having drawn 1 RIS candidate, even if it
    // produced no visible/non-zero sample (BSDF lobe missed everything,
    // shadow ray got fully occluded, etc.). Counting only the successful
    // candidates inflates M's denominator role under temporal+spatial reuse
    // — same root cause as the DI M-counting bug. With this, an invalid
    // sample contributes (M=1, valid=0, W=0) so temporal/spatial merges see
    // a denominator that includes the failed draw.
    {
        float pHat = 0.0f;
        float wCand = 0.0f;
        if (hasSample) {
            GIReservoir cand{};
            cand.visiblePos     = surf.position;
            cand.visibleNormal  = surf.normal;
            cand.samplePos      = samplePos;
            cand.sampleNormal   = sampleNormal;
            cand.sampleRadiance = Lo;
            cand.isEnv          = isEnvSample ? 1u : 0u;
            cand.valid          = 1u;
            cand.xrRoughness    = xrRoughness;     // §7.5 gate
            float3 wiOut;
            pHat  = giEvalTargetPdf(surf, cand, wiOut);
            wCand = (pdfBsdf > 0.0f) ? (pHat / pdfBsdf) : 0.0f;
        }
        float wSum = 0.0f;
        // gris_streamCandidate (via the legacy giReservoirUpdate shim) bumps
        // M unconditionally per paper §5.5/§5.7. Initial-candidate finalize
        // now applies wSum/(M·p̂) to absorb the 1/M Talbot factor (Eq. 5).
        giReservoirUpdate(r, wSum,
                          surf.position, surf.normal,
                          isEnvSample, samplePos, sampleNormal, Lo,
                          pHat, wCand, pcg32_float(rng));
        if (r.valid) {
            // For canonical samples c_i ≡ p̂; cache for downstream MIS.
            gris_cHat(r) = r.pHat;
            // Reconnection-vertex roughness for §7.5 connectability gate.
            r.xrRoughness = xrRoughness;
        }
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
    uint32_t frameIndex,
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
    gris_capM(pr, (float)mCap);          // §6.4

    uint32_t seedSalt = sampleIndex + frameIndex * 0x9E3779B9u;
    uint32_t rng = pcg32_seed(pixelIdx * 0x12345678u + seedSalt,
                              seedSalt * 0x9E3779B1u + 0xA5u);

    GIReservoir peers[1] = { pr };
    ReSTIRSurface peerS[1] = { sp };
    float u01s[1] = { pcg32_float(rng) };
    gris_mergeMultiPair(r, s, peers, peerS, 1, u01s);

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
    uint32_t frameIndex,
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

    uint32_t seedSalt = sampleIndex + frameIndex * 0x9E3779B9u;
    uint32_t rng = pcg32_seed(pixelIdx * 0xDEADBEEFu + seedSalt,
                              seedSalt * 0x85EBCA77u + 0xC1u);

    // Gather peers + surfaces, then perform a single GRIS multi-pair merge
    // (paper §5.6 defensive pairwise MIS). Stack-resident arrays bound by
    // GRIS_GI_MAX_NEIGHBORS — surplus neighbours silently dropped.
    constexpr uint32_t GRIS_GI_MAX_NEIGHBORS = 8;
    GIReservoir   peers[GRIS_GI_MAX_NEIGHBORS];
    ReSTIRSurface peerS[GRIS_GI_MAX_NEIGHBORS];
    float         u01s[GRIS_GI_MAX_NEIGHBORS];
    uint32_t collected = 0;

    uint32_t cap = numNeighbors;
    if (cap > GRIS_GI_MAX_NEIGHBORS) cap = GRIS_GI_MAX_NEIGHBORS;

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
        // Firefly clamp — long-distance reuse can occasionally produce
        // bright outliers that take many frames to wash out. Tightened
        // from 50 → 8 luminance to fix the M7 flash-and-decay artifact
        // (sudden ~10x bright pixel that decays over ~mCap frames). M7
        // has 9759 small emissive triangles, so the GRIS pairwise-MIS
        // denominator can occasionally let a near-grazing NEE-fire
        // sample (Lo·W ≈ tens of luminance) win temporal reuse. The
        // tighter cap bounds the per-frame indirect contribution, so a
        // bad reservoir that survives ~mCap frames in history can't
        // brighten the running accumulator beyond the cap.
        // Per-frame firefly clamp on the final estimator. Set to 50 lum to
        // bound runaway samples (paper §5.4 / Thm A.4) without darkening the
        // image: the upstream RIS-weight + NEE-source clamps already cap the
        // typical contribution, and 50 lum is well above the brightest
        // physical indirect bounce in normal scenes (1.0 ≈ matte white at
        // full sun). Earlier we used 8 here, which was a defensive over-
        // tightening that visibly desaturated/darkened ReSTIR GI output.
        float lum = luminance(L);
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
    uint32_t               frameIndex,
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
        sampleIndex, frameIndex, temporalMCap);
}

void launchReSTIRGISpatialReuse(
    const DeviceSceneData& scene,
    GIBuffers              buffers,
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
    kReSTIRGI_Spatial<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_reservoirsSpatial,
        buffers.d_surfaceCurr,
        width, height, sampleIndex, frameIndex,
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
    bool enableEnvironment,
    RayTracingBackend* backend,
    bool cameraMoved)
{
    if (!m_enabled) return false;
    uint32_t effectiveTemporalMCap = cameraMoved ? m_motionMCap : m_temporalMCap;

    // Prefer backend's native init (OptiX raygen → GAS). Falls back to the
    // CUDA BVH kernel when the backend has no implementation or returns
    // false. The CUDA kernel needs scene.d_bvhNodes patched in by the
    // backend; the OptiX raygen doesn't, so we don't gate on d_bvhNodes
    // until the fall-back path.
    bool initRan = false;
    if (backend) {
        // GI uses a single candidate per pixel (CUDA path matches), but we
        // pass it explicitly so OptiX picks the same M instead of its old
        // hardcoded 1. If we ever expose a numCandidates knob in
        // ReSTIRGIContext, plumb it through here.
        const uint32_t kGINumCandidates = 1;
        initRan = backend->runReSTIRGIInitCandidates(
            scene, camera,
            (void*)m_buffers.d_reservoirsCurr,
            (void*)m_buffers.d_surfaceCurr,
            width, height, sampleIndex, enableEnvironment,
            kGINumCandidates);
    }
    if (!initRan) {
        if (!scene.d_bvhNodes || scene.totalTriangles == 0) return false;
        launchReSTIRGIInitialCandidates(
            scene, camera, m_buffers,
            width, height, sampleIndex, enableEnvironment, m_temporalMCap);
    }
    launchReSTIRGITemporalReuse(
        scene, m_buffers, width, height,
        sampleIndex, camera.frameIndex, effectiveTemporalMCap);
    launchReSTIRGISpatialReuse(
        scene, m_buffers, width, height,
        sampleIndex, camera.frameIndex,
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
