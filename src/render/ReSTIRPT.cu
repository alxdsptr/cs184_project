#include "render/ReSTIRPT.h"
#include "render/ReSTIRGIDevice.cuh"   // GRIS primitives, shared with ReSTIR GI
#include "render/PathTraceHelpers.cuh"
#include "backend/RayTracingBackend.h"
#include "gpu/Random.h"
#include "accel/BVH.h"
#include "accel/LightBVHSample.h"
#include "util/CudaCheck.h"

#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>

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

// ─────────────────────────────────────────────────────────────────────────────
// §6.1 Unify DI + GI in a single ReSTIR PT reservoir.
//
// Adds NEE-sampled (d=2) direct-light candidates alongside the BSDF-sampled
// candidates inside the initial RIS. Conceptually attractive (one less pass,
// glossy primary highlights benefit from PT's shift mappings), but requires
// the host-side main path-trace kernel to suppress its own primary-hit direct
// lighting + emission so the d=2 light path isn't double-counted.
//
// Set to 1 ONLY together with the matching kernel-side patch. Default 0 keeps
// PT in indirect-only mode (matches Lin et al. 2022's published split): PT
// reservoir holds d≥3 paths, kernel handles d=2 directly.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PT_UNIFY_DIRECT_LIGHTING
#define PT_UNIFY_DIRECT_LIGHTING 0
#endif

// §5 duplication-map cap reduction. Now ON with a generous dead-zone
// (D < 0.25 → no scaling) and gentle α=2 inside the temporal kernel so
// legitimately-converged regions don't get darkened.
#ifndef PT_ENABLE_DUPMAP_CAP
#define PT_ENABLE_DUPMAP_CAP 1
#endif

// §6.3 vector-valued shading. Now safe to enable since the merge MIS is
// fixed (Talbot uniform partition of unity); previously the scalar p̂·W·|J|
// did not sum to 1 over the technique support and the vector substitution
// inherited that bias.
#ifndef PT_ENABLE_VECTOR_SHADING
#define PT_ENABLE_VECTOR_SHADING 1
#endif

namespace {

// One NEE bounce at vertex `s`. CUDA backend uses `bvh_anyHit` (opaque-only
// shadow trace; transparent geometry is not seen-through here, matching the
// main path tracer's CUDA NEE). 25-luminance source-side clamp on area-light
// contributions: tighter than GI's 50 because PT's path postfix can multiply
// Li through several throughput stages (random-walk bounces).
__device__ inline float3 ptDirectLightingAtVertex(
    const DeviceSceneData& scene, const ReSTIRSurface& s, uint32_t& rng)
{
    if (!scene.d_bvhNodes || scene.totalTriangles == 0)
        return make_float3(0.0f, 0.0f, 0.0f);

    auto traceShadow = [&](float3 origin, float3 dir, float dist) -> float3 {
        // CUDA's bvh_anyHit takes a target point; for finite light distances
        // we test [origin, origin + dir * (dist - 0.002)] so the traversal
        // stops just before the light surface (matching OptiX's tmax = d -
        // 0.002 convention). Directional NEE passes dist = 1e30 which we
        // cap to a 1e10 synthetic target.
        float t = (dist >= 1.0e29f) ? 1.0e10f : fmaxf(dist - 0.002f, 0.001f);
        float3 target = origin + dir * t;
        if (bvh_anyHit(origin, target, scene.d_bvhNodes, scene.bvhRootIndex,
                       scene.d_positions, scene.d_indices))
            return make_float3(0.0f, 0.0f, 0.0f);
        return make_float3(1.0f, 1.0f, 1.0f);
    };
    float3 Li = restirAreaLightNEE(scene, s, rng, traceShadow, /*fireflyClamp=*/25.0f);
    Li = Li + restirDirectionalLightsNEE(scene, s, traceShadow);
    return Li;
}

// Resolve a BVH closest-hit into shading attributes via the canonical
// restirDecodeHit (ReSTIRDevice.cuh). Output uses `hit.position` directly
// rather than re-interpolating from vertices — both yield the same point to
// floating-point precision but the BVH-fed value matches the closest-hit's
// own parametric position, keeping reservoir/visibility math consistent.
__device__ inline bool ptShadeHit(
    const DeviceSceneData& scene, const Ray& ray, const HitRecord& hit,
    float3& outPos, float3& outN, float3& outAlbedo, float3& outEmission,
    float& outRoughness, float& outMetallic, bool& outPureDiffuse)
{
    ReSTIRHitDecode h = restirDecodeHit(
        scene, (uint32_t)hit.primitiveIndex, hit.uv.x, hit.uv.y, ray.direction);
    if (!h.valid) return false;
    outPos         = hit.position;
    outN           = h.normal;
    outAlbedo      = h.albedo;
    outEmission    = h.emission;
    outRoughness   = fmaxf(h.mat.roughness, 0.04f);
    outMetallic    = h.mat.metallic;
    outPureDiffuse = h.pureDiffuse;
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
    float specProb_xr = computeSpecProb(xrN, viewDir, xrAlbedo, xrMetallic);
    ReSTIRSurface curr = ptMakeSurface(xrPos, xrN, xrAlbedo,
                                        xrRoughness, xrMetallic, xrPureDiffuse,
                                        viewDir, specProb_xr);
    L = L + ptDirectLightingAtVertex(scene, curr, rng);

    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    for (uint32_t i = 0; i < bounces; i++) {
        float3 prevPos = curr.position;          // BSDF-sample origin for MIS

        float3 wi;
        float  pdfBsdf = 0.0f;
        if (!restirSampleBsdfDir(curr, rng, wi, pdfBsdf)) break;

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
                float3 envColor = sampleEnvironment(wi, scene.envMapTex);
                float envLum = luminance(envColor);
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

        // BSDF-direct-hits-emitter at this postfix vertex. The previous
        // bounce's NEE (`ptDirectLightingAtVertex` on `curr` before this
        // update) already estimates the same direct lighting → balance with
        // powerHeuristic(pdfBsdf, pdfArea_solid). Without MIS the two
        // estimators sum to ~2× direct, matching the "everything brighter"
        // PT symptom. Helper in ReSTIRDevice.cuh.
        // Historical note: this path used to skip MIS entirely with a
        // comment about emitter-heavy scenes; revisit M7 darkening if MIS
        // here regresses that case.
        if (luminance(hEmis) > 0.0f) {
            float misEmis = restirBsdfHitsEmitterMISWeight(
                scene, (uint32_t)nh.primitiveIndex, prevPos, hPos, pdfBsdf);
            L = L + throughput * hEmis * misEmis;
        }

        float3 nViewDir = -wi;
        float specProb = computeSpecProb(hN, nViewDir, hAlbedo, hMetallic);
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
    float lum = luminance(L);
    const float clampMax = 200.0f;
    if (lum > clampMax) L = L * (clampMax / lum);
    return L;
}

// Map a screen-space pixel into a reuse texture and apply per-frame
// flip/transpose/offset randomisation. Returns the paired neighbor's screen
// coordinates (clamped/discarded by caller via -1 sentinel).
__device__ inline int2 ptReuseTextureLookup(
    uint32_t x, uint32_t y, uint32_t width, uint32_t height,
    const int2* tex, uint32_t texSize, uint32_t flipBits,
    uint32_t offX, uint32_t offY)
{
    // Per-frame offset (mod texSize).
    uint32_t tx = (x + offX) % texSize;
    uint32_t ty = (y + offY) % texSize;
    // Per-frame flip / transpose. flipBits[0]=flipX, [1]=flipY, [2]=transpose.
    if (flipBits & 1u) tx = texSize - 1u - tx;
    if (flipBits & 2u) ty = texSize - 1u - ty;
    if (flipBits & 4u) { uint32_t t = tx; tx = ty; ty = t; }
    int2 d = tex[ty * texSize + tx];
    if (flipBits & 4u) { int t = d.x; d.x = d.y; d.y = t; }
    if (flipBits & 1u) d.x = -d.x;
    if (flipBits & 2u) d.y = -d.y;
    int nx = (int)x + d.x;
    int ny = (int)y + d.y;
    if (nx < 0 || ny < 0 || nx >= (int)width || ny >= (int)height)
        return make_int2(-1, -1);
    return make_int2(nx, ny);
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
    Ray ray = generateRay(x, y, width, height, camera, jx, jy);

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
    surf.specProb    = computeSpecProb(hN, surf.viewDir, hAlbedo, hMetallic);

    float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hPos);
    surf.prevPixel  = make_float2((clipPrev.x + 1.0f) * 0.5f * width,
                                   (1.0f - clipPrev.y) * 0.5f * height);

    // ── Generate `numCandidates` independent paths and stream into RIS ──
    // ReSTIR PT Enhanced §6.1 — unify DI + GI in one reservoir.
    //
    // For each candidate slot we draw TWO competing samples and stream both:
    //   (a) BSDF-sampled continuation path (d ≥ 2; d=2 hits an emitter, d≥3
    //       runs the postfix random walk). Source PDF = pdfBsdf at x_1.
    //   (b) NEE-sampled direct lighting path (d = 2). Source PDF = solid-
    //       angle PDF of the chosen light point.
    //
    // RIS already implements balance-heuristic-style technique mixing when
    // each candidate is streamed with its own true source PDF — the
    // reservoir picks proportional to pHat / pSrc, so a sample more likely
    // to be drawn by one technique is upweighted there. This means no
    // separate per-technique MIS bookkeeping is required for unbiasedness.
    // Spatial/temporal reuse work unchanged: short (d=2) and long (d≥3)
    // paths share the same reconnection-shift Jacobian since the held
    // sample exposes (x_r, n_r, L_o) the same way for both.
    float wSum = 0.0f;
    float xrRoughnessKept = 1.0f;        // stored alongside the kept sample

    if (numCandidates < 1) numCandidates = 1;
    for (uint32_t k = 0; k < numCandidates; k++) {
#if PT_UNIFY_DIRECT_LIGHTING
        // ── Candidate (b): NEE-sampled d=2 direct lighting path ────────────
        // Tries first because it is cheap (one shadow ray, no path postfix).
        // M is incremented for both sub-candidates so the |R| accounting
        // tracks ALL canonical draws (paper §5.5 / §5.7).
        //
        // DISABLED BY DEFAULT: requires the main path-trace kernel to *not*
        // also accumulate primary-hit direct lighting; otherwise the d=2
        // contribution is double-counted (host-side patch needed). Until
        // that landing point is added, leaving this off keeps PT as
        // indirect-only (matching the published Lin et al. 2022 split) and
        // lets the kernel's existing NEE handle d=2 correctly.
        if (scene.d_areaLights && scene.areaLightCount > 0 &&
            scene.d_lightBVHNodes && scene.d_bvhNodes && scene.totalTriangles > 0)
        {
            uint32_t slot = 0; float pSelect = 0.0f;
            if (lightBVH_sample(scene.d_lightBVHNodes, scene.lightBVHRootIndex,
                                surf.position, pcg32_float(rng), slot, pSelect) &&
                pSelect > 0.0f)
            {
                uint32_t lightIdx = scene.d_lightOrderedIndices[slot];
                GPUAreaLight light = scene.d_areaLights[lightIdx];
                float r1 = pcg32_float(rng), r2 = pcg32_float(rng);
                float su = sqrtf(r1);
                float bb0 = 1.0f - su;
                float bb1 = su * (1.0f - r2);
                float bb2 = su * r2;
                float3 lp = light.v0 * bb0 + (light.v0 + light.e1) * bb1
                                            + (light.v0 + light.e2) * bb2;
                float3 toL = lp - surf.position;
                float d2 = fmaxf(dot(toL, toL), 1e-6f);
                float dlen = sqrtf(d2);
                float3 Ldir = toL * (1.0f / dlen);
                float NdotLn = fmaxf(dot(surf.normal, Ldir), 0.0f);
                float lightCos = fmaxf(dot(light.normal, -Ldir), 0.0f);
                if (NdotLn > 0.0f && lightCos > 0.0f) {
                    float3 shadowOrig = surf.position + surf.normal * 0.001f;
                    bool occluded = bvh_anyHit(shadowOrig, lp,
                                                scene.d_bvhNodes, scene.bvhRootIndex,
                                                scene.d_positions, scene.d_indices);
                    if (!occluded) {
                        float3 Le = light.emission;
                        if (light.emissiveTex != 0) {
                            float texU = light.uv0.x*bb0 + light.uv1.x*bb1 + light.uv2.x*bb2;
                            float texV = light.uv0.y*bb0 + light.uv1.y*bb1 + light.uv2.y*bb2;
                            float4 et = tex2D<float4>(light.emissiveTex, texU, texV);
                            Le = make_float3(et.x, et.y, et.z) * light.emission;
                        }
                        // Source PDF in solid angle: pTri / area * d^2 / cosL.
                        float pArea = pSelect / fmaxf(light.area, 1e-7f);
                        float pdfNEE = pArea * d2 / fmaxf(lightCos, 1e-7f);
                        if (pdfNEE > 0.0f) {
                            // Build a candidate reservoir: reconnect vertex
                            // is the light point; sample radiance is Le.
                            GIReservoir cand{};
                            cand.visiblePos     = surf.position;
                            cand.visibleNormal  = surf.normal;
                            cand.samplePos      = lp;
                            cand.sampleNormal   = light.normal;
                            cand.sampleRadiance = Le;
                            cand.isEnv          = 0u;
                            cand.valid          = 1u;
                            // Light surface counts as fully rough for the
                            // reconnection gate (it's not a BSDF).
                            cand.xrRoughness    = 1.0f;
                            float3 wiOut;
                            float pHatNEE = giEvalTargetPdf(surf, cand, wiOut);
                            if (pHatNEE > 0.0f) {
                                bool repNEE = gris_streamCandidate(
                                    r, wSum,
                                    surf.position, surf.normal,
                                    false, lp, light.normal, Le,
                                    pHatNEE, pdfNEE, pcg32_float(rng));
                                if (repNEE) xrRoughnessKept = 1.0f;
                            } else {
                                r.M += 1.0f;       // failed candidate slot
                            }
                        } else {
                            r.M += 1.0f;
                        }
                    } else {
                        r.M += 1.0f;
                    }
                } else {
                    r.M += 1.0f;
                }
            } else {
                r.M += 1.0f;
            }
        }
#endif // PT_UNIFY_DIRECT_LIGHTING

        // ── Candidate (a): BSDF-sampled multi-bounce path ──────────────────
        float3 wi;
        float  pdfBsdf = 0.0f;
        if (!restirSampleBsdfDir(surf, rng, wi, pdfBsdf)) {
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
                float3 envColor = sampleEnvironment(wi, scene.envMapTex);
                float envLum = luminance(envColor);
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
                // MIS-balance the BSDF-direct-hits-emitter contribution at x_r
                // against the path tracer's primary-NEE at q. Without this, an
                // x_r that lands on the area light double-counts direct
                // lighting (path-tracer NEE adds it with powerHeuristic(area,
                // bsdf), and we'd add the same direct again at full weight in
                // the postfix's `L = xrEmis` term). Helper in NEEHelpers.cuh.
                float3 xEmisMIS = xEmis * restirBsdfHitsEmitterMISWeight(
                    scene, (uint32_t)hit2.primitiveIndex,
                    hPos, xPos, pdfBsdf);
                Lo = ptPathPostfix(scene,
                                   xPos, xN, xAlbedo, xEmisMIS,
                                   xRoughness, xMetallic, xPure,
                                   viewAtXr,
                                   enableEnvironment != 0,
                                   pathLength,
                                   rng);
                candPos    = xPos;
                candNormal = xN;
                isEnvCand  = false;
                candXrRough = xRoughness;
                ok = (luminance(Lo) > 0.0f);
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
    const float*       dupPrev,                // §5 — prev-frame duplication map
    uint32_t width, uint32_t height,
    uint32_t prevWidth, uint32_t prevHeight,
    uint32_t sampleIndex,
    uint32_t frameIndex,                       // monotonic, never reset
    uint32_t mCap,
    uint32_t mCapMin)                          // §5 minimum cap for D=1
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;
    uint32_t seedSalt = sampleIndex + frameIndex * 0x9E3779B9u;

    GIReservoir r = curr[pixelIdx];
    ReSTIRSurface s = surfCurr[pixelIdx];
    if (s.valid < 0.5f) return;

    // ── Reprojection with dual-MV fallback (paper §6.4 — ReSTIR PT Enhanced)
    // Disocclusion at the primary motion vector returns no usable history.
    // Following Zeng et al. [2021], we try a second reprojection: scan the
    // 3×3 neighborhood around the primary backprojected pixel and pick the
    // first one whose surface still matches our G-buffer. This catches the
    // common case where the foreground occluder's MV points us into the
    // background: a neighbor pixel held the same background surface in the
    // previous frame and its MV is a much better hint. Pure ReSTIR PT
    // resampling already prevents the copy-paste pattern artifacts that
    // forced Zeng to store incident-radiance fields.
    int basePx = (int)floorf(s.prevPixel.x);
    int basePy = (int)floorf(s.prevPixel.y);
    // 3×3 search (center first, then ring) for a previous-frame pixel whose
    // surface still matches. 9 candidates total — center is the primary MV,
    // the 8 ring neighbors form the dual-MV fallback (Zeng 2021 simplified).
    const int dxs[9] = {  0,  0,  1,  0, -1,  1,  1, -1, -1 };
    const int dys[9] = {  0,  1,  0, -1,  0,  1, -1,  1, -1 };
    int px = -1, py = -1;
    for (int k = 0; k < 9; k++) {
        int tx = basePx + dxs[k];
        int ty = basePy + dys[k];
        if (tx < 0 || ty < 0 ||
            tx >= (int)prevWidth || ty >= (int)prevHeight) continue;
        uint32_t idx = (uint32_t)ty * prevWidth + (uint32_t)tx;
        ReSTIRSurface ss = surfPrev[idx];
        if (ss.valid < 0.5f) continue;
        if (dot(s.normal, ss.normal) < 0.9f) continue;
        float dr = length(s.position - ss.position);
        if (dr > 0.1f * fmaxf(length(s.position), 1.0f)) continue;
        px = tx; py = ty;
        break;
    }
    if (px < 0) return;
    uint32_t prevIdx = (uint32_t)py * prevWidth + (uint32_t)px;

    ReSTIRSurface sp = surfPrev[prevIdx];

    GIReservoir pr = prev[prevIdx];
    if (!pr.valid) return;

    // ── §5 adaptive cCap reduction via duplication map ─────────────────
    // The previous frame's duplication score at this reprojected pixel
    // tells us how strongly the prior reservoir's sample is correlated
    // with its spatial neighbours. The paper uses
    //     cCap = lerp(cDefault, cMin, D^alpha)
    // The paper's α=0.1 is too aggressive for our quantised-position-hash
    // duplication proxy: even legitimate global-illumination convergence
    // (where many pixels truly converge to the same dominant light path)
    // produces D≈0.05–0.15 in 17×17 windows, which under α=0.1 collapses
    // cCap to ~cMin=1 and prevents any temporal accumulation — visible as
    // dark output. We use a dead zone (D < 0.25 → no scaling) plus a
    // gentler α=2 so that only genuinely correlated regions trigger cap
    // reduction. This matches the paper's intent (kill correlation
    // fireflies) while preserving normal temporal accumulation everywhere
    // else.
    float effMCap = (float)mCap;
#if PT_ENABLE_DUPMAP_CAP
    if (dupPrev) {
        float dupScore = dupPrev[prevIdx];
        const float deadZone = 0.25f;
        if (dupScore > deadZone) {
            float remapped = (dupScore - deadZone) / (1.0f - deadZone);
            float t = remapped * remapped;             // α=2, gentle
            float cMin = (float)mCapMin;
            effMCap = (float)mCap * (1.0f - t) + cMin * t;
            if (effMCap < cMin) effMCap = cMin;
        }
    }
#else
    (void)dupPrev;
    (void)mCapMin;
#endif
    gris_capM(pr, effMCap);                    // §6.4 + §5 adaptive cap

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

// ─────────────────────────────────────────────────────────────────────────────
// Paired spatial reuse kernel (paper §3, ReSTIR PT Enhanced)
// Replaces the random uniform-disk neighbor sampler with a permutation-texture
// pairing scheme: each frame, pixel A is deterministically paired with pixel
// B such that A reuses B and B reuses A in lock-step. The Gaussian-shaped
// distance distribution (matched in mean to the prior R=30 disk) yields
// better-quality reuse and is a prerequisite for the §3 cost halving.
//
// Each spatial neighbor slot has its own reuse texture (different sizes to
// avoid period beats); per-frame flip/transpose/offset transforms break
// long-term correlation patterns.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kReSTIRPT_Spatial(
    DeviceSceneData scene,
    const GIReservoir* inRes,
    GIReservoir*       outRes,
    const ReSTIRSurface* surf,
    float3*            outShadeW,             // §6.3 — vector w sum
    const int2*        reuseTex0,
    uint32_t           reuseTex0Size,
    const int2*        reuseTex1,
    uint32_t           reuseTex1Size,
    const int2*        reuseTex2,
    uint32_t           reuseTex2Size,
    uint32_t           reuseFlip0,
    uint32_t           reuseFlip1,
    uint32_t           reuseFlip2,
    uint32_t           reuseOff0X, uint32_t reuseOff0Y,
    uint32_t           reuseOff1X, uint32_t reuseOff1Y,
    uint32_t           reuseOff2X, uint32_t reuseOff2Y,
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
    // Up to 3 reuse-texture slots; if more neighbors are requested, the
    // remaining slots fall back to random disk sampling.
    const int2*    rtex[3]  = { reuseTex0, reuseTex1, reuseTex2 };
    uint32_t       rsize[3] = { reuseTex0Size, reuseTex1Size, reuseTex2Size };
    uint32_t       rflip[3] = { reuseFlip0, reuseFlip1, reuseFlip2 };
    uint32_t       roffX[3] = { reuseOff0X, reuseOff1X, reuseOff2X };
    uint32_t       roffY[3] = { reuseOff0Y, reuseOff1Y, reuseOff2Y };
    (void)radiusPixels;

    for (uint32_t i = 0; i < cap; i++) {
        int nx = -1, ny = -1;
        if (i < 3 && rtex[i] && rsize[i] > 0) {
            int2 paired = ptReuseTextureLookup(x, y, width, height,
                                                rtex[i], rsize[i],
                                                rflip[i], roffX[i], roffY[i]);
            nx = paired.x;
            ny = paired.y;
        } else {
            // Fallback for slots beyond the prepared paired textures.
            float u1 = pcg32_float(rng);
            float u2 = pcg32_float(rng);
            float rr = sqrtf(u1) * radiusPixels;
            float th = 2.0f * M_PI_F * u2;
            nx = (int)x + (int)(rr * cosf(th));
            ny = (int)y + (int)(rr * sinf(th));
        }
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

    // ReSTIR PT Enhanced §6.3 — spatial merge accumulates a vector-valued
    // shade weight Σ m_i · F(y_i) · W_i · |J| in parallel with the scalar
    // resampling weights. Shade kernel divides by reservoir.pHat to compute
    // the final estimator, suppressing chroma noise at no extra cost.
    float3 shadeW = make_float3(0, 0, 0);
    if (collected > 0) {
        gris_mergeMultiPairVec(r, s, peers, peerS, collected, u01s, shadeW);
    } else if (r.valid && r.pHat > 0.0f && r.W > 0.0f) {
        // No peers admitted — fall back to held-only vector weight so the
        // shade kernel can still use the vector path uniformly.
        // m_dst = 1 in this case (held term is the only term).
        float3 wi_R, F_R;
        float pHatR_eval;
        if (giEvalTargetPdfVec(s, r, wi_R, pHatR_eval, F_R)) {
            shadeW = F_R * r.W;       // m=1, jac=1
            if (!isfinite(shadeW.x) || !isfinite(shadeW.y) || !isfinite(shadeW.z))
                shadeW = make_float3(0, 0, 0);
        }
    }
    outRes[pixelIdx] = r;
    if (outShadeW) outShadeW[pixelIdx] = shadeW;
    (void)scene;
}

// ─────────────────────────────────────────────────────────────────────────────
// Duplication-map kernel (paper §5, ReSTIR PT Enhanced)
//
// For each pixel, count how many reservoirs in the surrounding 17×17 window
// share the current pixel's sample. We approximate "shared sample" via a
// quantised hash of samplePos: two reservoirs hashing to the same cell are
// likely the same reconnection vertex (and thus the same physical sample).
//
// The score D = (count - 1) / 288 in [0, 1] — 288 = 17×17 - 1 excluding self,
// matching paper's normalisation (their max is reached when every neighbour
// shares the seed). cCap is then lerp'd toward `cMinCap` by D^alpha; this
// trades a small amount of bias for a large reduction in correlation
// fireflies that would otherwise persist for ~cCap frames.
//
// We compute on samplePos (and isEnv as a bucket disambiguator) since pure-
// reconnection reservoirs don't carry the random seed the paper uses; the
// quantisation cell size is set to roughly 1cm world space, which suffices
// because two paths that find genuinely the same reconnection vertex will
// land within a single cell almost always.
// ─────────────────────────────────────────────────────────────────────────────
#ifndef PT_DUP_WINDOW
#define PT_DUP_WINDOW 17        // 17×17 → 288 non-centre cells (paper §5)
#endif
#ifndef PT_DUP_CELL
#define PT_DUP_CELL   0.01f     // quantisation cell size (world space, m)
#endif

__device__ inline uint32_t ptSampleHash(const GIReservoir& r) {
    if (!r.valid) return 0u;
    if (r.isEnv) {
        // Direction quantisation — coarser since unit sphere is small.
        int qx = (int)floorf(r.samplePos.x * 64.0f);
        int qy = (int)floorf(r.samplePos.y * 64.0f);
        int qz = (int)floorf(r.samplePos.z * 64.0f);
        uint32_t h = 0x9E3779B9u ^ (uint32_t)qx * 0x85EBCA6Bu;
        h ^= (uint32_t)qy * 0xC2B2AE35u;
        h ^= (uint32_t)qz * 0x27D4EB2Fu;
        return h | 1u;          // ensure non-zero so 0 means "no sample"
    }
    int qx = (int)floorf(r.samplePos.x * (1.0f / PT_DUP_CELL));
    int qy = (int)floorf(r.samplePos.y * (1.0f / PT_DUP_CELL));
    int qz = (int)floorf(r.samplePos.z * (1.0f / PT_DUP_CELL));
    uint32_t h = (uint32_t)qx * 0x9E3779B9u;
    h ^= (uint32_t)qy * 0x85EBCA6Bu;
    h ^= (uint32_t)qz * 0xC2B2AE35u;
    return h | 1u;
}

__global__ void kReSTIRPT_DuplicationMap(
    const GIReservoir* res,
    float*             outDup,
    uint32_t           width,
    uint32_t           height)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    GIReservoir self = res[pixelIdx];
    uint32_t hSelf = ptSampleHash(self);
    if (hSelf == 0u) {
        outDup[pixelIdx] = 0.0f;
        return;
    }
    const int half = PT_DUP_WINDOW / 2;
    uint32_t dup = 0;
    for (int dy = -half; dy <= half; dy++) {
        int ny = (int)y + dy;
        if (ny < 0 || ny >= (int)height) continue;
        for (int dx = -half; dx <= half; dx++) {
            int nx = (int)x + dx;
            if (nx < 0 || nx >= (int)width) continue;
            if (dx == 0 && dy == 0) continue;
            uint32_t nIdx = (uint32_t)ny * width + (uint32_t)nx;
            uint32_t hN = ptSampleHash(res[nIdx]);
            if (hN == hSelf) dup++;
        }
    }
    // Normalise to [0, 1]. 288 = 17×17 − 1 (paper §5).
    const float kInvMax = 1.0f / 288.0f;
    float dupScore = (float)dup * kInvMax;
    if (dupScore > 1.0f) dupScore = 1.0f;
    outDup[pixelIdx] = dupScore;
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
    const float3*        shadeWeights,        // §6.3 vector resampling sum
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
    // DIAG_PT_FORCE_BRIGHT: set to 1 to force shade output to (0.5, 0, 0)
    // — quick check that consumer side is actually reading shadeWeights /
    // computing L. Should make the entire scene appear pink-tinted when on.
#define DIAG_PT_FORCE_BRIGHT 0
    if (s.valid >= 0.5f && r.valid && r.W > 0.0f) {
#if DIAG_PT_FORCE_BRIGHT
        L = make_float3(0.5f, 0.0f, 0.0f);
        outIndirect[pixelIdx] = L;
        return;
#endif
#if PT_ENABLE_VECTOR_SHADING
        // ── ReSTIR PT Enhanced §6.3 — vector-valued estimator ─────────────
        //   L = Σ m_i · F(y_i) · W_i · |J|   (paper §6.3, footnote 7)
        // Fallback to scalar estimator when shadeWeights is empty (e.g.
        // motion frames that skip spatial reuse).
        float3 wVec = (shadeWeights ? shadeWeights[pixelIdx] : make_float3(0,0,0));
        if (wVec.x == 0.0f && wVec.y == 0.0f && wVec.z == 0.0f) {
            float3 wi;
            float r2 = 0, cosQ = 0, cosS = 0;
            if (giConnect(s.position, s.normal, r, wi, r2, cosQ, cosS)) {
                float3 brdf = restirEvalBrdf(s, wi);
                L = brdf * r.sampleRadiance * (cosQ * r.W);
            }
        } else {
            L = wVec;
        }
#else
        (void)shadeWeights;
        float3 wi;
        float r2 = 0, cosQ = 0, cosS = 0;
        if (giConnect(s.position, s.normal, r, wi, r2, cosQ, cosS)) {
            float3 brdf = restirEvalBrdf(s, wi);
            L = brdf * r.sampleRadiance * (cosQ * r.W);
        }
#endif
        // Per-frame firefly clamp on the final estimator. 50 lum bounds
        // runaway samples without darkening — the upstream NEE + path-
        // postfix clamps + RIS weight cap already cap typical contributions.
        // Earlier we used 8, which was a defensive over-tightening that
        // visibly desaturated/darkened the ReSTIR PT result.
        float lum = luminance(L);
        const float clampMax = 50.0f;
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
    // §5 — minimum mCap when duplication = 1 (paper recommends 1).
    const uint32_t kMCapMin = 1;
    kReSTIRPT_Temporal<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_reservoirsPrev,
        buffers.d_surfaceCurr,    buffers.d_surfacePrev,
        buffers.d_duplicationPrev,           // prev-frame duplication map
        width, height, buffers.prevWidth, buffers.prevHeight,
        sampleIndex, frameIndex, temporalMCap, kMCapMin);
}

// §5 — duplication map computation, runs once at end of frame after spatial.
void launchReSTIRPTDuplicationMap(
    PTBuffers buffers, uint32_t width, uint32_t height)
{
    if (!buffers.d_duplicationCurr) return;
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIRPT_DuplicationMap<<<grid, block>>>(
        buffers.d_reservoirsCurr, buffers.d_duplicationCurr, width, height);
}

// Launcher takes pointers/sizes/transform state for up to 3 reuse textures.
// Caller (ReSTIRPTContext::runFrame) prepares them every frame.
struct PTReuseLaunchTex {
    const int2* d_offsets = nullptr;
    uint32_t    size      = 0;
    uint32_t    flipBits  = 0;
    uint32_t    offX      = 0;
    uint32_t    offY      = 0;
};

void launchReSTIRPTSpatialReusePaired(
    const DeviceSceneData& scene,
    PTBuffers              buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               frameIndex,
    uint32_t               numNeighbors,
    float                  radiusPixels,
    uint32_t               spatialMCap,
    const PTReuseLaunchTex* tex,        // length 3
    uint32_t               numTex)
{
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    PTReuseLaunchTex t[3] = {};
    for (uint32_t i = 0; i < numTex && i < 3u; i++) t[i] = tex[i];
    kReSTIRPT_Spatial<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr, buffers.d_reservoirsSpatial,
        buffers.d_surfaceCurr,
        buffers.d_shadeWeights,                 // §6.3 — vector w sum
        t[0].d_offsets, t[0].size,
        t[1].d_offsets, t[1].size,
        t[2].d_offsets, t[2].size,
        t[0].flipBits, t[1].flipBits, t[2].flipBits,
        t[0].offX, t[0].offY,
        t[1].offX, t[1].offY,
        t[2].offX, t[2].offY,
        width, height, sampleIndex, frameIndex,
        numNeighbors, radiusPixels, spatialMCap);
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
    // Source-compatible shim — falls back to random disk (no paired textures)
    // when the host hasn't prepared paired textures.
    launchReSTIRPTSpatialReusePaired(
        scene, buffers, width, height, sampleIndex, frameIndex,
        numNeighbors, radiusPixels, spatialMCap, nullptr, 0);
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
        buffers.d_shadeWeights,                 // §6.3
        buffers.d_indirectOut, width, height);
}

// ─────────────────────────────────────────────────────────────────────────────
// Paired spatial reuse — permutation reuse-texture builder (paper §3.1)
//
// We initialise a square texture with consecutive link indices, then perform
// `n_sigma` rounds of 2×2 block shuffles, alternating between an aligned grid
// and a (1,1)-offset grid. This produces a self-inverting permutation whose
// per-pixel coordinate delta (after mod-tiling) is approximately Gaussian-
// distributed with the requested standard deviation σ.
//
// Long-distance links wrap around the texture edge — we shorten them so the
// texture is tileable: any delta > size/2 has `size` subtracted; any delta <
// -size/2 has `size` added.
// ─────────────────────────────────────────────────────────────────────────────
static uint32_t computeShuffleRepeats(float sigma) {
    // Paper Eq. 3 — function-fit correction for small σ. Floor + 0.5 rounds.
    float v = sigma * sigma * 0.5f
            + 1.46f / sigma + 1.76f / (sigma * sigma)
            + 0.656f / (sigma * sigma * sigma) + 0.5f;
    if (v < 1.0f) v = 1.0f;
    return (uint32_t)v;
}

static std::vector<int2> buildReuseTextureCPU(uint32_t size, float sigma,
                                               std::mt19937& rng)
{
    const uint32_t total = size * size;
    // Pixel index → owning link index. Initialised so consecutive 2-pixel
    // pairs share a link (paper Fig. 3 left).
    std::vector<uint32_t> link(total);
    for (uint32_t i = 0; i < total; i++) link[i] = i / 2u;

    // Helper: shuffle each 2×2 block at the given (offX, offY) tiling.
    auto shuffleBlock = [&](int offX, int offY) {
        for (uint32_t by = 0; by < size; by += 2) {
            for (uint32_t bx = 0; bx < size; bx += 2) {
                // Four pixel slots in this 2×2 block (with toroidal offset).
                uint32_t idx[4];
                int xs[4] = { (int)bx + offX, (int)bx + 1 + offX,
                              (int)bx + offX, (int)bx + 1 + offX };
                int ys[4] = { (int)by + offY, (int)by + offY,
                              (int)by + 1 + offY, (int)by + 1 + offY };
                for (int k = 0; k < 4; k++) {
                    int xx = ((xs[k] % (int)size) + (int)size) % (int)size;
                    int yy = ((ys[k] % (int)size) + (int)size) % (int)size;
                    idx[k] = (uint32_t)yy * size + (uint32_t)xx;
                }
                // Random permutation of the four link indices among slots.
                uint32_t perm[4] = { link[idx[0]], link[idx[1]],
                                      link[idx[2]], link[idx[3]] };
                for (int k = 3; k > 0; k--) {
                    std::uniform_int_distribution<int> d(0, k);
                    int j = d(rng);
                    std::swap(perm[k], perm[j]);
                }
                for (int k = 0; k < 4; k++) link[idx[k]] = perm[k];
            }
        }
    };

    uint32_t n = computeShuffleRepeats(sigma);
    for (uint32_t i = 0; i < n; i++) {
        // Alternate aligned and (1,1)-offset 2×2 tilings (paper Fig. 3).
        shuffleBlock(0, 0);
        shuffleBlock(1, 1);
    }

    // Build the inverse: for each link id, the two pixels owning it.
    std::vector<int> first(total, -1);
    std::vector<int> second(total, -1);
    for (uint32_t p = 0; p < total; p++) {
        uint32_t L = link[p];
        if (first[L] < 0) first[L] = (int)p;
        else if (second[L] < 0) second[L] = (int)p;
        // If a link slot already has 2 owners (shouldn't happen with valid
        // permutations), the extra pixels are dropped — they remain self-
        // paired below.
    }

    std::vector<int2> deltas(total);
    for (uint32_t p = 0; p < total; p++) {
        uint32_t L = link[p];
        int partner = -1;
        if (first[L] == (int)p) partner = second[L];
        else if (second[L] == (int)p) partner = first[L];
        if (partner < 0) {
            deltas[p] = make_int2(0, 0);    // self-pair fallback
            continue;
        }
        int sx = (int)(p % size), sy = (int)(p / size);
        int tx = (int)((uint32_t)partner % size), ty = (int)((uint32_t)partner / size);
        int dx = tx - sx, dy = ty - sy;
        // Tileable wrap: shortest signed delta in either axis.
        int half = (int)size / 2;
        if (dx >  half) dx -= (int)size;
        if (dx < -half) dx += (int)size;
        if (dy >  half) dy -= (int)size;
        if (dy < -half) dy += (int)size;
        deltas[p] = make_int2(dx, dy);
    }
    return deltas;
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
    // §6.3 — vector-valued shade weights; cleared each frame by the spatial
    // kernel write, so init clear is for safety only.
    CUDA_CHECK(cudaMalloc(&m_buffers.d_shadeWeights, count * sizeof(float3)));
    CUDA_CHECK(cudaMemset(m_buffers.d_shadeWeights, 0, count * sizeof(float3)));
    // §5 — duplication map ping-pong.
    CUDA_CHECK(cudaMalloc(&m_buffers.d_duplicationCurr, count * sizeof(float)));
    CUDA_CHECK(cudaMemset(m_buffers.d_duplicationCurr, 0, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_buffers.d_duplicationPrev, count * sizeof(float)));
    CUDA_CHECK(cudaMemset(m_buffers.d_duplicationPrev, 0, count * sizeof(float)));
    m_buffers.width  = width;
    m_buffers.height = height;
    m_buffers.prevWidth  = width;
    m_buffers.prevHeight = height;
    m_buffers.historyValid = false;

    // ── §3 paired spatial reuse — build one-time permutation textures ─────
    // Sizes chosen to be near-coprime within typical 1080p widths so tiled
    // repeats don't beat. σ matched to mean disk distance for R≈30 (paper
    // §7 "Performance"): σ = sqrt(8 / (9π)) · R ≈ 16.0 for R=30.
    if (!m_reuseTexBuilt) {
        const uint32_t sizes[kPTReuseTexCount] = { 254, 230, 210 };
        const float    sigma = sqrtf(8.0f / (9.0f * 3.14159265358979323846f))
                              * m_spatialRadius;
        std::mt19937 rng(0xCA11AB1Eu);
        for (uint32_t i = 0; i < kPTReuseTexCount; i++) {
            std::vector<int2> cpu = buildReuseTextureCPU(sizes[i], sigma, rng);
            int2* d = nullptr;
            CUDA_CHECK(cudaMalloc(&d, cpu.size() * sizeof(int2)));
            CUDA_CHECK(cudaMemcpy(d, cpu.data(), cpu.size() * sizeof(int2),
                                   cudaMemcpyHostToDevice));
            m_reuseTex[i].d_offsets = d;
            m_reuseTex[i].size      = sizes[i];
        }
        m_reuseTexBuilt = true;
    }
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
    if (m_buffers.d_shadeWeights)      cudaFree(m_buffers.d_shadeWeights);
    if (m_buffers.d_duplicationCurr)   cudaFree(m_buffers.d_duplicationCurr);
    if (m_buffers.d_duplicationPrev)   cudaFree(m_buffers.d_duplicationPrev);
    for (uint32_t i = 0; i < kPTReuseTexCount; i++) {
        if (m_reuseTex[i].d_offsets) cudaFree(m_reuseTex[i].d_offsets);
        m_reuseTex[i] = PTReuseTexture{};
    }
    m_reuseTexBuilt = false;
    m_buffers = PTBuffers{};
}

void ReSTIRPTContext::swapHistory() {
    GIReservoir* tr = m_buffers.d_reservoirsCurr;
    m_buffers.d_reservoirsCurr = m_buffers.d_reservoirsPrev;
    m_buffers.d_reservoirsPrev = tr;
    ReSTIRSurface* ts = m_buffers.d_surfaceCurr;
    m_buffers.d_surfaceCurr = m_buffers.d_surfacePrev;
    m_buffers.d_surfacePrev = ts;
    // §5 — duplication map: this-frame's curr becomes next-frame's prev.
    float* td = m_buffers.d_duplicationCurr;
    m_buffers.d_duplicationCurr = m_buffers.d_duplicationPrev;
    m_buffers.d_duplicationPrev = td;
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

    // §3 — assemble per-frame paired-spatial transform state.
    // While the camera is moving, slow the rotation of flip/offset every-N
    // frames instead of every frame: per-pixel paired-neighbour selection
    // changing every frame compounds with reprojection error to produce
    // visible per-frame jitter. With slower rotation, neighbouring frames
    // share most of their reuse pattern, letting the temporal/spatial
    // reservoirs settle. Static camera keeps the original 1-frame rotation
    // (decorrelation needed when reservoirs persist with high M).
    PTReuseLaunchTex tex[kPTReuseTexCount] = {};
    uint32_t prepared = 0;
    if (m_reuseTexBuilt) {
        const uint32_t period = cameraMoved ? 4u : 1u;
        const uint32_t slow   = camera.frameIndex / period;
        for (uint32_t i = 0; i < kPTReuseTexCount; i++) {
            if (!m_reuseTex[i].d_offsets) break;
            tex[prepared].d_offsets = m_reuseTex[i].d_offsets;
            tex[prepared].size      = m_reuseTex[i].size;
            uint32_t h = slow * 0x9E3779B9u + i * 0x85EBCA6Bu;
            h ^= h >> 16;
            tex[prepared].flipBits = h & 7u;
            tex[prepared].offX     = (h >> 3) % m_reuseTex[i].size;
            tex[prepared].offY     = (h >> 13) % m_reuseTex[i].size;
            prepared++;
        }
    }
    launchReSTIRPTSpatialReusePaired(
        scene, m_buffers, width, height,
        sampleIndex, camera.frameIndex,
        m_numNeighbors, m_spatialRadius, m_spatialMCap,
        tex, prepared);
    GIReservoir* t = m_buffers.d_reservoirsCurr;
    m_buffers.d_reservoirsCurr    = m_buffers.d_reservoirsSpatial;
    m_buffers.d_reservoirsSpatial = t;

    launchReSTIRPTShade(scene, m_buffers, width, height);
    // §5 — compute duplication map AFTER the swap-back, so it analyses the
    // final reservoir set we just shaded with. swapHistory() promotes this
    // to the prev-frame map for next frame's temporal pass.
    launchReSTIRPTDuplicationMap(m_buffers, width, height);
    return true;
}
