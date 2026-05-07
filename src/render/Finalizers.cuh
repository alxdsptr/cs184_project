#pragma once
// Per-pixel end-of-frame helpers shared between the four path-trace kernels:
//   - Mono pair (PathTraceKernel.cu, OptiXPrograms.cu __raygen__path_trace):
//     accumulate radiance into a single accum buffer, optionally publish to a
//     Vulkan-shared HDR surface for DLSSOnly.
//   - Split pair (PathTraceKernelSplit.cu, OptiXPrograms.cu __raygen__path_trace_split):
//     accumulate per-bucket demodulated radiance, compute DLSS-RR guides, and
//     write the NRD / DLSS-RR surfaces. Caller passes its surface handles
//     directly (CUDA: `surfaces.foo`, OptiX: `params.splitFoo`).
//
// All functions are header-only (`__device__ inline`) so OptiX / CUDA share
// one definition without ODR conflicts.

#include "core/Math.h"
#include "gpu/NRDHelpers.cuh"
#include "render/PathTraceHelpers.cuh"
#include "render/GBufferWriters.cuh"
#include "render/AuxBuffers.h"        // SplitSurfaceOutputs
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────
// Mono pair
// ─────────────────────────────────────────────────────────────────

// Per-spp inside the spp loop: sanitize NaN/inf, luminance-clamp, accumulate.
__device__ inline void monoAccumulateSppSample(float3 radiance, float3& radianceSum) {
    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z) ||
        isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z)) {
        radiance = make_float3(0.0f, 0.0f, 0.0f);
    }
    // The luminance clamp is the only firefly defense in the mono path —
    // NEE contributions don't apply a per-contribution clamp; they rely on
    // this end-of-loop guard.
    float lum = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
    const float clampMax = 200.0f;
    if (lum > clampMax) radiance = radiance * (clampMax / lum);
    radianceSum = radianceSum + radiance;
}

// Once per pixel after the spp loop: accum-buffer add, normalize to HDR,
// write the optional output buffer and the optional Vulkan-shared HDR
// surface (DLSSOnly path).
__device__ inline void monoFinalizePixel(
    float3 radianceSum,
    float4* accumBuffer, float4* outputBuffer,
    cudaSurfaceObject_t hdrColorSurf,
    uint32_t pixelIdx, uint32_t x, uint32_t y,
    uint32_t sampleIndex, uint32_t samplesPerPixel)
{
    // Accumulate: add all spp samples at once. The caller advances the sample
    // counter by `samplesPerPixel`, so the divisor below stays correct.
    float4 sumTexel = make_float4(radianceSum.x, radianceSum.y, radianceSum.z,
                                  (float)samplesPerPixel);
    accumBuffer[pixelIdx] = accumBuffer[pixelIdx] + sumTexel;
    float invN = 1.0f / (float)(sampleIndex + samplesPerPixel);
    float4 hdr = accumBuffer[pixelIdx] * invN;
    if (outputBuffer) outputBuffer[pixelIdx] = hdr;

    // DLSSOnly: also publish HDR into the Vulkan-shared interop image so the
    // post-processing chain can sample it (RGBA16F, alpha=1).
    writeRGBA16F(hdrColorSurf, x, y, make_float4(hdr.x, hdr.y, hdr.z, 1.0f));
}

// ─────────────────────────────────────────────────────────────────
// Split pair
// ─────────────────────────────────────────────────────────────────

// Primary-hit g-buffer slot — describes the visible-surface point one spp
// sample produces. The kernel keeps one of these as a per-spp local while
// stepping the bounce loop, then `splitAccumulateSppSample` copies it into
// `SplitAccumState::primary` on the first sample that lands a primary hit
// (`gbufferWritten` flips to true). Averaging across samples would soften
// silhouettes and break NRD's disocclusion test, so first-sample wins.
//
// `viewZ = 1e6` is the sky sentinel (NRD treats anything beyond its
// denoising range as sky); `ndcZ = 1` is the DLSS far-plane sentinel.
struct PrimaryGBuffer {
    float3 albedo    = {0.0f, 0.0f, 0.0f};
    float3 normal    = {0.0f, 1.0f, 0.0f};
    float  roughness = 1.0f;
    float  metallic  = 0.0f;     // for DLSS-RR specF0 derivation
    float  viewZ     = 1.0e6f;
    float  ndcZ      = 1.0f;
    float2 mvPx      = {0.0f, 0.0f};
    float3 hitPos    = {0.0f, 0.0f, 0.0f};   // for the post-spp DLSS-RR mirror trace
    float3 rayDir    = {0.0f, 0.0f, -1.0f};  // for spec-albedo NoV and the mirror direction
};

// Per-pixel state across spp samples. Filled inside the spp loop by
// `splitAccumulateSppSample`, consumed by `splitFinalizeAndWrite` (which
// computes per-pixel averages locally and writes the surfaces). Default-
// initialize with `SplitAccumState acc{};`.
struct SplitAccumState {
    // Running sums across spp.
    float3   demodDiffSum  = {0.0f, 0.0f, 0.0f};
    float3   demodSpecSum  = {0.0f, 0.0f, 0.0f};
    float3   emissiveSum   = {0.0f, 0.0f, 0.0f};
    float3   noisyColorSum = {0.0f, 0.0f, 0.0f};   // DLSS-RR un-demodulated guide
    float    diffHitSum    = 0.0f;
    uint32_t diffHitCount  = 0;
    float    specHitSum    = 0.0f;
    uint32_t specHitCount  = 0;

    // First-sample-wins primary g-buffer.
    bool           gbufferWritten = false;
    PrimaryGBuffer primary;
};

// Per-spp: sanitize, per-channel clamp, demodulate by albedo, accumulate into
// the per-pixel running state, and capture first-sample primary g-buffer.
__device__ inline void splitAccumulateSppSample(
    SplitAccumState& acc,
    float3 pathRadiance, float3 emissiveContrib,
    bool haveGbuffer, int pickedBucket,
    const PrimaryGBuffer& primary,
    float bucketHitDist, bool bucketHitDistSet)
{
    if (isnan(pathRadiance.x) || isnan(pathRadiance.y) || isnan(pathRadiance.z) ||
        isinf(pathRadiance.x) || isinf(pathRadiance.y) || isinf(pathRadiance.z)) {
        pathRadiance = make_float3(0.0f, 0.0f, 0.0f);
    }
    // Per-channel clamp at 15 (not luminance-only at 200): a luminance-only
    // clamp at 200 lets a single saturated green firefly through at ~280
    // (since g-weight is 0.72); RELAX then takes ~30 frames to fade it. A
    // per-channel cap at 15 kills those spikes hard.
    pathRadiance.x = fminf(fmaxf(pathRadiance.x, 0.0f), 15.0f);
    pathRadiance.y = fminf(fmaxf(pathRadiance.y, 0.0f), 15.0f);
    pathRadiance.z = fminf(fmaxf(pathRadiance.z, 0.0f), 15.0f);

    // Demodulate by albedo so NRD sees the irradiance component; composite
    // remultiplies. Guard against zero albedo (pure metallic → specular bucket).
    float3 demodDiff = make_float3(0.0f, 0.0f, 0.0f);
    float3 demodSpec = make_float3(0.0f, 0.0f, 0.0f);
    if (haveGbuffer) {
        if (pickedBucket == 0) {
            float3 invA = make_float3(
                1.0f / fmaxf(primary.albedo.x, 1e-3f),
                1.0f / fmaxf(primary.albedo.y, 1e-3f),
                1.0f / fmaxf(primary.albedo.z, 1e-3f));
            demodDiff = pathRadiance * invA;
        } else {
            demodSpec = pathRadiance;
        }
    }

    acc.demodDiffSum = acc.demodDiffSum + demodDiff;
    acc.demodSpecSum = acc.demodSpecSum + demodSpec;
    acc.emissiveSum  = acc.emissiveSum  + emissiveContrib;
    // Noisy combined color: pathRadiance already incorporates 1/pickedP, so
    // E_buckets[pathRadiance] = full primary-hit radiance. Adding emissive
    // gives the un-demodulated color DLSS-RR wants.
    acc.noisyColorSum = acc.noisyColorSum + pathRadiance + emissiveContrib;

    // Per-bucket hit distances — only count toward the bucket the sample
    // actually filled. DLSS-RR's spec hit distance comes from an explicit
    // mirror-ray trace in splitFinalizeAndWrite, not from per-bucket
    // averaging (an earlier "anyHitAvg" mixing both lobes flickered
    // frame-to-frame on glossy mirrors).
    if (haveGbuffer && bucketHitDistSet) {
        if (pickedBucket == 0) {
            acc.diffHitSum += bucketHitDist; acc.diffHitCount++;
        } else {
            acc.specHitSum += bucketHitDist; acc.specHitCount++;
        }
    }

    // First-sample-wins primary capture. Averaging normals / viewZ across
    // samples would soften silhouettes and break NRD's disocclusion test.
    if (!acc.gbufferWritten && haveGbuffer) {
        acc.primary        = primary;
        acc.gbufferWritten = true;
    }
}

// HDR-color clamp policy for the DLSS-RR `splitHdrColor` write — the only
// real behavioral divergence between the two split kernels.
//   - PerChannel30: CUDA path. Clamps each channel independently at 30; can
//     shift hue on saturated colors but lets bright RGB through.
//   - MaxChannel10: OptiX path. RTXPT-style — `max3(c) <= 10` (rescale all
//     channels uniformly when the max exceeds the threshold). RR's neural
//     model can't denoise outliers above its training distribution, and a
//     single high-energy firefly bleeds into nearby pixels for multiple
//     frames during motion. RTXPT's PostProcess.hlsl §354 uses this form;
//     per-channel clamping shifts hue on saturated colors which RR misreads
//     as motion.
enum class SplitHdrClampPolicy { PerChannel30, MaxChannel10 };

// Post-spp reduce + surface writes for the split path. Folds the optional
// volume composite, averages per-pixel radiance over spp, computes DLSS-RR
// specular albedo and the explicit mirror-ray spec hit distance, then emits
// every NRD/DLSS-RR surface.
//
// `traceMirror(origin, dir) -> hitDistance` wraps the backend ray trace
// (returns 1e4f on miss; the caller's lambda enforces that).
//
// `applyDlssRRMinAlbedoGuard` and `hdrClampPolicy` carry the two real
// CUDA↔OptiX behavioral divergences that we preserve as-is — see the
// SplitHdrClampPolicy doc above for the HDR clamp story; the min-albedo
// guard is RTXPT PostProcess.hlsl §349-351 (OptiX-only today).
template <typename TraceMirrorFn>
__device__ inline void splitFinalizeAndWrite(
    const SplitAccumState& acc, uint32_t samplesPerPixel,
    TraceMirrorFn traceMirror,
    const SplitSurfaceOutputs& s, uint32_t x, uint32_t y,
    bool applyDlssRRMinAlbedoGuard,
    SplitHdrClampPolicy hdrClampPolicy)
{
    float invSpp = 1.0f / (float)samplesPerPixel;
    float3 demodDiffAvg  = acc.demodDiffSum  * invSpp;
    float3 demodSpecAvg  = acc.demodSpecSum  * invSpp;
    float3 emissiveAvg   = acc.emissiveSum   * invSpp;
    float3 noisyColorAvg = acc.noisyColorSum * invSpp;
    // Per-bucket hit-distance averaging: average only over samples that
    // actually filled the bucket, so pixels where one sample went diffuse
    // and the others specular don't bias the diff-bucket hitT toward zero.
    float diffHitAvg = acc.diffHitCount > 0 ? (acc.diffHitSum / (float)acc.diffHitCount) : 0.0f;
    float specHitAvg = acc.specHitCount > 0 ? (acc.specHitSum / (float)acc.specHitCount) : 0.0f;

    // DLSS-RR specular albedo: F0 = lerp(0.04, primaryAlbedo, metallic) per
    // the integration guide §3.4.2 + Appendix EnvBRDFApprox2. Metallic
    // preserved through outPrimaryMetallic so dielectric vs metal surfaces
    // get the right F0. NoV uses the actual primary ray direction (not the
    // unjittered camera.forward) so the spec-albedo guide buffer moves
    // smoothly across frames. Sky / no-hit pixels get a neutral default.
    float3 specAlbedoAvg;
    if (acc.gbufferWritten) {
        float3 specF0 = lerp(make_float3(0.04f, 0.04f, 0.04f),
                             acc.primary.albedo, acc.primary.metallic);
        float NoV = fmaxf(-dot(acc.primary.rayDir, acc.primary.normal), 0.0f);
        specAlbedoAvg = envBRDFApprox2(
            specF0, acc.primary.roughness * acc.primary.roughness, NoV);
    } else {
        specAlbedoAvg = make_float3(0.5f, 0.5f, 0.5f);  // §3.4.2 sky default
    }

    // DLSS-RR specular hit distance (§3.4.9): "World Space distance between
    // the Specular Ray Origin and Hit Point. Specular Ray Origin must be on
    // the Primary Surface." Prior implementations averaged secondary-bounce
    // distances across BOTH lobes — diffuse-bucket samples land on a cosine-
    // sampled bounce, NOT where the spec reflection would land. That made
    // the value flicker frame-to-frame depending on which lobe the bucket
    // roll picks, producing surface-wide motion shimmer. Trace ONE explicit
    // mirror ray per pixel from the primary hit along the perfect-reflection
    // direction: deterministic, sub-pixel-stable, and matches the canonical-
    // reflection semantics RR expects for deriving specular MV.
    float rrSpecHitT = 0.0f;
    if (acc.gbufferWritten) {
        float3 rd = acc.primary.rayDir;
        float3 N  = acc.primary.normal;
        float3 mirrorDir = normalize(rd - N * (2.0f * dot(rd, N)));
        float3 mOrigin   = acc.primary.hitPos + N * 0.001f;
        rrSpecHitT = traceMirror(mOrigin, mirrorDir);
    }
    if (isnan(rrSpecHitT) || isinf(rrSpecHitT) || rrSpecHitT < 0.0f) rrSpecHitT = 0.0f;

    // ── Surface writes ──────────────────────────────────────────────
    float4 diffTexel = nrd_helpers::packRadianceHitDist(demodDiffAvg, diffHitAvg);
    float4 specTexel = nrd_helpers::packRadianceHitDist(demodSpecAvg, specHitAvg);
    float4 normTexel = nrd_helpers::packNormalRoughness(acc.primary.normal, acc.primary.roughness);
    float3 albClamped = make_float3(
        fminf(fmaxf(acc.primary.albedo.x, 0.0f), 1.0f),
        fminf(fmaxf(acc.primary.albedo.y, 0.0f), 1.0f),
        fminf(fmaxf(acc.primary.albedo.z, 0.0f), 1.0f));
    float4 emTexel = make_float4(emissiveAvg.x, emissiveAvg.y, emissiveAvg.z, 1.0f);

    writeRGBA16F(s.diffuseRadianceHitDist,  x, y, diffTexel);
    writeRGBA16F(s.specularRadianceHitDist, x, y, specTexel);
    if (s.normalRoughness) {
        // packRGBA8_uint produces the same byte layout as a uchar4 surf2Dwrite
        // but works around an OptiX-PTX uchar4-store alignment bug on Ampere+.
        // CUDA-compiled code can also use this path safely (same output bytes).
        uint32_t packed = packRGBA8_uint(normTexel.x, normTexel.y, normTexel.z, normTexel.w);
        surf2Dwrite<uint32_t>(packed, s.normalRoughness, x * 4, y);
    }
    writeR32F(s.viewZ,    x, y, acc.primary.viewZ);
    writeR32F(s.ndcDepth, x, y, acc.primary.ndcZ);
    writeRG16F(s.motionVectors, x, y, acc.primary.mvPx);

    if (s.albedo) {
        // DLSS-RR min-albedo guard (RTXPT PostProcess.hlsl §349-351): when
        // both diffAlbedo AND specAlbedo are near-zero, the RR network has
        // no reflectance signal to demodulate against and produces splotchy
        // output that flickers during motion. Bumping diffAlbedo by 0.05 if
        // their sum is below 0.05 keeps RR's reflectance estimator stable on
        // near-black surfaces (deep wood, dark fabric). Only relevant when
        // this aux image feeds DLSS-RR (flag set); NRD-only mode still gets
        // the un-bumped albedo for the composite shader's diff*alb modulation.
        float3 dA = albClamped;
        if (applyDlssRRMinAlbedoGuard && s.specAlbedo) {
            float avg = (dA.x + dA.y + dA.z + specAlbedoAvg.x +
                         specAlbedoAvg.y + specAlbedoAvg.z) * (1.0f / 3.0f);
            if (avg < 0.05f) {
                dA.x += 0.05f; dA.y += 0.05f; dA.z += 0.05f;
                dA.x = fminf(dA.x, 1.0f); dA.y = fminf(dA.y, 1.0f); dA.z = fminf(dA.z, 1.0f);
            }
        }
        // Alpha = surface-valid mask. 0 on primary-miss (sky) pixels so the
        // composite shader can suppress NRD's stale OUT_SPEC values there;
        // see composite_tonemap.frag.
        float aMask = acc.gbufferWritten ? 1.0f : 0.0f;
        uint32_t packed = packRGBA8_uint(dA.x, dA.y, dA.z, aMask);
        surf2Dwrite<uint32_t>(packed, s.albedo, x * 4, y);
    }
    writeRGBA16F(s.emissive, x, y, emTexel);

    // ── DLSS-RR specific surfaces (zero-handle in NRD-only mode) ──
    if (s.hdrColor) {
        float3 c = noisyColorAvg;
        if (isnan(c.x) || isnan(c.y) || isnan(c.z) ||
            isinf(c.x) || isinf(c.y) || isinf(c.z)) c = make_float3(0,0,0);
        if (hdrClampPolicy == SplitHdrClampPolicy::MaxChannel10) {
            const float kRRBrightnessClamp = 10.0f;
            float maxC = fmaxf(c.x, fmaxf(c.y, c.z));
            if (maxC > kRRBrightnessClamp) c = c * (kRRBrightnessClamp / maxC);
            c.x = fmaxf(c.x, 0.0f);
            c.y = fmaxf(c.y, 0.0f);
            c.z = fmaxf(c.z, 0.0f);
        } else {
            c.x = fminf(fmaxf(c.x, 0.0f), 30.0f);
            c.y = fminf(fmaxf(c.y, 0.0f), 30.0f);
            c.z = fminf(fmaxf(c.z, 0.0f), 30.0f);
        }
        writeRGBA16F(s.hdrColor, x, y, make_float4(c.x, c.y, c.z, 1.0f));
    }
    // RGBA16F: world-space shading normal in xyz (fp16), linear roughness in w.
    // DLSS-RR §3.4.3 — RGB16/32 float, packed roughness via Roughness_Mode_Packed.
    writeRGBA16F(s.worldNormalRoughness, x, y,
                 make_float4(acc.primary.normal.x, acc.primary.normal.y,
                             acc.primary.normal.z, acc.primary.roughness));
    writeRGBA16F(s.specAlbedo, x, y, make_float4(
        clampf(specAlbedoAvg.x, 0.0f, 4.0f),
        clampf(specAlbedoAvg.y, 0.0f, 4.0f),
        clampf(specAlbedoAvg.z, 0.0f, 4.0f),
        1.0f));
    // §3.4.9: world-space distance, primary surface to spec-reflected hit.
    writeR32F(s.specHitT, x, y, rrSpecHitT);
}
