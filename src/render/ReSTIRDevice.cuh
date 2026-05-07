#pragma once
// ReSTIR DI device-side helpers shared between the CUDA ReSTIR kernel
// (render/ReSTIR.cu) and the OptiX raygen (backend/OptiXPrograms.cu).
//
// Contains the target-pdf evaluator and the reservoir streaming primitives
// (Bitterli Alg. 2). General BRDF / luminance / camera-ray helpers live in
// render/PathTraceHelpers.cuh and are pulled in here so this header doesn't
// re-define them.
//
// All functions are header-only `__device__ inline` so including this from
// multiple translation units does not cause multiple-definition link errors
// under nvrtc / optixir.

#include "render/ReSTIR.h"
#include "render/PathTraceHelpers.cuh"
#include "gpu/Random.h"
#include "gpu/Sampling.h"

// Sample a BSDF direction at a ReSTIR surface — shared by ReSTIR DI / GI / PT
// initial-candidate generators (CUDA + OptiX). Mixture sampling between GGX
// importance and cosine-hemisphere; pdf is evaluated with the same canonical
// helpers the path tracer uses, so reservoirs from both backends remain
// binary-compatible.
__device__ inline bool restirSampleBsdfDir(
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
    outPdf = pureDiffuse
        ? bsdfDiffusePdf(dot(s.normal, dir))
        : bsdfMixturePdf(s.normal, s.viewDir, dir, s.roughness, specProb);
    return outPdf > 1e-7f;
}

__device__ inline float3 restirEvalBrdf(
    const ReSTIRSurface& s, const float3& L)
{
    float NdotL = fmaxf(dot(s.normal, L), 0.0f);
    float NdotV = fmaxf(dot(s.normal, s.viewDir), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0,0,0);
    if (s.pureDiffuse) return s.albedo * (1.0f / M_PI_F);

    float3 H = normalize(s.viewDir + L);
    float NdotH = fmaxf(dot(s.normal, H), 0.0f);
    float LdotH = fmaxf(dot(L, H), 0.0f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), s.albedo, s.metallic);
    float3 F = fresnelSchlick_local(LdotH, F0);
    float Dt = ggxD_local(NdotH, s.roughness);
    float alpha = s.roughness * s.roughness;
    float Gt = smithG1_GGX(NdotL, alpha) * smithG1_GGX(NdotV, alpha);

    float3 spec = F * (Dt * Gt / (4.0f * NdotL * NdotV + 1e-7f));
    float3 kd = (make_float3(1,1,1) - F) * (1.0f - s.metallic);
    float3 diff = kd * s.albedo * (1.0f / M_PI_F);
    return diff + spec;
}

// Target pdf for RIS: luminance(Le) * |BRDF * NdotL| * geometry, NO visibility.
// Returns 0 if the sample is back-facing on either surface.
//
// IMPORTANT: when the area light has an emissive texture, the *actual* Le
// used by the path tracer's NEE comes from sampling that texture at the
// barycentric position. We must mirror that here, otherwise pHat reflects
// only the base `light.emission` (often a fraction of the textured value)
// while the final estimator evaluates `f * Le * W` against the textured Le —
// the resulting W is too large for textured emitters, producing the bright
// overexposure observed on scenes with screen / strip / billboard emitters.
__device__ inline float restirEvalTargetPdf(
    const ReSTIRSurface& s,
    const GPUAreaLight&  light,
    float b1, float b2)
{
    float b0 = 1.0f - b1 - b2;
    if (b0 < 0.0f || b1 < 0.0f || b2 < 0.0f) return 0.0f;
    float3 pOnLight = light.v0 + light.e1 * b1 + light.e2 * b2;
    float3 toL = pOnLight - s.position;
    float  dist2 = fmaxf(dot(toL, toL), 1e-6f);
    float  dist  = sqrtf(dist2);
    float3 L = toL * (1.0f / dist);

    float NdotL = fmaxf(dot(s.normal, L), 0.0f);
    float lightNdot = fmaxf(dot(light.normal, -L), 0.0f);
    if (NdotL <= 0.0f || lightNdot <= 0.0f) return 0.0f;

    float3 Le = light.emission;
    if (light.emissiveTex != 0) {
        float u = light.uv0.x * b0 + light.uv1.x * b1 + light.uv2.x * b2;
        float v = light.uv0.y * b0 + light.uv1.y * b1 + light.uv2.y * b2;
        float4 t = tex2D<float4>(light.emissiveTex, u, v);
        Le = make_float3(t.x, t.y, t.z) * light.emission;
    }
    float Lum = luminance(Le);
    if (Lum <= 0.0f) return 0.0f;

    float3 brdf = restirEvalBrdf(s, L);
    float  fLum = luminance(brdf) * NdotL;
    if (fLum <= 0.0f) return 0.0f;

    float geom = lightNdot / dist2;
    return Lum * fLum * geom;
}

// ── Reservoir primitives (Bitterli 2020 Alg. 2) ─────────────────
__device__ inline void restir_reservoirReset(ReSTIRReservoir& r) {
    r.lightIndex = 0xFFFFFFFFu;
    r.baryB1 = 0.0f;
    r.baryB2 = 0.0f;
    r.pHat   = 0.0f;
    r.W      = 0.0f;
    r.M      = 0.0f;
}

// ── Defensive resampling-weight clamps (M7 flash-and-decay fix) ──────────
// With many small emissive triangles, occasional samples land at tiny
// dist²/cos_light combinations. Their pHat (luminance×|f|×G) blows up by
// 6+ orders of magnitude. The unbiased estimator absorbs this in
// expectation, but RIS's selection probability w_i ∝ pHat·W·M makes the
// bad sample dominate every neighbor it spreads to via spatial reuse and
// every history pixel it passes through via temporal reuse — that's the
// "blob lights up then decays over mCap frames" pathology Lin et al.
// flag in §5.4 (and which Theorem A.4 protects against under proper MIS,
// but only with bounded resampling weights).
//
// We clamp the per-candidate resampling weight wCandidate at init time
// (RESTIR_DI_MAX_WCAND) and the reservoir's contribution weight W after
// finalize (RESTIR_DI_MAX_W). The clamp constants are well above any
// physical luminance the path tracer's NEE will produce in practice
// (visible-luminance × albedo·invPi ≈ O(10) for the brightest emitters
// in the bundled scenes; we cap at 1e4 so only fp32 outliers from BVH
// near-zero pSelect are clipped).
#ifndef RESTIR_DI_MAX_WCAND
#define RESTIR_DI_MAX_WCAND 1.0e4f
#endif
#ifndef RESTIR_DI_MAX_W
#define RESTIR_DI_MAX_W     1.0e4f
#endif
#ifndef RESTIR_DI_MAX_COMBINE_W
#define RESTIR_DI_MAX_COMBINE_W 1.0e6f
#endif

__device__ inline bool restir_reservoirUpdate(
    ReSTIRReservoir& r, float& wSum,
    uint32_t lightIdx, float b1, float b2, float pHat,
    float wCandidate, float u01)
{
    // Paper Algorithm 1 increments M unconditionally (counts every candidate
    // considered, including those with wCandidate=0). Skipping the increment
    // for zero-weight candidates inflates W = wSum/(M·pHat) by a factor of
    // (true M / non-zero M) — a small bias on its own, but it stacks under
    // temporal+spatial reuse since the inflated W gets blended forward.
    r.M += 1.0f;
    if (!(wCandidate > 0.0f)) return false;
    // Defensive clamp: a near-zero pSelect from the deep light-BVH
    // descent can make wCandidate = pHat/pSelect explode in fp32.
    if (wCandidate > RESTIR_DI_MAX_WCAND) wCandidate = RESTIR_DI_MAX_WCAND;
    wSum += wCandidate;
    if (u01 * wSum < wCandidate) {
        r.lightIndex = lightIdx;
        r.baryB1     = b1;
        r.baryB2     = b2;
        r.pHat       = pHat;
        return true;
    }
    return false;
}

__device__ inline void restir_reservoirFinalize(ReSTIRReservoir& r, float wSum) {
    if (r.lightIndex == 0xFFFFFFFFu || r.pHat <= 0.0f || r.M <= 0.0f) {
        r.W = 0.0f;
        return;
    }
    float W = wSum / (r.M * r.pHat);
    // Hard cap on W. Once a high-W reservoir leaves this kernel, the
    // temporal pass propagates it forward (with M cap=20, so the bad
    // sample lingers ~20 frames), and the spatial pass spreads it to
    // every neighbor that passes the geometric gate. Capping W here
    // bounds the worst-case visible artifact to roughly clampMax × Le.
    if (W > RESTIR_DI_MAX_W) W = RESTIR_DI_MAX_W;
    if (!isfinite(W)) W = 0.0f;
    r.W = W;
}

__device__ inline bool restir_reservoirCombine(
    ReSTIRReservoir& dst, float& wSum,
    const ReSTIRReservoir& src, float pHatAtDst, float u01)
{
    if (src.lightIndex == 0xFFFFFFFFu || src.M <= 0.0f) {
        dst.M += src.M;
        return false;
    }
    float w = pHatAtDst * src.W * src.M;
    // Belt-and-suspenders: even with W capped at finalize, src.W * src.M
    // can still combine to a large value, and pHatAtDst at a different
    // surface may amplify it further. Cap the resampling weight so a
    // single rogue source can't dominate the merged reservoir.
    if (!(w > 0.0f) || !isfinite(w)) {
        dst.M += src.M;
        return false;
    }
    if (w > RESTIR_DI_MAX_COMBINE_W) w = RESTIR_DI_MAX_COMBINE_W;
    bool accepted = false;
    wSum += w;
    if (u01 * wSum < w) {
        dst.lightIndex = src.lightIndex;
        dst.baryB1     = src.baryB1;
        dst.baryB2     = src.baryB2;
        dst.pHat       = pHatAtDst;
        accepted = true;
    }
    dst.M += src.M;
    return accepted;
}

// ─────────────────────────────────────────────────────────────────
// ReSTIR hit decode + ReSTIRSurface construction
// ─────────────────────────────────────────────────────────────────

// Result of `restirDecodeHit` — bundles vertex-interpolated shading attributes
// + textured material values for one BVH / OptiX hit point. ReSTIR's shading
// model is simpler than the path tracers' — no normal mapping (the ReSTIR
// surface is mainly used for reservoir-compatibility tests and BRDF eval, not
// for the most accurate shading frame).
struct ReSTIRHitDecode {
    bool        valid;       // false if material index is out of range
    float3      pos;
    float3      normal;      // interpolated geom-or-vertex, back-face-flipped
    float2      uv;
    float3      albedo;      // mat.albedo * albedoTex
    float3      emission;    // mat.emission * emissionStrength * emissiveTex
    GPUMaterial mat;         // mat.metallic / mat.roughness already adjusted from MR texture
    bool        pureDiffuse;
};

// Resolve a primary or secondary hit (`primIdx` + barycentrics) into shading
// attributes ReSTIR needs: position, back-face-flipped shading normal, UV,
// albedo (with optional albedoTex), MR-adjusted material, emissive radiance
// (with optional emissiveTex × emissionStrength). Vertex-normal / UV fetch
// gracefully fall back when those streams are null.
//
// Used by:
//   * ReSTIRPT.cu  ptShadeHit (CUDA)
//   * OptiXProgramsPT.inl  ptShadeHitOptiX (OptiX)
//   * ReSTIRGI.cu  primary + secondary hit decodes
//   * OptiXProgramsGI.inl  primary + secondary hit decodes inside the raygen
__device__ inline ReSTIRHitDecode restirDecodeHit(
    const DeviceSceneData& scene,
    uint32_t primIdx, float baryU, float baryV,
    float3 rayDir)
{
    ReSTIRHitDecode h{};

    int matIdx = scene.d_materialIndices ? scene.d_materialIndices[primIdx] : -1;
    if (matIdx < 0 || (uint32_t)matIdx >= scene.materialCount) return h;
    h.mat = scene.d_materials[matIdx];

    uint32_t i0 = scene.d_indices[primIdx * 3 + 0];
    uint32_t i1 = scene.d_indices[primIdx * 3 + 1];
    uint32_t i2 = scene.d_indices[primIdx * 3 + 2];
    float baryW = 1.0f - baryU - baryV;

    float3 v0 = scene.d_positions[i0];
    float3 v1 = scene.d_positions[i1];
    float3 v2 = scene.d_positions[i2];
    h.pos = v0 * baryW + v1 * baryU + v2 * baryV;

    if (scene.d_normals) {
        h.normal = normalize(scene.d_normals[i0] * baryW
                           + scene.d_normals[i1] * baryU
                           + scene.d_normals[i2] * baryV);
    } else {
        h.normal = normalize(cross(v1 - v0, v2 - v0));
    }
    if (dot(h.normal, rayDir) > 0.0f) h.normal = -h.normal;

    h.uv = scene.d_uvs
        ? (scene.d_uvs[i0] * baryW + scene.d_uvs[i1] * baryU + scene.d_uvs[i2] * baryV)
        : make_float2(0.0f, 0.0f);

    h.albedo = h.mat.albedo;
    if (h.mat.albedoTex != 0) {
        float4 t = tex2D<float4>(h.mat.albedoTex, h.uv.x, h.uv.y);
        h.albedo = h.albedo * make_float3(t.x, t.y, t.z);
    }
    if (h.mat.metallicRoughTex != 0) {
        float4 mrT = tex2D<float4>(h.mat.metallicRoughTex, h.uv.x, h.uv.y);
        h.mat.roughness *= mrT.y;
        h.mat.metallic  *= mrT.z;
    }

    h.emission = h.mat.emission * h.mat.emissionStrength;
    if (h.mat.emissiveTex != 0) {
        float4 et = tex2D<float4>(h.mat.emissiveTex, h.uv.x, h.uv.y);
        h.emission = make_float3(et.x, et.y, et.z) * h.mat.emissionStrength;
    }

    h.pureDiffuse = (h.mat.pureDiffuse != 0);
    h.valid = true;
    return h;
}

// Build a ReSTIRSurface from raw shading attributes + a precomputed specProb.
// Used by ReSTIR PT's path postfix (CUDA + OptiX) at every random-walk vertex
// — the caller passes the just-decoded hit attributes plus the BSDF spec
// probability for fast pHat eval inside the reservoir streamer.
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
