#pragma once
// Canonical __device__ helpers used by every CUDA / OptiX rendering kernel
// in src/render and src/backend: BT.709 luminance, BSDF (metallic-roughness +
// specular-glossiness), MIS power heuristic, environment sampling, camera-ray
// generation, and area-light helpers. Defined here once, header-only, so the
// path tracer (PathTraceKernel.cu / PathTraceKernelSplit.cu), ReSTIR DI / GI /
// PT (ReSTIR.cu / ReSTIRGI.cu / ReSTIRPT.cu), the volume integrator
// (VolumeRender.cu / VolumeNEE.cuh), and the OptiX raygens (OptiXPrograms.cu)
// all share a single source of truth.
//
// All functions are `__device__ inline` so multiple TUs can include this
// header without ODR conflicts under nvrtc / optixir.

#include "core/Math.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/RayTypes.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Sampling.h"
#include "gpu/SHEnv.cuh"

#ifndef M_PI_F
#  define M_PI_F 3.14159265358979323846f
#endif

// ── Color ────────────────────────────────────────────────────
// BT.709 relative luminance. Shared by the path tracer, ReSTIR, and the
// volume integrator — kept here so every call site uses the same coefficients.
__device__ inline float luminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// ── Environment ──────────────────────────────────────────────
__device__ inline float3 sampleEnvironment(float3 dir, cudaTextureObject_t envMap) {
    if (envMap != 0) {
        float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f));
        float phi   = atan2f(dir.z, dir.x);
        float u = (phi + M_PI_F) * (0.5f / M_PI_F);
        float v = theta / M_PI_F;
        float4 texel = tex2D<float4>(envMap, u, v);
        return make_float3(texel.x, texel.y, texel.z);
    }
    float t = 0.5f * (dir.y + 1.0f);
    float3 skyTop = make_float3(0.5f, 0.7f, 1.0f);
    float3 skyBot = make_float3(1.0f, 1.0f, 1.0f);
    return lerp(skyBot, skyTop, t) * 0.8f;
}

// Environment sampling with optional SH shortcut for indirect rays. When
// `useSH` is true and coefficients are available, the indirect miss returns
// the SH-reconstructed radiance (smooth, noise-free) — exactly the case where
// low-order SH is a perfect approximation of the env (only the diffuse band
// is physically meaningful). Primary-ray miss still uses the full HDR texture
// so the directly-visible sky remains sharp.
__device__ inline float3 sampleEnvironmentForBounce(
    float3 dir,
    cudaTextureObject_t envMap,
    const float3* shCoeffs,
    bool useSH,
    bool isPrimary)
{
    if (useSH && shCoeffs && !isPrimary) {
        return sh_evalRadiance(dir, shCoeffs);
    }
    return sampleEnvironment(dir, envMap);
}

// ── Ray generation ───────────────────────────────────────────
__device__ inline Ray generateRay(
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

// ── Cook-Torrance BRDF (inline) ─────────────────────────────
__device__ inline float ggxD_local(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * denom * denom + 1e-14f);
}

__device__ inline float3 fresnelSchlick_local(float cosTheta, float3 F0) {
    float t = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    return F0 + (make_float3(1,1,1) - F0) * t5;
}

__device__ inline float smithG1_GGX(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    float cos2 = NdotX * NdotX;
    return 2.0f * NdotX / (NdotX + sqrtf(a2 + (1.0f - a2) * cos2) + 1e-7f);
}

__device__ inline float powerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-7f);
}

__device__ inline float bsdfDiffusePdf(float NdotL) {
    return fmaxf(NdotL, 0.0f) * (1.0f / M_PI_F);
}

__device__ inline float bsdfSpecularPdf(
    const float3& N,
    const float3& V,
    const float3& L,
    float roughness)
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

__device__ inline float computeSpecProb(
    const float3& N,
    const float3& V,
    const float3& albedo,
    float metallic)
{
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float t = 1.0f - fminf(fmaxf(NdotV, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    float3 F = F0 + (make_float3(1,1,1) - F0) * t5;
    float specW = 0.2126f * F.x + 0.7152f * F.y + 0.0722f * F.z;
    float3 kd = (make_float3(1,1,1) - F) * (1.0f - metallic);
    float diffW = 0.2126f * (kd.x * albedo.x) + 0.7152f * (kd.y * albedo.y) + 0.0722f * (kd.z * albedo.z);
    float p = specW / fmaxf(specW + diffW, 1e-7f);
    return fminf(fmaxf(p, 0.1f), 0.9f);
}

__device__ inline float bsdfMixturePdf(
    const float3& N, const float3& V, const float3& L,
    float roughness, float specProb)
{
    float diffusePdf = bsdfDiffusePdf(dot(N, L));
    float specPdf = bsdfSpecularPdf(N, V, L, roughness);
    return specProb * specPdf + (1.0f - specProb) * diffusePdf;
}

__device__ inline float3 bsdfEvaluate(
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
    float3 F = fresnelSchlick_local(LdotH, F0);
    float D_val = ggxD_local(NdotH, roughness);
    float alpha = roughness * roughness;
    float G_val = smithG1_GGX(NdotL, alpha) * smithG1_GGX(NdotV, alpha);

    float3 specular = F * (D_val * G_val / (4.0f * NdotL * NdotV + 1e-7f));
    float3 kd = (make_float3(1, 1, 1) - F) * (1.0f - metallic);
    float3 diffuse = kd * albedo * (1.0f / M_PI_F);
    return diffuse + specular;
}

// ── Specular-Glossiness BRDF ─────────────────────────────────
// Same Cook-Torrance form as `bsdfEvaluate` but takes F0 directly (no
// metallic→F0 substitution). The diffuse term still uses (1-F) for energy
// conservation but is not darkened by metallic, matching glTF KHR_materials_pbrSpecularGlossiness.

__device__ inline float computeSpecProbSG(
    const float3& N, const float3& V,
    const float3& albedo, const float3& F0)
{
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float t = 1.0f - fminf(fmaxf(NdotV, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    float3 F = F0 + (make_float3(1,1,1) - F0) * t5;
    float specW = 0.2126f * F.x + 0.7152f * F.y + 0.0722f * F.z;
    float3 kd = (make_float3(1,1,1) - F);
    float diffW = 0.2126f * (kd.x * albedo.x) + 0.7152f * (kd.y * albedo.y) + 0.0722f * (kd.z * albedo.z);
    float p = specW / fmaxf(specW + diffW, 1e-7f);
    return fminf(fmaxf(p, 0.1f), 0.9f);
}

__device__ inline float3 bsdfEvaluateSG(
    const float3& N, const float3& V, const float3& L,
    const float3& albedo, float roughness, const float3& F0)
{
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0,0,0);

    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float LdotH = fmaxf(dot(L, H), 0.0f);

    float3 F = fresnelSchlick_local(LdotH, F0);
    float D_val = ggxD_local(NdotH, roughness);
    float alpha = roughness * roughness;
    float G_val = smithG1_GGX(NdotL, alpha) * smithG1_GGX(NdotV, alpha);

    float3 specular = F * (D_val * G_val / (4.0f * NdotL * NdotV + 1e-7f));
    float3 kd = (make_float3(1, 1, 1) - F);
    float3 diffuse = kd * albedo * (1.0f / M_PI_F);
    return diffuse + specular;
}

// ── Material-aware wrappers ─────────────────────────────────
// These respect GPUMaterial::pureDiffuse: when set, they bypass the
// Cook-Torrance specular lobe entirely and behave as a pure Lambertian BRDF
// (albedo/π, cosine-weighted sampling). Used for legacy Collada Phong
// materials that only carry a <diffuse> term.
// They also dispatch to the Specular-Glossiness path when
// GPUMaterial::useSpecularGlossiness is set, taking F0 from
// GPUMaterial::specularColor directly.

__device__ inline float materialSpecProb(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& albedo)
{
    if (mat.pureDiffuse) return 0.0f;
    // SG materials are remapped per-pixel into MR before BRDF evaluation.
    return computeSpecProb(N, V, albedo, mat.metallic);
}

__device__ inline float materialMixturePdf(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& L,
    float specProb)
{
    if (mat.pureDiffuse) return bsdfDiffusePdf(dot(N, L));
    return bsdfMixturePdf(N, V, L, mat.roughness, specProb);
}

__device__ inline float3 materialBsdfEvaluate(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& L,
    const float3& albedo)
{
    if (mat.pureDiffuse) {
        float NdotL = fmaxf(dot(N, L), 0.0f);
        float NdotV = fmaxf(dot(N, V), 0.0f);
        if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0, 0, 0);
        return albedo * (1.0f / M_PI_F);
    }
    // SG materials are remapped to MR per-pixel before reaching here.
    return bsdfEvaluate(N, V, L, albedo, mat.roughness, mat.metallic);
}

// ── Lobe-only BRDF evaluators ────────────────────────────────
// Diffuse / specular halves of `bsdfEvaluate`. Used by NRD-/DLSS-RR-targeted
// split kernels at the primary hit, where NEE and BSDF sampling are forced to
// the picked bucket so that diff_bucket * albedo + spec_bucket recovers the
// full primary-hit radiance after demodulation.

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

// SG variants: F0 comes from the material's specularColor instead of being
// derived from albedo+metallic. Diffuse lobe drops the (1-metallic) darkening.
__device__ inline float3 bsdfDiffuseLobeSG(
    const float3& N, const float3& V, const float3& L,
    const float3& albedo, const float3& F0)
{
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0,0,0);
    float3 H = normalize(V + L);
    float LdotH = fmaxf(dot(L, H), 0.0f);
    float3 F  = fresnelSchlick_local(LdotH, F0);
    float3 kd = (make_float3(1,1,1) - F);
    return kd * albedo * (1.0f / M_PI_F);
}

__device__ inline float3 bsdfSpecularLobeSG(
    const float3& N, const float3& V, const float3& L,
    float roughness, const float3& F0)
{
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0,0,0);
    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float LdotH = fmaxf(dot(L, H), 0.0f);
    float3 F  = fresnelSchlick_local(LdotH, F0);
    float D_val = ggxD_local(NdotH, roughness);
    float alpha = roughness * roughness;
    float G_val = smithG1_GGX(NdotL, alpha) * smithG1_GGX(NdotV, alpha);
    return F * (D_val * G_val / (4.0f * NdotL * NdotV + 1e-7f));
}

// Material-aware lobe wrappers — pureDiffuse materials have no specular lobe,
// and the diffuse lobe is pure albedo/π (no F0 dielectric scaling). SG
// materials are remapped to MR per-pixel before reaching here.
__device__ inline float3 materialDiffuseLobe(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& L, const float3& albedo)
{
    if (mat.pureDiffuse) {
        float NdotL = fmaxf(dot(N, L), 0.0f);
        float NdotV = fmaxf(dot(N, V), 0.0f);
        if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0, 0, 0);
        return albedo * (1.0f / M_PI_F);
    }
    return bsdfDiffuseLobe(N, V, L, albedo, mat.roughness, mat.metallic);
}

__device__ inline float3 materialSpecularLobe(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& L, const float3& albedo)
{
    if (mat.pureDiffuse) return make_float3(0, 0, 0);
    return bsdfSpecularLobe(N, V, L, albedo, mat.roughness, mat.metallic);
}

// ── Firefly clamp ────────────────────────────────────────────
// Per-contribution luminance-bounded clamp. NRD's RELAX denoiser is sensitive
// to single-sample spikes — one 100x outlier survives temporal filtering for
// many frames as a shimmering bright speck (water-ripple look). Clamp each
// NEE / emissive contribution by luminance before adding it to the running
// path radiance, rather than only clamping the sum once at the end.
__device__ inline float3 clampFirefly(float3 c, float maxLum) {
    float lum = 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
    if (lum > maxLum && lum > 1e-7f) c = c * (maxLum / lum);
    return c;
}

// ── DLSS-RR specular albedo guide ────────────────────────────
// DLSS-RR §3.4.2 / Appendix: per-pixel specular albedo from F0, alpha, NoV.
// F0 is derived from the material's specular reflectance
// (lerp(0.04, albedo, metallic)). Used as the demodulation factor for the
// specular guide. Sky pixels get a neutral default — see guide §3.4.2.
__device__ inline float3 envBRDFApprox2(float3 F0, float alpha, float NoV) {
    NoV = fabsf(NoV);
    float NoV2 = NoV * NoV;
    float NoV3 = NoV2 * NoV;
    float a3   = alpha * alpha * alpha;
    // M1 = [[0.99044, -1.28514], [1.29678, -0.755907]]
    float M1xy_top = 0.99044f - 1.28514f * NoV;
    float M1xy_bot = 1.29678f - 0.755907f * NoV;
    float biasNum = M1xy_top + M1xy_bot * alpha;
    // M2 = [[1, 2.92338, 59.4188], [20.3225, -27.0302, 222.592], [121.563, 626.13, 316.627]]
    float M2_0 = 1.0f + 2.92338f * NoV + 59.4188f * NoV3;
    float M2_1 = 20.3225f - 27.0302f * NoV + 222.592f * NoV3;
    float M2_2 = 121.563f + 626.13f * NoV + 316.627f * NoV3;
    float biasDen = M2_0 + M2_1 * alpha + M2_2 * a3;
    float bias = biasNum / fmaxf(biasDen, 1e-7f);
    // M3 = [[0.0365463, 3.32707], [9.0632, -9.04756]]
    float M3xy_top = 0.0365463f + 3.32707f * NoV;
    float M3xy_bot = 9.0632f    - 9.04756f * NoV;
    float scaleNum = M3xy_top + M3xy_bot * alpha;
    // M4 = [[1, 3.59685, -1.36772], [9.04401, -16.3174, 9.22949], [5.56589, 19.7886, -20.2123]]
    float M4_0 = 1.0f + 3.59685f * NoV2 - 1.36772f * NoV3;
    float M4_1 = 9.04401f - 16.3174f * NoV2 + 9.22949f * NoV3;
    float M4_2 = 5.56589f + 19.7886f * NoV2 - 20.2123f * NoV3;
    float scaleDen = M4_0 + M4_1 * alpha + M4_2 * a3;
    float scale = scaleNum / fmaxf(scaleDen, 1e-7f);
    bias *= fminf(fmaxf(F0.y * 50.0f, 0.0f), 1.0f);
    scale = fmaxf(scale, 0.0f);
    bias  = fmaxf(bias, 0.0f);
    return make_float3(F0.x * scale + bias,
                       F0.y * scale + bias,
                       F0.z * scale + bias);
}

// Fetch Le at a barycentric point on an area light (texture-aware).
__device__ inline float3 sampleAreaLightLe(
    const GPUAreaLight& light, float b0, float b1, float b2)
{
    if (light.emissiveTex == 0) return light.emission;
    float u = light.uv0.x * b0 + light.uv1.x * b1 + light.uv2.x * b2;
    float v = light.uv0.y * b0 + light.uv1.y * b1 + light.uv2.y * b2;
    float4 texel = tex2D<float4>(light.emissiveTex, u, v);
    return make_float3(texel.x, texel.y, texel.z) * light.emission;
}

__device__ inline uint32_t sampleAreaLightIndex(
    const float* cdf, uint32_t count, float target)
{
    uint32_t low = 0;
    uint32_t high = count;
    while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (target <= cdf[mid]) high = mid;
        else                    low = mid + 1;
    }
    return (low >= count) ? (count - 1) : low;
}
