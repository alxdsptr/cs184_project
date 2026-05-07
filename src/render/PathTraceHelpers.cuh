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
#include "gpu/BRDF.h"
#include "gpu/RayTypes.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
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

// ── Material texture sampling + Specular-Glossiness remap ────
// Applies albedo / metallic-roughness / SG textures to a GPUMaterial and reads
// emissiveTex into `emissive`. Mutates `mat.metallic`, `mat.roughness`, and
// `albedo` so the rest of the kernel sees a uniform metallic-roughness
// representation regardless of how the asset was authored.
//
// Specular-Glossiness "soft" interpretation: the spec map's chromaticity is
// unreliable across assets (saturated magentas / yellows that aren't physical
// F0), but its *luminance* is a meaningful "how reflective is this pixel"
// signal. We use spec luminance to drive both roughness and F0 strength so
// that bright-spec areas (e.g. MEASURE_SEVEN's polished floor) get visible
// mirror-like highlights, while dark-spec areas (matte walls, fabric) stay
// properly diffuse. Albedo keeps its BaseColor chromaticity — only its
// weighting in the BRDF changes.
//
// Encoding the target F0 through (metallic, albedo) so we don't fork the BRDF:
// setting metallic=m and albedo=a gives F0 = lerp(0.04, a, m) = 0.04*(1-m)+a*m.
// We pick m so F0(white) = 0.04 + (Ftarget - 0.04) = Ftarget when a is treated
// as white at the F0-mixing site. To keep diffuse colour intact we apply this
// ONLY to F0; the diffuse term still uses the original BaseColor through
// kd = (1-F)*(1-m), which gracefully fades diffuse as the surface becomes
// more metallic — the right behaviour for polished stone / brushed metal.
//
// Three SG packing branches are recognized:
//   1. C4D custom (mat.useFBXCustomPacking): G=roughness, B=per-material spec
//      strength scaling specularColor up from the dielectric baseline. R/A
//      unused. Encodes F0 = lerp(0.04, specularColor, B) by setting
//      albedo := specColor, metallic := B — diffuse routes through
//      (1-F)*(1-metallic)*albedo and fades on highly reflective pixels.
//   2. Unreal / standard PBR-Specular (mat.useFBXUEPacking): G=glossiness
//      (high=smooth), B=metallic mask. Albedo keeps its BaseColor — gives
//      metals their characteristic tinted F0 = lerp(0.04, baseColor, 1).
//   3. Generic glTF KHR_materials_pbrSpecularGlossiness: spec luminance drives
//      a "soft" F0 target (0.04 + 0.56 * sqrt(specLum)) routed through the
//      metallic+albedo path. The sqrt curve lets even mid-luminance spec give
//      noticeable polish (matching how artists paint masks), and the 0.56 cap
//      keeps F0 below pure mirror so BaseColor stays visible in the diffuse
//      lobe. When alpha carries glossiness data it modulates the material
//      factor; otherwise spec luminance directly drives glossiness so
//      bright-spec areas show clear reflections.
__device__ inline void applyMaterialTextures(
    GPUMaterial& mat, float2 texUV,
    float3& albedo, float3& emissive)
{
    albedo = mat.albedo;
    if (mat.albedoTex != 0) {
        float4 tc = tex2D<float4>(mat.albedoTex, texUV.x, texUV.y);
        albedo = make_float3(tc.x, tc.y, tc.z);
    }
    // glTF MR convention: G = roughness, B = metallic.
    if (mat.metallicRoughTex != 0) {
        float4 mr = tex2D<float4>(mat.metallicRoughTex, texUV.x, texUV.y);
        mat.roughness = mat.roughness * mr.y;
        mat.metallic  = mat.metallic  * mr.z;
    }
    if (mat.useSpecularGlossiness) {
        if (mat.useFBXCustomPacking && mat.specularGlossTex != 0) {
            // C4D: G=roughness, B=spec strength.
            float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
            float B = clampf(sg.z, 0.0f, 1.0f);
            float G = clampf(sg.y, 0.0f, 1.0f);
            albedo = mat.specularColor;
            mat.metallic  = B;
            mat.roughness = G;
        } else if (mat.useFBXUEPacking && mat.specularGlossTex != 0) {
            // UE / standard PBR-Spec: G=glossiness (high=smooth), B=metallic.
            float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
            float G = clampf(sg.y, 0.0f, 1.0f);
            float B = clampf(sg.z, 0.0f, 1.0f);
            mat.metallic  = B;
            mat.roughness = 1.0f - G;
        } else {
            // Generic glTF KHR_materials_pbrSpecularGlossiness — soft F0 mapping.
            float3 specRGB = mat.specularColor;
            float  alphaG  = 1.0f;
            if (mat.specularGlossTex != 0) {
                float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
                specRGB = mat.specularColor * make_float3(sg.x, sg.y, sg.z);
                alphaG  = sg.w;
            }
            float specLum = 0.2126f * specRGB.x + 0.7152f * specRGB.y + 0.0722f * specRGB.z;
            specLum = clampf(specLum, 0.0f, 1.0f);
            // sqrt curve: even mid-luminance spec gives noticeable polish.
            float specStrength = sqrtf(specLum);
            // Target F0 ranges from dielectric baseline (0.04) to ~0.6 for the
            // brightest spec values — polished-metal territory but capped below
            // pure mirror to leave headroom for the BaseColor tint in diffuse.
            float F0_target = 0.04f + 0.56f * specStrength;
            mat.metallic = clampf((F0_target - 0.04f) / 0.96f, 0.0f, 1.0f);
            // Alpha-as-glossiness uses the material factor; otherwise spec
            // luminance directly drives glossiness so bright-spec areas show
            // clear reflections, and `glossiness` acts as an upper-bound trim.
            float gloss = mat.specularGlossAlphaIsGlossiness
                            ? (mat.glossiness * alphaG)
                            :  specStrength;
            mat.roughness = 1.0f - clampf(gloss, 0.0f, 0.95f);
        }
    }
    // Clamp to stable ranges for BRDF sampling/evaluation.
    mat.roughness = fmaxf(mat.roughness, 0.045f);
    mat.metallic  = clampf(mat.metallic, 0.0f, 1.0f);

    emissive = mat.emission;
    if (mat.emissiveTex != 0) {
        float4 et = tex2D<float4>(mat.emissiveTex, texUV.x, texUV.y);
        emissive = make_float3(et.x, et.y, et.z);
    }
}

// Clamp an environment-map sample by luminance. NRD/DLSS feed bright HDR sky
// pixels (e.g. the sun) directly into the temporal accumulator, where a single
// firefly-strength sample survives reprojection for many frames as a shimmering
// speck. The mono kernels use a generous cap (100); the split / NRD kernels
// use a tighter cap (20) since RELAX is more sensitive to outliers.
__device__ inline float3 clampEnvLuminance(float3 envColor, float maxLum) {
    float lum = 0.2126f * envColor.x + 0.7152f * envColor.y + 0.0722f * envColor.z;
    if (lum > maxLum) envColor = envColor * (maxLum / lum);
    return envColor;
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

// ── Glass / dielectric delta bounce ──────────────────────────
// Picks reflection or refraction stochastically against exact dielectric
// Fresnel; falls back to a mirror reflection on TIR. Applies a glass tint
// (multiply throughput by albedo) only when *exiting* the medium and only if
// the albedo is intentionally non-white — keeps clear glass clear.
//
// Caller is responsible for: ray.origin/direction/tmin/tmax updates, setting
// `lastBounceDelta = true`, prevSurface bookkeeping, post-bounce RR, and any
// aux-buffer N-flip after the bounce.
struct GlassBounce {
    float3 newDir;
    float3 newOrigin;     // hit position offset 0.002 along the outgoing side
    float3 throughputMul; // (1,1,1) on entry / near-white albedo, else albedo tint
};

__device__ inline GlassBounce sampleGlassBounce(
    const float3& rayDir, const float3& hitPos, const float3& Nshade,
    bool entering, float ior, const float3& albedo,
    uint32_t& rng)
{
    float3 Nglass = entering ? Nshade : -Nshade;
    if (dot(Nglass, rayDir) > 0.0f) Nglass = -Nglass;

    float eta = (entering ? 1.0f : ior) / (entering ? ior : 1.0f);
    float cosI = fmaxf(dot(-rayDir, Nglass), 0.0f);
    float Fr = fresnelDielectric(cosI, eta);

    float3 newDir;
    if (pcg32_float(rng) < Fr) {
        newDir = normalize(rayDir - Nglass * (2.0f * dot(rayDir, Nglass)));
    } else if (!refractDir(rayDir, Nglass, eta, newDir)) {
        newDir = normalize(rayDir - Nglass * (2.0f * dot(rayDir, Nglass)));
    }

    GlassBounce r;
    r.newDir = newDir;
    float3 off = (dot(newDir, Nglass) > 0.0f) ? Nglass : -Nglass;
    r.newOrigin = hitPos + off * 0.002f;
    r.throughputMul = make_float3(1.0f, 1.0f, 1.0f);
    if (!entering) {
        float lum = 0.2126f * albedo.x + 0.7152f * albedo.y + 0.0722f * albedo.z;
        if (lum < 0.9f) r.throughputMul = albedo;
    }
    return r;
}

// ── Tangent interpolation + normal-map application ──────────
// Interpolates the per-vertex tangents at a hit's barycentric coordinates and
// applies the material's tangent-space normal map. Optionally writes the
// interpolated tangent through `outTangent` so the mono CUDA debug-viz can
// inspect handedness.
//
// Caller is responsible for gating the call (e.g. `mat.transmission <= 0 &&
// mat.normalTex != 0 && scene.d_tangents`).
__device__ inline float3 applyInterpolatedNormalMap(
    float3 N, const DeviceSceneData& scene,
    uint32_t i0, uint32_t i1, uint32_t i2,
    float baryU, float baryV, float baryW,
    cudaTextureObject_t normalTex, float2 texUV,
    float4* outTangent = nullptr)
{
    float4 t0 = scene.d_tangents[i0];
    float4 t1 = scene.d_tangents[i1];
    float4 t2 = scene.d_tangents[i2];
    float4 tangent = t0 * baryW + t1 * baryU + t2 * baryV;
    if (outTangent) *outTangent = tangent;
    return applyNormalMap(N, tangent, normalTex, texUV);
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
