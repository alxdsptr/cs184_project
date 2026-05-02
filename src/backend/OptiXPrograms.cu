// ── OptiX device programs ─────────────────────────────────────
// Raygen: outer bounce loop + shading (ported from PathTraceKernel.cu).
// Closest-hit: records primitive index + barycentrics into payload.
// Miss (radiance): marks "no hit".
// Any-hit (shadow): accumulates glass transmittance, ignores or terminates.
// Miss (shadow): no-op (transmittance slots carry final value).

#include <optix.h>
#include <cuda_fp16.h>
#include <surface_indirect_functions.h>

#include "backend/OptiXLaunchParams.h"
#include "gpu/NRDHelpers.cuh"
#include "core/Math.h"
#include "core/Halton.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/LightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "gpu/Sampling.h"
#include "gpu/BRDF.h"
#include "gpu/RayTypes.h"
#include "gpu/SHEnv.cuh"
#include "accel/LightBVHSample.h"
#include "render/ReSTIRDevice.cuh"
#include "render/ReSTIRGIDevice.cuh"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

extern "C" __constant__ LaunchParams params;

// ── Payload packing helpers ──────────────────────────────────
static __forceinline__ __device__ uint32_t floatBitsToUint(float f) {
    return __float_as_uint(f);
}
static __forceinline__ __device__ float uintBitsToFloat(uint32_t u) {
    return __uint_as_float(u);
}

// Radiance payload (5 slots): hit, primIdx, baryU, baryV, tHit
struct RadiancePayload {
    uint32_t hit;
    uint32_t primIdx;
    float    baryU;
    float    baryV;
    float    tHit;
};

static __forceinline__ __device__ RadiancePayload traceRadianceRay(
    OptixTraversableHandle h,
    float3 origin, float3 dir, float tmin, float tmax)
{
    uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    optixTrace(
        h,
        origin, dir,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        /*SBT offset*/0,
        /*SBT stride*/2,
        /*miss SBT  */0,
        p0, p1, p2, p3, p4);
    RadiancePayload r;
    r.hit     = p0;
    r.primIdx = p1;
    r.baryU   = uintBitsToFloat(p2);
    r.baryV   = uintBitsToFloat(p3);
    r.tHit    = uintBitsToFloat(p4);
    return r;
}

// Shadow payload (4 slots): transmittance xyz + occluded flag
static __forceinline__ __device__ float3 traceShadowRay(
    OptixTraversableHandle h,
    float3 origin, float3 dir, float tmin, float tmax)
{
    uint32_t p0 = floatBitsToUint(1.0f);
    uint32_t p1 = floatBitsToUint(1.0f);
    uint32_t p2 = floatBitsToUint(1.0f);
    uint32_t p3 = 0;  // occluded flag
    optixTrace(
        h,
        origin, dir,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
        /*SBT offset*/1,
        /*SBT stride*/2,
        /*miss SBT  */1,
        p0, p1, p2, p3);
    if (p3 != 0) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(uintBitsToFloat(p0), uintBitsToFloat(p1), uintBitsToFloat(p2));
}

// ── Environment ──────────────────────────────────────────────
static __forceinline__ __device__ float3 sampleEnvironment(float3 dir, cudaTextureObject_t envMap) {
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

static __forceinline__ __device__ float3 sampleEnvironmentForBounce(
    float3 dir, cudaTextureObject_t envMap,
    const float3* shCoeffs, bool useSH, bool isPrimary)
{
    if (useSH && shCoeffs && !isPrimary) {
        return sh_evalRadiance(dir, shCoeffs);
    }
    return sampleEnvironment(dir, envMap);
}

// ── BSDF helpers (ported from PathTraceKernel.cu) ─────────────
static __forceinline__ __device__ float ggxD_local(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * denom * denom + 1e-14f);
}
static __forceinline__ __device__ float3 fresnelSchlick_local(float cosTheta, float3 F0) {
    float t = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    return F0 + (make_float3(1,1,1) - F0) * t5;
}
static __forceinline__ __device__ float smithG1_GGX(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    float cos2 = NdotX * NdotX;
    return 2.0f * NdotX / (NdotX + sqrtf(a2 + (1.0f - a2) * cos2) + 1e-7f);
}
static __forceinline__ __device__ float powerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    return a2 / fmaxf(a2 + b2, 1e-7f);
}
static __forceinline__ __device__ float bsdfDiffusePdf(float NdotL) {
    return fmaxf(NdotL, 0.0f) * (1.0f / M_PI_F);
}
static __forceinline__ __device__ float bsdfSpecularPdf(
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
static __forceinline__ __device__ float computeSpecProb(
    const float3& N, const float3& V, const float3& albedo, float metallic)
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
static __forceinline__ __device__ float bsdfMixturePdf(
    const float3& N, const float3& V, const float3& L, float roughness, float specProb)
{
    float diffusePdf = bsdfDiffusePdf(dot(N, L));
    float specPdf = bsdfSpecularPdf(N, V, L, roughness);
    return specProb * specPdf + (1.0f - specProb) * diffusePdf;
}
static __forceinline__ __device__ float3 bsdfEvaluate(
    const float3& N, const float3& V, const float3& L,
    const float3& albedo, float roughness, float metallic)
{
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
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

// Material-aware wrappers — pureDiffuse materials bypass the Cook-Torrance
// specular lobe entirely and behave as a pure Lambertian BRDF. Used for
// legacy Collada Phong materials that only carry a <diffuse> term.
static __forceinline__ __device__ float materialSpecProb(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& albedo)
{
    if (mat.pureDiffuse) return 0.0f;
    return computeSpecProb(N, V, albedo, mat.metallic);
}

static __forceinline__ __device__ float materialMixturePdf(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& L,
    float specProb)
{
    if (mat.pureDiffuse) return bsdfDiffusePdf(dot(N, L));
    return bsdfMixturePdf(N, V, L, mat.roughness, specProb);
}

static __forceinline__ __device__ float3 materialBsdfEvaluate(
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
    return bsdfEvaluate(N, V, L, albedo, mat.roughness, mat.metallic);
}

// Texture-aware Le fetch at a barycentric point on an area light.
static __forceinline__ __device__ float3 sampleAreaLightLe(
    const GPUAreaLight& light, float b0, float b1, float b2)
{
    if (light.emissiveTex == 0) return light.emission;
    float u = light.uv0.x * b0 + light.uv1.x * b1 + light.uv2.x * b2;
    float v = light.uv0.y * b0 + light.uv1.y * b1 + light.uv2.y * b2;
    float4 texel = tex2D<float4>(light.emissiveTex, u, v);
    return make_float3(texel.x, texel.y, texel.z) * light.emission;
}

static __forceinline__ __device__ uint32_t sampleAreaLightIndex(
    const float* cdf, uint32_t count, float target)
{
    uint32_t low = 0, high = count;
    while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (target <= cdf[mid]) high = mid;
        else low = mid + 1;
    }
    return (low >= count) ? (count - 1) : low;
}

// ── Ray generation helper ─────────────────────────────────────
static __forceinline__ __device__ Ray generateRay(
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

// ── Raygen ────────────────────────────────────────────────────
extern "C" __global__ void __raygen__path_trace()
{
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    const DeviceSceneData& scene = params.scene;
    const CameraParams&    camera = params.camera;
    uint32_t samplesPerPixel = params.spp < 1u ? 1u : params.spp;
    uint32_t maxBounces = params.maxBounces;
    bool enableEnvironment = params.enableEnvironment != 0;
    OptixTraversableHandle handle = params.handle;

    float3 radianceSum = make_float3(0, 0, 0);
    bool gbufferWritten = false;

    // DLSSOnly publishes to `gbuffer.hdrColor`; in that path the sub-pixel
    // offset must exactly match `camera.jitterOffset` (Halton). An extra
    // per-sample random offset would confuse DLSS's temporal reprojection
    // and produce ghosting / shimmer.
    const bool dlssPublish = (params.gbuffer.hdrColor != 0);

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
        uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                                  params.sampleIndex * 0x85EBCA6Bu + s);

        // ReSTIR (DI / GI / PT) reservoirs were generated against the surface
        // hit by the unjittered camera ray (only camera.jitterOffset, no
        // per-sample random sub-pixel jitter). To consume them at this pixel
        // the primary ray must hit the SAME surface — otherwise the stored
        // pHat / W is for a different shading point than the integrand we
        // evaluate below, producing strong overexposure on glossy/reflective
        // surfaces. Mirrors PathTraceKernel.cu's `restirActiveBounce0`.
        bool restirActiveBounce0 =
            (s == 0) && (
                (scene.restirEnabled   != 0 && scene.d_restirReservoirs != nullptr) ||
                (scene.restirGIEnabled != 0 && scene.d_restirGIIndirect != nullptr) ||
                (scene.restirPTEnabled != 0 && scene.d_restirPTIndirect != nullptr));

        float jx, jy;
        if (dlssPublish || restirActiveBounce0) {
            jx = camera.jitterOffset.x;
            jy = camera.jitterOffset.y;
        } else {
            jx = pcg32_float(rng) - 0.5f;
            jy = pcg32_float(rng) - 0.5f;
            jx += camera.jitterOffset.x;
            jy += camera.jitterOffset.y;
        }

        Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

        float3 throughput = make_float3(1, 1, 1);
        float3 radiance   = make_float3(0, 0, 0);
        bool firstBounce  = true;
        // True only when the previous bounce used a *delta* BSDF (perfect
        // mirror / glass refraction). See PathTraceKernel.cu for the longer
        // comment: Cook-Torrance specular is NOT delta and must not set this.
        bool lastBounceDelta = false;
        bool havePrevSurface = false;
        float3 prevSurfacePos = make_float3(0, 0, 0);
        float prevBsdfPdf = 1.0f;

        for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
            RadiancePayload rp = traceRadianceRay(
                handle, ray.origin, ray.direction, ray.tmin, ray.tmax);

            bool didHit = (rp.hit != 0);

            if (!didHit) {
                if (enableEnvironment) {
                    bool shForThisBounce = (bounce > 0) && !lastBounceDelta;
                    float3 envColor = sampleEnvironmentForBounce(
                        ray.direction, scene.envMapTex,
                        scene.d_shEnvCoeffs, scene.envUseSH != 0,
                        !shForThisBounce);
                    float envLum = 0.2126f * envColor.x + 0.7152f * envColor.y + 0.0722f * envColor.z;
                    float envClamp = 100.0f;
                    if (envLum > envClamp) envColor = envColor * (envClamp / envLum);
                    radiance += throughput * envColor;
                }
                // Sky pixel: write a sentinel viewZ so NRD treats it as sky
                // (any value > denoisingRange) and DLSS sees uniform far depth.
                // Motion vector stays 0 — sky doesn't reproject by camera.
                if (firstBounce && !gbufferWritten) {
                    if (params.gbuffer.viewZ) {
                        surf2Dwrite<float>(1.0e6f, params.gbuffer.viewZ, x * 4, y);
                    }
                    if (params.gbuffer.motionVectors) {
                        ushort2 zero = make_ushort2(0, 0);
                        surf2Dwrite<ushort2>(zero, params.gbuffer.motionVectors, x * 4, y);
                    }
                    if (params.gbuffer.ndcDepth) {
                        surf2Dwrite<float>(1.0f, params.gbuffer.ndcDepth, x * 4, y); // far
                    }
                    gbufferWritten = true;
                    firstBounce = false;
                }
                break;
            }

            // Reconstruct hit record from payload + scene buffers.
            HitRecord hit;
            hit.t             = rp.tHit;
            hit.primitiveIndex = (int)rp.primIdx;
            float baryU = rp.baryU;
            float baryV = rp.baryV;
            float baryW = 1.0f - baryU - baryV;

            uint32_t triIdx = rp.primIdx;
            uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
            uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
            uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
            float3 v0 = scene.d_positions[i0];
            float3 v1 = scene.d_positions[i1];
            float3 v2 = scene.d_positions[i2];
            hit.position = v0 * baryW + v1 * baryU + v2 * baryV;

            float3 geomN = normalize(cross(v1 - v0, v2 - v0));
            hit.shadingNormal = geomN;
            hit.normal        = geomN;
            hit.frontFace = (dot(ray.direction, geomN) < 0.0f);
            hit.uv = make_float2(baryU, baryV);
            hit.materialIndex = scene.d_materialIndices ? scene.d_materialIndices[triIdx] : -1;

            GPUMaterial mat;
            if (hit.materialIndex >= 0 && (uint32_t)hit.materialIndex < scene.materialCount)
                mat = scene.d_materials[hit.materialIndex];
            else {
                mat.albedo = make_float3(0.8f, 0.2f, 0.8f);
                mat.roughness = 0.5f;
                mat.metallic = 0.0f;
                mat.emission = make_float3(0,0,0);
                mat.emissionStrength = 0.0f;
                mat.ior = 1.5f;
                mat.transmission = 0.0f;
                mat.albedoTex = 0;
                mat.metallicRoughTex = 0;
                mat.emissiveTex = 0;
                mat.normalTex = 0;
                mat.specularGlossTex = 0;
                mat.useSpecularGlossiness = 0;
                mat.specularGlossAlphaIsGlossiness = 0;
                mat.useFBXCustomPacking = 0;
                mat.useFBXUEPacking = 0;
                mat.specularColor = make_float3(1.0f, 1.0f, 1.0f);
                mat.glossiness = 0.5f;
                mat.pureDiffuse = 0;
            }

            float2 texUV = make_float2(0.0f, 0.0f);
            if (scene.d_uvs) {
                float2 uv0 = scene.d_uvs[i0];
                float2 uv1 = scene.d_uvs[i1];
                float2 uv2 = scene.d_uvs[i2];
                texUV = uv0 * baryW + uv1 * baryU + uv2 * baryV;
            }

            float3 albedo = mat.albedo;
            if (mat.albedoTex != 0) {
                float4 texColor = tex2D<float4>(mat.albedoTex, texUV.x, texUV.y);
                albedo = make_float3(texColor.x, texColor.y, texColor.z);
            }
            if (mat.metallicRoughTex != 0) {
                float4 mrTexel = tex2D<float4>(mat.metallicRoughTex, texUV.x, texUV.y);
                mat.roughness = mat.roughness * mrTexel.y;
                mat.metallic = mat.metallic * mrTexel.z;
            }
            // SG "soft" interpretation (see PathTraceKernel.cu for rationale).
            if (mat.useSpecularGlossiness) {
                if (mat.useFBXCustomPacking && mat.specularGlossTex != 0) {
                    float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
                    float B = clampf(sg.z, 0.0f, 1.0f);
                    float G = clampf(sg.y, 0.0f, 1.0f);
                    albedo = mat.specularColor;
                    mat.metallic = B;
                    mat.roughness = G;
                } else if (mat.useFBXUEPacking && mat.specularGlossTex != 0) {
                    float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
                    float G = clampf(sg.y, 0.0f, 1.0f);
                    float B = clampf(sg.z, 0.0f, 1.0f);
                    mat.metallic  = B;
                    mat.roughness = 1.0f - G;
                } else {
                    float3 specRGB = mat.specularColor;
                    float  alphaG  = 1.0f;
                    if (mat.specularGlossTex != 0) {
                        float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
                        specRGB = mat.specularColor * make_float3(sg.x, sg.y, sg.z);
                        alphaG  = sg.w;
                    }
                    float specLum = 0.2126f * specRGB.x + 0.7152f * specRGB.y + 0.0722f * specRGB.z;
                    specLum = clampf(specLum, 0.0f, 1.0f);
                    float specStrength = sqrtf(specLum);
                    float F0_target = 0.04f + 0.56f * specStrength;
                    mat.metallic = clampf((F0_target - 0.04f) / 0.96f, 0.0f, 1.0f);
                    float gloss = mat.specularGlossAlphaIsGlossiness
                                    ? (mat.glossiness * alphaG)
                                    :  specStrength;
                    mat.roughness = 1.0f - clampf(gloss, 0.0f, 0.95f);
                }
            }
            mat.roughness = fmaxf(mat.roughness, 0.045f);
            mat.metallic = clampf(mat.metallic, 0.0f, 1.0f);

            float3 emissiveColor = mat.emission;
            if (mat.emissiveTex != 0) {
                float4 emissiveTexel = tex2D<float4>(mat.emissiveTex, texUV.x, texUV.y);
                emissiveColor = make_float3(emissiveTexel.x, emissiveTexel.y, emissiveTexel.z);
            }

            float3 N = hit.shadingNormal;
            if (scene.d_normals) {
                float3 n0 = scene.d_normals[i0];
                float3 n1 = scene.d_normals[i1];
                float3 n2 = scene.d_normals[i2];
                N = normalize(n0 * baryW + n1 * baryU + n2 * baryV);
            }
            if (mat.transmission <= 0.0f && mat.normalTex != 0 && scene.d_tangents) {
                float4 t0 = scene.d_tangents[i0];
                float4 t1 = scene.d_tangents[i1];
                float4 t2 = scene.d_tangents[i2];
                float4 tangent = t0 * baryW + t1 * baryU + t2 * baryV;
                N = applyNormalMap(N, tangent, mat.normalTex, texUV);
            }
            if (mat.transmission <= 0.0f) {
                if (dot(N, ray.direction) > 0) N = -N;
            }

            if (firstBounce) {
                if (!gbufferWritten) {
                    float viewZprim = dot(hit.position - camera.position, camera.forward);
                    float3 clipCurr = mat4_transformPoint(camera.viewProjMatrix, hit.position);
                    float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hit.position);
                    float2 screenCurr = make_float2((clipCurr.x + 1.0f) * 0.5f * params.width,
                                                     (1.0f - clipCurr.y) * 0.5f * params.height);
                    float2 screenPrev = make_float2((clipPrev.x + 1.0f) * 0.5f * params.width,
                                                     (1.0f - clipPrev.y) * 0.5f * params.height);
                    // DLSS / NRD MV convention: "where was this pixel last
                    // frame" = `prev - curr`. See PathTraceKernel.cu for the
                    // longer comment.
                    float2 mvPx = screenPrev - screenCurr;

                    if (params.aux.d_linearDepth)   params.aux.d_linearDepth[pixelIdx] = viewZprim;
                    if (params.aux.d_albedo)        params.aux.d_albedo[pixelIdx]      = albedo;
                    if (params.aux.d_normal)        params.aux.d_normal[pixelIdx]      = N;
                    if (params.aux.d_motionVectors) params.aux.d_motionVectors[pixelIdx] = mvPx;

                    // DLSSOnly / NRD: also write to Vulkan-shared surfaces so
                    // post-processing can read them as VkImages directly. Only
                    // the first-sample primary hit wins (averaging across SPP
                    // would soften silhouettes and break temporal reprojection).
                    if (params.gbuffer.viewZ) {
                        surf2Dwrite<float>(viewZprim, params.gbuffer.viewZ, x * 4, y);  // R32F
                    }
                    if (params.gbuffer.motionVectors) {
                        __half hx = __float2half(mvPx.x);
                        __half hy = __float2half(mvPx.y);
                        ushort2 packed;
                        packed.x = *reinterpret_cast<unsigned short*>(&hx);
                        packed.y = *reinterpret_cast<unsigned short*>(&hy);
                        surf2Dwrite<ushort2>(packed, params.gbuffer.motionVectors, x * 4, y);  // RG16F
                    }
                    if (params.gbuffer.ndcDepth) {
                        // DLSS needs NDC depth in [0,1], not linear viewZ.
                        // `clipCurr` has already been perspective-divided.
                        float ndcZ = clampf(clipCurr.z * 0.5f + 0.5f, 0.0f, 1.0f);
                        surf2Dwrite<float>(ndcZ, params.gbuffer.ndcDepth, x * 4, y);
                    }

                    gbufferWritten = true;
                }
                firstBounce = false;
            }

            // Glass / transmissive
            bool handledAsGlass = false;
            if (mat.transmission > 0.0f) {
                bool entering = hit.frontFace;
                float3 Nglass = entering ? N : -N;
                if (dot(Nglass, ray.direction) > 0.0f) Nglass = -Nglass;

                float etaI = entering ? 1.0f : mat.ior;
                float etaT = entering ? mat.ior : 1.0f;
                float eta = etaI / etaT;
                float cosThetaI = fmaxf(dot(-ray.direction, Nglass), 0.0f);
                float Fr = fresnelDielectric(cosThetaI, eta);

                float3 newDirGlass;
                if (pcg32_float(rng) < Fr) {
                    newDirGlass = ray.direction - Nglass * (2.0f * dot(ray.direction, Nglass));
                    newDirGlass = normalize(newDirGlass);
                } else {
                    if (!refractDir(ray.direction, Nglass, eta, newDirGlass)) {
                        newDirGlass = ray.direction - Nglass * (2.0f * dot(ray.direction, Nglass));
                        newDirGlass = normalize(newDirGlass);
                    }
                }

                if (!entering) {
                    float albedoLum = 0.2126f * albedo.x + 0.7152f * albedo.y + 0.0722f * albedo.z;
                    if (albedoLum < 0.9f) {
                        throughput = throughput * albedo;
                    }
                }

                float3 offsetN = (dot(newDirGlass, Nglass) > 0.0f) ? Nglass : -Nglass;
                ray.origin    = hit.position + offsetN * 0.002f;
                ray.direction = newDirGlass;
                ray.tmin      = 0.001f;
                ray.tmax      = 1e30f;

                lastBounceDelta = true;   // glass is delta
                prevSurfacePos = hit.position;
                prevBsdfPdf = 1.0f;
                havePrevSurface = true;
                handledAsGlass = true;
            }
            if (handledAsGlass) {
                if (dot(N, ray.direction) > 0) N = -N;
                if (bounce >= 6) {
                    if (pcg32_float(rng) > 0.9f) break;
                }
                continue;
            }

            bool isEmissive = mat.emissionStrength > 0.0f &&
                              (emissiveColor.x > 0.0f || emissiveColor.y > 0.0f || emissiveColor.z > 0.0f);
            if (isEmissive) {
                float3 Le = emissiveColor * mat.emissionStrength;
                float weight = 1.0f;
                bool isAreaLight = false;
                if (bounce > 0 && havePrevSurface && !lastBounceDelta && scene.d_triangleAreaLightIndex) {
                    int areaLightIndex = scene.d_triangleAreaLightIndex[(uint32_t)hit.primitiveIndex];
                    if (areaLightIndex >= 0 && scene.d_areaLights && scene.areaLightCount > 0) {
                        isAreaLight = true;
                        GPUAreaLight light = scene.d_areaLights[areaLightIndex];
                        float3 toLight = hit.position - prevSurfacePos;
                        float dist2 = fmaxf(dot(toLight, toLight), 1e-6f);
                        float3 wi = normalize(toLight);
                        float lightNdot = fmaxf(dot(light.normal, -wi), 0.0f);
                        if (lightNdot > 0.0f) {
                            float pTri = light.weight / fmaxf(scene.areaLightTotalWeight, 1e-7f);
                            float pArea = pTri / fmaxf(light.area, 1e-7f);
                            float pLight = pArea * dist2 / fmaxf(lightNdot, 1e-7f);
                            float pBsdf = prevBsdfPdf;
                            weight = powerHeuristic(pBsdf, pLight);
                        }
                    }
                }
                radiance += throughput * Le * weight;
                if (mat.emissiveTex != 0) {
                    // fall through
                } else {
                    break;
                }
            }

            // Direct lighting NEE
            if (scene.d_areaLights && scene.areaLightCount > 0 &&
                scene.d_areaLightCDF && scene.areaLightTotalWeight > 0.0f) {
                // ReSTIR DI is applied at the primary hit (bounce 0) only —
                // the reservoir buffer is populated for camera rays only.
                bool restirActive = (scene.restirEnabled != 0) &&
                                    (scene.d_restirReservoirs != nullptr) &&
                                    (bounce == 0) && (s == 0);
                bool restirSkip = false;
                uint32_t lightIndex = 0;
                float    b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
                float    restirW = 0.0f;
                if (restirActive) {
                    const ReSTIRReservoir* res =
                        reinterpret_cast<const ReSTIRReservoir*>(scene.d_restirReservoirs);
                    ReSTIRReservoir r = res[pixelIdx];
                    if (r.lightIndex == 0xFFFFFFFFu || r.W <= 0.0f || r.pHat <= 0.0f) {
                        // Empty reservoir (e.g. surface-culled) → skip NEE at
                        // this bounce. Indirect via BSDF / GI/PT consumption
                        // below still works, so the image is unbiased on
                        // average.
                        restirSkip = true;
                    } else {
                        lightIndex = r.lightIndex;
                        b1 = r.baryB1;
                        b2 = r.baryB2;
                        b0 = 1.0f - b1 - b2;
                        restirW = r.W;
                    }
                } else {
                    lightIndex = sampleAreaLightIndex(
                        scene.d_areaLightCDF, scene.areaLightCount,
                        pcg32_float(rng));

                    float r1 = pcg32_float(rng);
                    float r2 = pcg32_float(rng);
                    float su = sqrtf(r1);
                    b0 = 1.0f - su;
                    b1 = su * (1.0f - r2);
                    b2 = su * r2;
                }

                if (!restirSkip) {
                GPUAreaLight light = scene.d_areaLights[lightIndex];

                float3 lightV0 = light.v0;
                float3 lightV1 = light.v0 + light.e1;
                float3 lightV2 = light.v0 + light.e2;
                float3 lightPos = lightV0 * b0 + lightV1 * b1 + lightV2 * b2;

                float3 toLight = lightPos - hit.position;
                float dist2 = fmaxf(dot(toLight, toLight), 1e-6f);
                float dist = sqrtf(dist2);
                float3 Ld = toLight * (1.0f / dist);

                float NdotL = fmaxf(dot(N, Ld), 0.0f);
                float lightNdot = fmaxf(dot(light.normal, -Ld), 0.0f);
                if (NdotL > 0.0f && lightNdot > 0.0f) {
                    float3 shadowOrigin = hit.position + N * 0.001f;
                    float shadowTmax = fmaxf(dist - 0.002f, 0.001f);
                    float3 shadowTransmittance = traceShadowRay(
                        handle, shadowOrigin, Ld, 0.001f, shadowTmax);
                    float shadowLum = 0.2126f * shadowTransmittance.x +
                                      0.7152f * shadowTransmittance.y +
                                      0.0722f * shadowTransmittance.z;
                    if (shadowLum > 1e-6f) {
                        float3 V = -ray.direction;
                        float3 brdf = materialBsdfEvaluate(mat, N, V, Ld, albedo);
                        float3 Le = sampleAreaLightLe(light, b0, b1, b2);

                        if (restirActive) {
                            // ReSTIR estimator: f(x) * W where f is the
                            // unshadowed integrand (BRDF * Le * G * NdotL) and
                            // W is the reservoir's contribution weight. No MIS
                            // against BSDF sampling — ReSTIR *is* the
                            // light-side strategy at the primary hit.
                            float geom = lightNdot / dist2;
                            float3 neeContrib = throughput * shadowTransmittance *
                                                brdf * Le * (NdotL * geom) * restirW;
                            // Mirror PathTraceKernel.cu / Split: cap the per-
                            // frame contribution so a near-grazing reservoir
                            // sample doesn't dump a 50-lum spike into the
                            // accumulator (M7 flash-and-decay artifact).
                            float lumNee = 0.2126f * neeContrib.x +
                                           0.7152f * neeContrib.y +
                                           0.0722f * neeContrib.z;
                            const float clampMax = 10.0f;
                            if (lumNee > clampMax) {
                                neeContrib = neeContrib * (clampMax / lumNee);
                            }
                            radiance += neeContrib;
                        } else {
                            float pTri = light.weight / scene.areaLightTotalWeight;
                            float pArea = pTri / fmaxf(light.area, 1e-7f);
                            float pdfOmega = pArea * dist2 / fmaxf(lightNdot, 1e-7f);

                            float neeSpecProb = materialSpecProb(mat, N, V, albedo);
                            float pdfBsdf = materialMixturePdf(mat, N, V, Ld, neeSpecProb);
                            float weight = powerHeuristic(pdfOmega, pdfBsdf);

                            radiance += throughput * shadowTransmittance * brdf *
                                        Le *
                                        (NdotL / fmaxf(pdfOmega, 1e-7f)) * weight;
                        }
                    }
                }
                } // end !restirSkip
            }

            // Point lights are delta emitters — always sampled, regardless of
            // whether area lights also exist. See PathTraceKernel.cu comment.
            else if (scene.d_pointLights && scene.pointLightCount > 0) {
                float3 direct = make_float3(0.0f, 0.0f, 0.0f);
                float3 V = -ray.direction;
                for (uint32_t li = 0; li < scene.pointLightCount; li++) {
                    GPUPointLight light = scene.d_pointLights[li];
                    float3 toLight = light.position - hit.position;
                    float dist2 = fmaxf(dot(toLight, toLight), 1e-6f);
                    float dist = sqrtf(dist2);
                    float3 Ld = toLight * (1.0f / dist);
                    float NdotL = fmaxf(dot(N, Ld), 0.0f);
                    if (NdotL <= 0.0f) continue;

                    float3 shadowOrigin = hit.position + N * 0.001f;
                    float shadowTmax = fmaxf(dist - 0.002f, 0.001f);
                    float3 shadowTransmittancePL = traceShadowRay(
                        handle, shadowOrigin, Ld, 0.001f, shadowTmax);
                    float shadowLumPL = 0.2126f * shadowTransmittancePL.x +
                                        0.7152f * shadowTransmittancePL.y +
                                        0.0722f * shadowTransmittancePL.z;
                    if (shadowLumPL < 1e-6f) continue;

                    float attenDen = light.constantAttenuation
                                   + light.linearAttenuation * dist
                                   + light.quadraticAttenuation * dist2;
                    float attenuation = 1.0f / fmaxf(attenDen, 1e-4f);
                    float3 Li = light.color * (light.intensity * attenuation);
                    float3 brdf = materialBsdfEvaluate(mat, N, V, Ld, albedo);
                    direct += brdf * shadowTransmittancePL * Li * NdotL;
                }
                radiance += throughput * direct;
            }

            if (scene.d_directionalLights && scene.directionalLightCount > 0) {
                float3 direct = make_float3(0.0f, 0.0f, 0.0f);
                float3 V = -ray.direction;
                for (uint32_t li = 0; li < scene.directionalLightCount; li++) {
                    GPUDirectionalLight light = scene.d_directionalLights[li];
                    float3 Ld = light.direction;
                    float NdotL = fmaxf(dot(N, Ld), 0.0f);
                    if (NdotL <= 0.0f) continue;

                    float3 shadowTransmittanceDL = traceShadowRay(
                        handle, hit.position + N * 0.001f, Ld, 0.001f, 1e30f);
                    float shadowLumDL = 0.2126f * shadowTransmittanceDL.x +
                                        0.7152f * shadowTransmittanceDL.y +
                                        0.0722f * shadowTransmittanceDL.z;
                    if (shadowLumDL < 1e-6f) continue;

                    float3 brdf = materialBsdfEvaluate(mat, N, V, Ld, albedo);
                    direct += brdf * shadowTransmittanceDL * light.color * NdotL;
                }
                radiance += throughput * direct;
            }

            // ReSTIR PT / GI consumption at the primary hit on sample s==0.
            // PT takes precedence (its postfix already contains GI's 1-bounce
            // NEE plus k more bounces' worth of light transport). Either branch
            // adds the pre-computed indirect estimate and skips continuation
            // bounces; the direct lighting at the primary hit was already
            // added above. Restricted to bounce==0 because the PT/GI buffer is
            // populated for camera rays only — higher bounces fall through to
            // plain BSDF sampling.
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

            // BRDF sampling
            float3 V = -ray.direction;
            float specProb = materialSpecProb(mat, N, V, albedo);
            float3 newDir;

            if (pcg32_float(rng) < specProb) {
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
                // Cook-Torrance specular lobe is NOT delta — MIS is valid.
                lastBounceDelta = false;
            } else {
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

            float pdf = materialMixturePdf(mat, N, V, newDir, specProb);
            if (pdf < 1e-7f) break;

            float3 brdf = materialBsdfEvaluate(mat, N, V, newDir, albedo);
            throughput = throughput * brdf * (NdotL_new / (pdf + 1e-7f));

            prevSurfacePos = hit.position;
            prevBsdfPdf = pdf;
            havePrevSurface = true;

            if (bounce >= 2) {
                float lum = 0.2126f * throughput.x + 0.7152f * throughput.y + 0.0722f * throughput.z;
                float p = fminf(fmaxf(lum, 0.05f), 0.95f);
                if (pcg32_float(rng) >= p) break;
                throughput = throughput * (1.0f / p);
            }

            ray.origin    = hit.position + N * 0.001f;
            ray.direction = newDir;
            ray.tmin      = 0.001f;
            ray.tmax      = 1e30f;
        }

        if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z) ||
            isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z)) {
            radiance = make_float3(0.0f, 0.0f, 0.0f);
        }
        float luminance = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
        float clampMax = 200.0f;
        if (luminance > clampMax) {
            radiance = radiance * (clampMax / luminance);
        }
        radianceSum = radianceSum + radiance;
    }

    float4 sumTexel = make_float4(radianceSum.x, radianceSum.y, radianceSum.z, (float)samplesPerPixel);
    params.accum[pixelIdx] = params.accum[pixelIdx] + sumTexel;
    float invN = 1.0f / (float)(params.sampleIndex + samplesPerPixel);
    float4 hdr = params.accum[pixelIdx] * invN;
    if (params.output) params.output[pixelIdx] = hdr;

    // DLSSOnly: also publish HDR into the Vulkan-shared interop image so the
    // post-processing chain can sample it. Packed as four halves matching the
    // RGBA16F VkImage format of `m_hdrColor`.
    if (params.gbuffer.hdrColor) {
        __half hx = __float2half(hdr.x);
        __half hy = __float2half(hdr.y);
        __half hz = __float2half(hdr.z);
        __half hw = __float2half(1.0f);
        ushort4 p;
        p.x = *reinterpret_cast<unsigned short*>(&hx);
        p.y = *reinterpret_cast<unsigned short*>(&hy);
        p.z = *reinterpret_cast<unsigned short*>(&hz);
        p.w = *reinterpret_cast<unsigned short*>(&hw);
        surf2Dwrite<ushort4>(p, params.gbuffer.hdrColor, x * 8, y);  // RGBA16F = 8B
    }
}

// ── Miss: radiance ────────────────────────────────────────────
extern "C" __global__ void __miss__radiance()
{
    optixSetPayload_0(0);  // hit = false
}

// ── Closest-hit: radiance ─────────────────────────────────────
extern "C" __global__ void __closesthit__radiance()
{
    float2 bary = optixGetTriangleBarycentrics();
    optixSetPayload_0(1);
    optixSetPayload_1(optixGetPrimitiveIndex());
    optixSetPayload_2(__float_as_uint(bary.x));
    optixSetPayload_3(__float_as_uint(bary.y));
    optixSetPayload_4(__float_as_uint(optixGetRayTmax()));
}

// ── Miss: shadow (no-op; leaves transmittance untouched) ──────
extern "C" __global__ void __miss__shadow()
{
    // Unoccluded miss — payload already holds accumulated transmittance.
}

// ── Any-hit: shadow (glass-transparent occlusion) ─────────────
extern "C" __global__ void __anyhit__shadow()
{
    uint32_t primIdx = optixGetPrimitiveIndex();
    const DeviceSceneData& scene = params.scene;

    int matIdx = scene.d_materialIndices ? scene.d_materialIndices[primIdx] : -1;
    if (matIdx < 0 || (uint32_t)matIdx >= scene.materialCount) {
        // Unknown material: treat as opaque.
        optixSetPayload_3(1);
        optixTerminateRay();
        return;
    }
    GPUMaterial mat = scene.d_materials[matIdx];
    if (mat.transmission > 0.0f) {
        // Attenuate by albedo for colored glass; near-white treated as transparent.
        float tx = __uint_as_float(optixGetPayload_0());
        float ty = __uint_as_float(optixGetPayload_1());
        float tz = __uint_as_float(optixGetPayload_2());
        float albLum = 0.2126f * mat.albedo.x + 0.7152f * mat.albedo.y + 0.0722f * mat.albedo.z;
        if (albLum < 0.9f) {
            tx *= mat.albedo.x;
            ty *= mat.albedo.y;
            tz *= mat.albedo.z;
        }
        optixSetPayload_0(__float_as_uint(tx));
        optixSetPayload_1(__float_as_uint(ty));
        optixSetPayload_2(__float_as_uint(tz));
        optixIgnoreIntersection();
        return;
    }
    // Opaque hit: terminate.
    optixSetPayload_3(1);
    optixTerminateRay();
}

// ─────────────────────────────────────────────────────────────────────────────
// Split-output raygen (NRD).
// Mirrors PathTraceKernelSplit.cu's algorithm but uses optixTrace + the same
// hit/miss/anyhit programs as the regular raygen above. Writes 7 Vulkan-shared
// surfaces (diff/spec radiance + g-buffer) instead of an HDR accum buffer.
//
// Per-pixel SPP averaging happens in this raygen (NRD wants the mean, and
// averaging cuts single-sample bucket spikes that read as ripples after
// temporal filtering). Each spp-loop iteration does one full path.
// ─────────────────────────────────────────────────────────────────────────────

// Lobe-only BRDF evaluators (port of bsdfDiffuseLobe / bsdfSpecularLobe in
// PathTraceKernelSplit.cu). Used at the primary hit when forcing NEE/BSDF
// to the picked bucket, so that diff_bucket * albedo + spec_bucket recovers
// the full primary-hit radiance.
static __forceinline__ __device__ float3 splitDiffuseLobe(
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

static __forceinline__ __device__ float3 splitSpecularLobe(
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

static __forceinline__ __device__ float3 materialDiffuseLobe_split(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& L, const float3& albedo)
{
    if (mat.pureDiffuse) {
        float NdotL = fmaxf(dot(N, L), 0.0f);
        float NdotV = fmaxf(dot(N, V), 0.0f);
        if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0, 0, 0);
        return albedo * (1.0f / M_PI_F);
    }
    return splitDiffuseLobe(N, V, L, albedo, mat.roughness, mat.metallic);
}

static __forceinline__ __device__ float3 materialSpecularLobe_split(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& L, const float3& albedo)
{
    if (mat.pureDiffuse) return make_float3(0, 0, 0);
    return splitSpecularLobe(N, V, L, albedo, mat.roughness, mat.metallic);
}

// Per-contribution firefly clamp (luminance-bounded). RELAX is sensitive to
// single-sample spikes that survive temporal filtering for many frames as
// shimmering specks ("water ripples"); clamp each NEE / emissive contribution.
static __forceinline__ __device__ float3 clampFirefly_split(float3 c, float maxLum) {
    float lum = 0.2126f*c.x + 0.7152f*c.y + 0.0722f*c.z;
    if (lum > maxLum && lum > 1e-7f) c = c * (maxLum / lum);
    return c;
}

// Lift packHalf4 out of the raygen body — lambdas inside OptiX raygen
// functions have caused misaligned-stack issues in some toolchain versions.
// Use the __half_as_ushort intrinsic (no pointer reinterpret) to avoid any
// alignment ambiguity.
static __forceinline__ __device__ ushort4 packHalf4_split(float4 v) {
    ushort4 r;
    r.x = __half_as_ushort(__float2half(v.x));
    r.y = __half_as_ushort(__float2half(v.y));
    r.z = __half_as_ushort(__float2half(v.z));
    r.w = __half_as_ushort(__float2half(v.w));
    return r;
}

static __forceinline__ __device__ ushort2 packHalf2_split(float x, float y) {
    ushort2 r;
    r.x = __half_as_ushort(__float2half(x));
    r.y = __half_as_ushort(__float2half(y));
    return r;
}

// Workaround: surf2Dwrite<uchar4> in OptiX-compiled device code emits a
// PTX store that faults with "misaligned address" on Ampere+ (verified on
// RTX 4070, OptiX 9.0). Writing the same 4 bytes as a uint32_t works.
// Same byte layout (LE: byte0=r, byte1=g, byte2=b, byte3=a).
static __forceinline__ __device__ uint32_t packRGBA8(float r, float g, float b, float a) {
    uint32_t br = (uint32_t)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t bg = (uint32_t)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t bb = (uint32_t)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t ba = (uint32_t)(fminf(fmaxf(a, 0.0f), 1.0f) * 255.0f + 0.5f);
    return (ba << 24) | (bb << 16) | (bg << 8) | br;
}

// DLSS-RR §3.4.2 / Appendix: per-pixel specular albedo from F0, alpha, NoV.
static __forceinline__ __device__ float3 envBRDFApprox2_split(
    float3 F0, float alpha, float NoV)
{
    NoV = fabsf(NoV);
    float NoV2 = NoV * NoV;
    float NoV3 = NoV2 * NoV;
    float a3   = alpha * alpha * alpha;
    float M1xy_top = 0.99044f - 1.28514f * NoV;
    float M1xy_bot = 1.29678f - 0.755907f * NoV;
    float biasNum = M1xy_top + M1xy_bot * alpha;
    float M2_0 = 1.0f + 2.92338f * NoV + 59.4188f * NoV3;
    float M2_1 = 20.3225f - 27.0302f * NoV + 222.592f * NoV3;
    float M2_2 = 121.563f + 626.13f * NoV + 316.627f * NoV3;
    float biasDen = M2_0 + M2_1 * alpha + M2_2 * a3;
    float bias = biasNum / fmaxf(biasDen, 1e-7f);
    float M3xy_top = 0.0365463f + 3.32707f * NoV;
    float M3xy_bot = 9.0632f    - 9.04756f * NoV;
    float scaleNum = M3xy_top + M3xy_bot * alpha;
    float M4_0 = 1.0f + 3.59685f * NoV2 - 1.36772f * NoV3;
    float M4_1 = 9.04401f - 16.3174f * NoV2 + 9.22949f * NoV3;
    float M4_2 = 5.56589f + 19.7886f * NoV2 - 20.2123f * NoV3;
    float scaleDen = M4_0 + M4_1 * alpha + M4_2 * a3;
    float scale = scaleNum / fmaxf(scaleDen, 1e-7f);
    bias  *= fminf(fmaxf(F0.y * 50.0f, 0.0f), 1.0f);
    scale = fmaxf(scale, 0.0f);
    bias  = fmaxf(bias, 0.0f);
    return make_float3(F0.x * scale + bias,
                       F0.y * scale + bias,
                       F0.z * scale + bias);
}

extern "C" __global__ void __raygen__path_trace_split()
{
    uint3 idx = optixGetLaunchIndex();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    const DeviceSceneData& scene  = params.scene;
    const CameraParams&    camera = params.camera;
    uint32_t samplesPerPixel = params.spp < 1u ? 1u : params.spp;
    uint32_t maxBounces      = params.maxBounces;
    bool enableEnvironment   = params.enableEnvironment != 0;
    OptixTraversableHandle handle = params.handle;

    // Per-pixel accumulators across spp.
    float3 demodDiffSum = make_float3(0, 0, 0);
    float3 demodSpecSum = make_float3(0, 0, 0);
    float3 emissiveSum  = make_float3(0, 0, 0);
    float  diffHitSum = 0.0f; uint32_t diffHitCount = 0;
    float  specHitSum = 0.0f; uint32_t specHitCount = 0;
    // DLSS-RR additional accumulators.
    float3 noisyColorSum = make_float3(0, 0, 0);
    float  anyHitSum = 0.0f; uint32_t anyHitCount = 0;

    bool   gbufferWritten = false;
    float3 outPrimaryAlbedo    = make_float3(0, 0, 0);
    float3 outPrimaryNormal    = make_float3(0, 1, 0);
    float  outPrimaryRoughness = 1.0f;
    float  outPrimaryViewZ     = 0.0f;
    float2 outPrimaryMvPx      = make_float2(0.0f, 0.0f);
    float  outPrimaryNdcZ      = 1.0f;
    // DLSS-RR fix: see PathTraceKernelSplit.cu — preserve primary hit pos /
    // view ray dir / metallic for post-loop mirror-ray spec hitT trace and
    // metallic-aware spec albedo F0.
    float3 outPrimaryHitPos    = make_float3(0, 0, 0);
    float3 outPrimaryRayDir    = make_float3(0, 0, -1);
    float  outPrimaryMetallic  = 0.0f;

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
        uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                                  params.sampleIndex * 0x85EBCA6Bu + s);

        // NRD/DLSS require the ray-gen sub-pixel offset to exactly match
        // `camera.jitterOffset` (Halton). See PathTraceKernelSplit.cu for
        // the longer comment — an extra per-sample random offset makes
        // history reprojection chase the wrong sub-pixel and shows up as
        // still-frame "water wave" jitter.
        float jx = camera.jitterOffset.x;
        float jy = camera.jitterOffset.y;

        Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

        float3 throughput      = make_float3(1, 1, 1);
        float3 pathRadiance    = make_float3(0, 0, 0);
        float3 emissiveContrib = make_float3(0, 0, 0);

        // Per-sample primary-hit state for bucket classification.
        bool haveGbuffer = false;
        float3 primaryAlbedo    = make_float3(0, 0, 0);
        float3 primaryNormal    = make_float3(0, 1, 0);
        float  primaryRoughness = 1.0f;
        float  primaryViewZ     = 0.0f;
        float2 primaryMvPx      = make_float2(0.0f, 0.0f);
        float  primaryNdcZ      = 1.0f;
        int    pickedBucket     = 0;       // 0 = diff, 1 = spec
        float  bucketHitDist    = 0.0f;
        bool   bucketHitDistSet = false;
        float3 primaryHitPos    = make_float3(0, 0, 0);
        float3 primaryRayDir    = make_float3(0, 0, -1);
        float  primaryMetallic  = 0.0f;

        bool firstBounce      = true;
        bool lastBounceDelta  = false;
        bool havePrevSurface  = false;
        float3 prevSurfacePos = make_float3(0, 0, 0);
        float prevBsdfPdf     = 1.0f;

        for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
            // Whether the primary lobe override is active this iteration.
            bool primaryLobeOverride = false;

            RadiancePayload rp = traceRadianceRay(
                handle, ray.origin, ray.direction, ray.tmin, ray.tmax);
            bool didHit = (rp.hit != 0);

            if (!didHit) {
                if (enableEnvironment) {
                    bool shForThisBounce = (bounce > 0) && !lastBounceDelta;
                    float3 envColor = sampleEnvironmentForBounce(
                        ray.direction, scene.envMapTex,
                        scene.d_shEnvCoeffs, scene.envUseSH != 0,
                        !shForThisBounce);
                    float envLum = 0.2126f*envColor.x + 0.7152f*envColor.y + 0.0722f*envColor.z;
                    if (envLum > 20.0f) envColor = envColor * (20.0f / envLum);
                    pathRadiance += clampFirefly_split(throughput * envColor, 10.0f);
                }
                // Sky pixel sentinel viewZ — only the first sample's miss wins.
                if (firstBounce && !haveGbuffer) {
                    primaryViewZ = 1.0e6f;
                    primaryMvPx  = make_float2(0.0f, 0.0f);
                    // Don't set haveGbuffer — sky pixels don't contribute to
                    // diff/spec bucket and must not trigger the demodulation
                    // path below. We still want a sentinel viewZ written, so
                    // capture it via outPrimary fields directly.
                    if (!gbufferWritten) {
                        outPrimaryViewZ = primaryViewZ;
                        outPrimaryMvPx  = primaryMvPx;
                        // Leave normal/roughness/albedo at their defaults.
                    }
                }
                break;
            }

            // Reconstruct hit record from payload + scene buffers.
            HitRecord hit;
            hit.t              = rp.tHit;
            hit.primitiveIndex = (int)rp.primIdx;
            float baryU = rp.baryU;
            float baryV = rp.baryV;
            float baryW = 1.0f - baryU - baryV;

            uint32_t triIdx = rp.primIdx;
            uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
            uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
            uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
            float3 v0 = scene.d_positions[i0];
            float3 v1 = scene.d_positions[i1];
            float3 v2 = scene.d_positions[i2];
            hit.position = v0 * baryW + v1 * baryU + v2 * baryV;

            float3 geomN = normalize(cross(v1 - v0, v2 - v0));
            hit.shadingNormal = geomN;
            hit.normal        = geomN;
            hit.frontFace = (dot(ray.direction, geomN) < 0.0f);
            hit.uv = make_float2(baryU, baryV);
            hit.materialIndex = scene.d_materialIndices ? scene.d_materialIndices[triIdx] : -1;

            GPUMaterial mat;
            if (hit.materialIndex >= 0 && (uint32_t)hit.materialIndex < scene.materialCount)
                mat = scene.d_materials[hit.materialIndex];
            else {
                mat.albedo = make_float3(0.8f, 0.2f, 0.8f);
                mat.roughness = 0.5f; mat.metallic = 0.0f;
                mat.emission = make_float3(0,0,0); mat.emissionStrength = 0.0f;
                mat.ior = 1.5f; mat.transmission = 0.0f;
                mat.albedoTex = 0; mat.metallicRoughTex = 0;
                mat.emissiveTex = 0; mat.normalTex = 0; mat.specularGlossTex = 0;
                mat.useSpecularGlossiness = 0; mat.specularGlossAlphaIsGlossiness = 0;
                mat.useFBXCustomPacking = 0; mat.useFBXUEPacking = 0;
                mat.specularColor = make_float3(1.0f, 1.0f, 1.0f);
                mat.glossiness = 0.5f; mat.pureDiffuse = 0;
            }

            float2 texUV = make_float2(0.0f, 0.0f);
            if (scene.d_uvs) {
                float2 uv0 = scene.d_uvs[i0];
                float2 uv1 = scene.d_uvs[i1];
                float2 uv2 = scene.d_uvs[i2];
                texUV = uv0 * baryW + uv1 * baryU + uv2 * baryV;
            }

            float3 albedo = mat.albedo;
            if (mat.albedoTex != 0) {
                float4 tc = tex2D<float4>(mat.albedoTex, texUV.x, texUV.y);
                albedo = make_float3(tc.x, tc.y, tc.z);
            }
            if (mat.metallicRoughTex != 0) {
                float4 mr = tex2D<float4>(mat.metallicRoughTex, texUV.x, texUV.y);
                mat.roughness = mat.roughness * mr.y;
                mat.metallic  = mat.metallic  * mr.z;
            }
            // SG remap (mirrors regular OptiX raygen).
            if (mat.useSpecularGlossiness) {
                if (mat.useFBXCustomPacking && mat.specularGlossTex != 0) {
                    float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
                    float B = clampf(sg.z, 0.0f, 1.0f);
                    float G = clampf(sg.y, 0.0f, 1.0f);
                    albedo = mat.specularColor;
                    mat.metallic = B; mat.roughness = G;
                } else if (mat.useFBXUEPacking && mat.specularGlossTex != 0) {
                    float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
                    float G = clampf(sg.y, 0.0f, 1.0f);
                    float B = clampf(sg.z, 0.0f, 1.0f);
                    mat.metallic = B; mat.roughness = 1.0f - G;
                } else {
                    float3 specRGB = mat.specularColor;
                    float  alphaG  = 1.0f;
                    if (mat.specularGlossTex != 0) {
                        float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
                        specRGB = mat.specularColor * make_float3(sg.x, sg.y, sg.z);
                        alphaG  = sg.w;
                    }
                    float specLum = 0.2126f * specRGB.x + 0.7152f * specRGB.y + 0.0722f * specRGB.z;
                    specLum = clampf(specLum, 0.0f, 1.0f);
                    float specStrength = sqrtf(specLum);
                    float F0_target = 0.04f + 0.56f * specStrength;
                    mat.metallic = clampf((F0_target - 0.04f) / 0.96f, 0.0f, 1.0f);
                    float gloss = mat.specularGlossAlphaIsGlossiness
                                    ? (mat.glossiness * alphaG) : specStrength;
                    mat.roughness = 1.0f - clampf(gloss, 0.0f, 0.95f);
                }
            }
            mat.roughness = fmaxf(mat.roughness, 0.045f);
            mat.metallic  = clampf(mat.metallic, 0.0f, 1.0f);

            float3 emissiveColor = mat.emission;
            if (mat.emissiveTex != 0) {
                float4 et = tex2D<float4>(mat.emissiveTex, texUV.x, texUV.y);
                emissiveColor = make_float3(et.x, et.y, et.z);
            }

            float3 N = hit.shadingNormal;
            if (scene.d_normals) {
                float3 n0 = scene.d_normals[i0];
                float3 n1 = scene.d_normals[i1];
                float3 n2 = scene.d_normals[i2];
                N = normalize(n0 * baryW + n1 * baryU + n2 * baryV);
            }
            if (mat.transmission <= 0.0f && mat.normalTex != 0 && scene.d_tangents) {
                float4 t0 = scene.d_tangents[i0];
                float4 t1 = scene.d_tangents[i1];
                float4 t2 = scene.d_tangents[i2];
                float4 tangent = t0 * baryW + t1 * baryU + t2 * baryV;
                N = applyNormalMap(N, tangent, mat.normalTex, texUV);
            }
            if (mat.transmission <= 0.0f) {
                if (dot(N, ray.direction) > 0) N = -N;
            }

            // Primary-hit g-buffer capture + bucket classification.
            if (firstBounce) {
                primaryAlbedo    = albedo;
                primaryNormal    = N;
                primaryRoughness = mat.roughness;
                primaryHitPos    = hit.position;
                primaryRayDir    = ray.direction;
                primaryMetallic  = mat.metallic;
                primaryViewZ     = nrd_helpers::computeViewZ(hit.position, camera.position, camera.forward);
                primaryMvPx      = nrd_helpers::computeMotionVectorPx(
                    hit.position, camera.viewProjMatrix, camera.prevViewProjMatrix,
                    params.width, params.height);
                {
                    float3 ndc = mat4_transformPoint(camera.viewProjMatrix, hit.position);
                    primaryNdcZ = clampf(ndc.z * 0.5f + 0.5f, 0.0f, 1.0f);
                }

                float3 V = -ray.direction;
                float specProb = materialSpecProb(mat, N, V, albedo);
                pickedBucket = (pcg32_float(rng) < specProb) ? 1 : 0;
                float pickedP = (pickedBucket == 1) ? specProb : (1.0f - specProb);
                throughput = throughput * (1.0f / fmaxf(pickedP, 1e-4f));

                haveGbuffer = true;
                firstBounce = false;
                primaryLobeOverride = true;
            }

            // Glass / transmissive (delta BSDF — never affects bucket classification).
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
                lastBounceDelta = true;
                prevSurfacePos = hit.position; prevBsdfPdf = 1.0f; havePrevSurface = true;
                if (bounce >= 6 && pcg32_float(rng) > 0.9f) break;
                continue;
            }

            // First indirect surface distance (drives RELAX spatial filter radius).
            if (!bucketHitDistSet && bounce == 1) {
                bucketHitDist = hit.t;
                bucketHitDistSet = true;
            }

            bool isEmissive = mat.emissionStrength > 0.0f &&
                (emissiveColor.x > 0.0f || emissiveColor.y > 0.0f || emissiveColor.z > 0.0f);
            if (isEmissive) {
                float3 Le = emissiveColor * mat.emissionStrength;
                float weight = 1.0f;
                if (bounce > 0 && havePrevSurface && !lastBounceDelta && scene.d_triangleAreaLightIndex) {
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
                    emissiveContrib = Le * weight;   // Primary emissive — separate image.
                } else {
                    pathRadiance += clampFirefly_split(throughput * Le * weight, 10.0f);
                }
                if (mat.emissiveTex == 0) break;
            }

            // NEE area lights.  ReSTIR DI replaces the per-frame CDF pick at
            // the primary hit (s==0, bounce==0) with the resampled
            // reservoir's selected sample. The lobe-override still applies
            // — ReSTIR DI feeds the chosen bucket only, multiplied by
            // 1/pickedP via `throughput`. Estimator: f * W (no MIS).
            bool restirActive = primaryLobeOverride && (s == 0) &&
                                (scene.restirEnabled != 0) &&
                                (scene.d_restirReservoirs != nullptr);
            if (scene.d_areaLights && scene.areaLightCount > 0 &&
                scene.d_areaLightCDF && scene.areaLightTotalWeight > 0.0f)
            {
                uint32_t li;
                float b0, b1, b2;
                float restirW = 0.0f;
                bool restirSkip = false;
                if (restirActive) {
                    const ReSTIRReservoir* res =
                        reinterpret_cast<const ReSTIRReservoir*>(scene.d_restirReservoirs);
                    ReSTIRReservoir r = res[pixelIdx];
                    if (r.lightIndex == 0xFFFFFFFFu || r.W <= 0.0f || r.pHat <= 0.0f) {
                        restirSkip = true;
                    } else {
                        li = r.lightIndex;
                        b1 = r.baryB1;
                        b2 = r.baryB2;
                        b0 = 1.0f - b1 - b2;
                        restirW = r.W;
                    }
                } else {
                    li = sampleAreaLightIndex(scene.d_areaLightCDF, scene.areaLightCount,
                                              pcg32_float(rng));
                    float r1 = pcg32_float(rng), r2 = pcg32_float(rng);
                    float su = sqrtf(r1);
                    b0 = 1.0f - su;
                    b1 = su * (1.0f - r2);
                    b2 = su * r2;
                }
                if (!restirSkip) {
                GPUAreaLight light = scene.d_areaLights[li];
                float3 lp = light.v0 * b0
                          + (light.v0 + light.e1) * b1
                          + (light.v0 + light.e2) * b2;
                float3 toL = lp - hit.position;
                float d2 = fmaxf(dot(toL, toL), 1e-6f);
                float d  = sqrtf(d2);
                float3 Ld = toL * (1.0f / d);
                float NdotL = fmaxf(dot(N, Ld), 0.0f);
                float lNdot = fmaxf(dot(light.normal, -Ld), 0.0f);
                if (NdotL > 0.0f && lNdot > 0.0f) {
                    float3 shadowOrigin = hit.position + N * 0.001f;
                    float shadowTmax = fmaxf(d - 0.002f, 0.001f);
                    float3 st = traceShadowRay(handle, shadowOrigin, Ld, 0.001f, shadowTmax);
                    float slum = 0.2126f*st.x + 0.7152f*st.y + 0.0722f*st.z;
                    if (slum > 1e-6f) {
                        float3 V = -ray.direction;
                        float3 brdf;
                        if (primaryLobeOverride) {
                            brdf = (pickedBucket == 0)
                                ? materialDiffuseLobe_split(mat, N, V, Ld, albedo)
                                : materialSpecularLobe_split(mat, N, V, Ld, albedo);
                        } else {
                            brdf = materialBsdfEvaluate(mat, N, V, Ld, albedo);
                        }
                        float3 Le = sampleAreaLightLe(light, b0, b1, b2);

                        if (restirActive) {
                            float geom = lNdot / d2;
                            float3 neeContrib = throughput * st * brdf * Le *
                                                (NdotL * geom) * restirW;
                            pathRadiance += clampFirefly_split(neeContrib, 10.0f);
                        } else {
                            float pTri = light.weight / scene.areaLightTotalWeight;
                            float pArea = pTri / fmaxf(light.area, 1e-7f);
                            float pdfOmega = pArea * d2 / fmaxf(lNdot, 1e-7f);
                            float pdfBs;
                            if (primaryLobeOverride) {
                                pdfBs = (pickedBucket == 0)
                                    ? bsdfDiffusePdf(NdotL)
                                    : bsdfSpecularPdf(N, V, Ld, mat.roughness);
                            } else {
                                float spP = materialSpecProb(mat, N, V, albedo);
                                pdfBs = materialMixturePdf(mat, N, V, Ld, spP);
                            }
                            float w = powerHeuristic(pdfOmega, pdfBs);
                            float3 neeContrib = throughput * st * brdf * Le *
                                                (NdotL / fmaxf(pdfOmega, 1e-7f)) * w;
                            pathRadiance += clampFirefly_split(neeContrib, 10.0f);
                        }
                    }
                }
                } // end !restirSkip
            }

            // Point lights: only sampled when no area lights (matches the
            // non-split CUDA / OptiX raygens). Mixed scenes prefer area-light
            // emissive geometry; explicit point lights are a fallback for
            // scenes that have no area lights at all.
            else if (scene.d_pointLights && scene.pointLightCount > 0) {
                float3 V = -ray.direction;
                float3 direct = make_float3(0,0,0);
                for (uint32_t li = 0; li < scene.pointLightCount; li++) {
                    GPUPointLight light = scene.d_pointLights[li];
                    float3 toL = light.position - hit.position;
                    float d2 = fmaxf(dot(toL, toL), 1e-6f);
                    float d  = sqrtf(d2);
                    float3 Ld = toL * (1.0f / d);
                    float NdotL = fmaxf(dot(N, Ld), 0.0f);
                    if (NdotL <= 0.0f) continue;
                    float3 shadowOrigin = hit.position + N * 0.001f;
                    float shadowTmax = fmaxf(d - 0.002f, 0.001f);
                    float3 st = traceShadowRay(handle, shadowOrigin, Ld, 0.001f, shadowTmax);
                    float slum = 0.2126f*st.x + 0.7152f*st.y + 0.0722f*st.z;
                    if (slum < 1e-6f) continue;
                    float attenDen = light.constantAttenuation
                                   + light.linearAttenuation  * d
                                   + light.quadraticAttenuation * d2;
                    float atten = 1.0f / fmaxf(attenDen, 1e-4f);
                    float3 Li = light.color * (light.intensity * atten);
                    float3 brdf;
                    if (primaryLobeOverride) {
                        brdf = (pickedBucket == 0)
                            ? materialDiffuseLobe_split(mat, N, V, Ld, albedo)
                            : materialSpecularLobe_split(mat, N, V, Ld, albedo);
                    } else {
                        brdf = materialBsdfEvaluate(mat, N, V, Ld, albedo);
                    }
                    direct += clampFirefly_split(brdf * st * Li * NdotL, 10.0f);
                }
                pathRadiance += throughput * direct;
            }

            if (scene.d_directionalLights && scene.directionalLightCount > 0) {
                float3 V = -ray.direction;
                float3 direct = make_float3(0,0,0);
                for (uint32_t li = 0; li < scene.directionalLightCount; li++) {
                    GPUDirectionalLight light = scene.d_directionalLights[li];
                    float3 Ld = light.direction;
                    float NdotL = fmaxf(dot(N, Ld), 0.0f);
                    if (NdotL <= 0.0f) continue;

                    float3 st = traceShadowRay(handle, hit.position + N * 0.001f, Ld, 0.001f, 1e30f);
                    float slum = 0.2126f*st.x + 0.7152f*st.y + 0.0722f*st.z;
                    if (slum < 1e-6f) continue;

                    float3 brdf;
                    if (primaryLobeOverride) {
                        brdf = (pickedBucket == 0)
                            ? materialDiffuseLobe_split(mat, N, V, Ld, albedo)
                            : materialSpecularLobe_split(mat, N, V, Ld, albedo);
                    } else {
                        brdf = materialBsdfEvaluate(mat, N, V, Ld, albedo);
                    }
                    direct += clampFirefly_split(brdf * st * light.color * NdotL, 10.0f);
                }
                pathRadiance += throughput * direct;
            }

            // ReSTIR PT / GI consumption at the primary hit on sample s==0.
            // PT takes precedence; both branches add the pre-computed indirect
            // estimate and skip continuation bounces.
            //
            // CRITICAL: do NOT multiply by `throughput`. throughput == 1/pickedP
            // from the bucket override; the bucket routing redistributes
            // pathRadiance to the picked demod bucket already. Multiplying
            // by throughput would scale indirect by 1/pickedP, and the
            // demod-composite recovery becomes 2*indirect (double-counted).
            if (primaryLobeOverride && s == 0) {
                if (scene.restirPTEnabled != 0 && scene.d_restirPTIndirect != nullptr) {
                    float3 indirect = scene.d_restirPTIndirect[pixelIdx];
                    pathRadiance += indirect;
                    break;
                }
                if (scene.restirGIEnabled != 0 && scene.d_restirGIIndirect != nullptr) {
                    float3 indirect = scene.d_restirGIIndirect[pixelIdx];
                    pathRadiance += indirect;
                    break;
                }
            }

            // BSDF sampling for next bounce. Forced lobe at primary hit.
            float3 V = -ray.direction;
            float specProb = materialSpecProb(mat, N, V, albedo);
            float3 newDir;
            bool sampleSpecular;
            if (primaryLobeOverride) {
                sampleSpecular = (pickedBucket == 1);
            } else {
                sampleSpecular = (pcg32_float(rng) < specProb);
            }
            if (sampleSpecular) {
                float a = mat.roughness * mat.roughness;
                float u1 = pcg32_float(rng), u2 = pcg32_float(rng);
                float cosT = sqrtf((1.0f - u1) / (1.0f + (a*a - 1.0f)*u1 + 1e-7f));
                float sinT = sqrtf(fmaxf(0.0f, 1.0f - cosT*cosT));
                float phi  = 2.0f * M_PI_F * u2;
                float3 lH  = make_float3(sinT*cosf(phi), cosT, sinT*sinf(phi));
                float3 T, B; buildONB(N, T, B);
                float3 H = localToWorld(lH, T, N, B);
                newDir = normalize(ray.direction - H * (2.0f * dot(ray.direction, H)));
                lastBounceDelta = false;
            } else {
                float u1 = pcg32_float(rng), u2 = pcg32_float(rng);
                float dummy;
                float3 lD = sampleCosineHemisphere(u1, u2, dummy);
                float3 T, B; buildONB(N, T, B);
                newDir = localToWorld(lD, T, N, B);
                lastBounceDelta = false;
            }
            float NdotLn = dot(N, newDir);
            if (NdotLn < 1e-6f) break;
            float pdf;
            float3 brdf;
            if (primaryLobeOverride) {
                if (pickedBucket == 0) {
                    pdf  = bsdfDiffusePdf(NdotLn);
                    brdf = materialDiffuseLobe_split(mat, N, V, newDir, albedo);
                } else {
                    pdf  = bsdfSpecularPdf(N, V, newDir, mat.roughness);
                    brdf = materialSpecularLobe_split(mat, N, V, newDir, albedo);
                }
            } else {
                pdf  = materialMixturePdf(mat, N, V, newDir, specProb);
                brdf = materialBsdfEvaluate(mat, N, V, newDir, albedo);
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
            ray.origin    = hit.position + N * 0.001f;
            ray.direction = newDir;
            ray.tmin      = 0.001f;
            ray.tmax      = 1e30f;
        } // end bounce loop

        // Sanitize + clamp per channel (matches CUDA split kernel).
        if (isnan(pathRadiance.x) || isnan(pathRadiance.y) || isnan(pathRadiance.z) ||
            isinf(pathRadiance.x) || isinf(pathRadiance.y) || isinf(pathRadiance.z)) {
            pathRadiance = make_float3(0,0,0);
        }
        pathRadiance.x = fminf(fmaxf(pathRadiance.x, 0.0f), 15.0f);
        pathRadiance.y = fminf(fmaxf(pathRadiance.y, 0.0f), 15.0f);
        pathRadiance.z = fminf(fmaxf(pathRadiance.z, 0.0f), 15.0f);

        // Demodulate by albedo (NRD wants irradiance — composite remultiplies).
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

        demodDiffSum = demodDiffSum + demodDiff;
        demodSpecSum = demodSpecSum + demodSpec;
        emissiveSum  = emissiveSum  + emissiveContrib;
        if (haveGbuffer && bucketHitDistSet) {
            if (pickedBucket == 0) { diffHitSum += bucketHitDist; diffHitCount++; }
            else                    { specHitSum += bucketHitDist; specHitCount++; }
            anyHitSum += bucketHitDist; anyHitCount++;
        }
        // DLSS-RR noisy combined color: pathRadiance already incorporates
        // 1/pickedP, so its expected value over both buckets is the un-modulated
        // primary-hit radiance.
        noisyColorSum = noisyColorSum + pathRadiance + emissiveContrib;
        if (!gbufferWritten && haveGbuffer) {
            outPrimaryAlbedo    = primaryAlbedo;
            outPrimaryNormal    = primaryNormal;
            outPrimaryRoughness = primaryRoughness;
            outPrimaryViewZ     = primaryViewZ;
            outPrimaryMvPx      = primaryMvPx;
            outPrimaryNdcZ      = primaryNdcZ;
            outPrimaryHitPos    = primaryHitPos;
            outPrimaryRayDir    = primaryRayDir;
            outPrimaryMetallic  = primaryMetallic;
            gbufferWritten = true;
        }
    } // end spp loop

    // Average per-pixel radiance across spp.
    float invSpp = 1.0f / (float)samplesPerPixel;
    float3 demodDiffAvg = demodDiffSum * invSpp;
    float3 demodSpecAvg = demodSpecSum * invSpp;
    float3 emissiveAvg  = emissiveSum  * invSpp;
    float3 noisyColorAvg = noisyColorSum * invSpp;
    float diffHitAvg = diffHitCount > 0 ? (diffHitSum / (float)diffHitCount) : 0.0f;
    float specHitAvg = specHitCount > 0 ? (specHitSum / (float)specHitCount) : 0.0f;
    float anyHitAvg  = anyHitCount  > 0 ? (anyHitSum  / (float)anyHitCount)  : 0.0f;

    // DLSS-RR specular albedo: F0 = lerp(0.04, primaryAlbedo, metallic) per
    // §3.4.2 + Appendix EnvBRDFApprox2. Metallic is now preserved through
    // outPrimaryMetallic. NoV uses the actual primary ray direction (rather
    // than camera.forward) for sub-pixel-stable, jittered-ray-correct values.
    float3 specF0_RR = lerp(make_float3(0.04f, 0.04f, 0.04f),
                            outPrimaryAlbedo, outPrimaryMetallic);
    float NoV_RR = fmaxf(-dot(outPrimaryRayDir, outPrimaryNormal), 0.0f);
    float3 specAlbedoAvg = envBRDFApprox2_split(
        specF0_RR, outPrimaryRoughness * outPrimaryRoughness, NoV_RR);
    if (!gbufferWritten) {
        specAlbedoAvg = make_float3(0.5f, 0.5f, 0.5f);
    }

    // DLSS-RR specular hit distance (§3.4.9): explicit mirror-ray trace from
    // the primary hit. See PathTraceKernelSplit.cu for the full rationale.
    // The previous `anyHitAvg` was a per-bucket-roll average that flickers
    // frame-to-frame, producing the surface shimmer measured at 2.19 (vs
    // DLSS-SR 1.45). One mirror trace per pixel is deterministic.
    float rrSpecHitT = 0.0f;
    if (gbufferWritten) {
        float3 rd = outPrimaryRayDir;
        float3 N  = outPrimaryNormal;
        float3 mirrorDir = normalize(rd - N * (2.0f * dot(rd, N)));
        float3 mOrigin   = outPrimaryHitPos + N * 0.001f;
        RadiancePayload mrp = traceRadianceRay(
            handle, mOrigin, mirrorDir, 0.001f, 1e30f);
        rrSpecHitT = mrp.hit ? mrp.tHit : 1.0e4f;
    }
    if (isnan(rrSpecHitT) || isinf(rrSpecHitT) || rrSpecHitT < 0.0f) rrSpecHitT = 0.0f;

    float4 diffTexel = nrd_helpers::packRadianceHitDist(demodDiffAvg, diffHitAvg);
    float4 specTexel = nrd_helpers::packRadianceHitDist(demodSpecAvg, specHitAvg);
    float4 normTexel = nrd_helpers::packNormalRoughness(outPrimaryNormal, outPrimaryRoughness);
    float4 albTexel  = make_float4(
        fminf(fmaxf(outPrimaryAlbedo.x, 0.0f), 1.0f),
        fminf(fmaxf(outPrimaryAlbedo.y, 0.0f), 1.0f),
        fminf(fmaxf(outPrimaryAlbedo.z, 0.0f), 1.0f),
        1.0f);
    float4 emTexel = make_float4(emissiveAvg.x, emissiveAvg.y, emissiveAvg.z, 1.0f);

    if (params.splitDiffuseRadianceHitDist) {
        ushort4 p = packHalf4_split(diffTexel);
        surf2Dwrite<ushort4>(p, params.splitDiffuseRadianceHitDist, x * 8, y);
    }
    if (params.splitSpecularRadianceHitDist) {
        ushort4 p = packHalf4_split(specTexel);
        surf2Dwrite<ushort4>(p, params.splitSpecularRadianceHitDist, x * 8, y);
    }
    if (params.splitNormalRoughness) {
        uint32_t packed = packRGBA8(normTexel.x, normTexel.y, normTexel.z, normTexel.w);
        surf2Dwrite<uint32_t>(packed, params.splitNormalRoughness, x * 4, y);
    }
    if (params.splitViewZ)
        surf2Dwrite<float>(outPrimaryViewZ, params.splitViewZ, x * 4, y);
    if (params.splitNdcDepth)
        surf2Dwrite<float>(outPrimaryNdcZ, params.splitNdcDepth, x * 4, y);
    if (params.splitMotionVectors) {
        ushort2 packed = packHalf2_split(outPrimaryMvPx.x, outPrimaryMvPx.y);
        surf2Dwrite<ushort2>(packed, params.splitMotionVectors, x * 4, y);
    }
    if (params.splitAlbedo) {
        uint32_t packed = packRGBA8(albTexel.x, albTexel.y, albTexel.z, 1.0f);
        surf2Dwrite<uint32_t>(packed, params.splitAlbedo, x * 4, y);
    }
    if (params.splitEmissive) {
        ushort4 p = packHalf4_split(emTexel);
        surf2Dwrite<ushort4>(p, params.splitEmissive, x * 8, y);
    }

    // ── DLSS-RR specific surfaces (only written in Mode::DLSSRR) ──
    if (params.splitHdrColor) {
        float3 c = noisyColorAvg;
        if (isnan(c.x) || isnan(c.y) || isnan(c.z) ||
            isinf(c.x) || isinf(c.y) || isinf(c.z)) c = make_float3(0,0,0);
        c.x = fminf(fmaxf(c.x, 0.0f), 30.0f);
        c.y = fminf(fmaxf(c.y, 0.0f), 30.0f);
        c.z = fminf(fmaxf(c.z, 0.0f), 30.0f);
        ushort4 p = packHalf4_split(make_float4(c.x, c.y, c.z, 1.0f));
        surf2Dwrite<ushort4>(p, params.splitHdrColor, x * 8, y);
    }
    if (params.splitWorldNormalRoughness) {
        ushort4 p = packHalf4_split(make_float4(
            outPrimaryNormal.x, outPrimaryNormal.y, outPrimaryNormal.z,
            outPrimaryRoughness));
        surf2Dwrite<ushort4>(p, params.splitWorldNormalRoughness, x * 8, y);
    }
    if (params.splitSpecAlbedo) {
        ushort4 p = packHalf4_split(make_float4(
            fminf(fmaxf(specAlbedoAvg.x, 0.0f), 4.0f),
            fminf(fmaxf(specAlbedoAvg.y, 0.0f), 4.0f),
            fminf(fmaxf(specAlbedoAvg.z, 0.0f), 4.0f),
            1.0f));
        surf2Dwrite<ushort4>(p, params.splitSpecAlbedo, x * 8, y);
    }
    if (params.splitSpecHitT) {
        // §3.4.9: world-space distance, primary surface to spec-reflected hit.
        surf2Dwrite<float>(rrSpecHitT, params.splitSpecHitT, x * 4, y);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// ReSTIR DI initial-candidates raygen (OptiX path).
//
// Mirrors `kReSTIR_InitCandidates` from render/ReSTIR.cu but traces the
// primary ray against the OptiX GAS via `traceRadianceRay()` instead of the
// CUDA SAH BVH. Output layout matches exactly — the CUDA temporal / spatial
// passes consume `params.restirReservoirsCurr` unchanged.
//
// All RIS helpers (target pdf, reservoir streaming) come from
// render/ReSTIRDevice.cuh so this program is a thin harness around the
// closest-hit + material-resolve logic already proven in the megakernel.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__ void __raygen__restir_init_candidates()
{
    uint3 idx = optixGetLaunchIndex();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    const DeviceSceneData& scene  = params.scene;
    const CameraParams&    camera = params.camera;
    OptixTraversableHandle handle = params.handle;

    ReSTIRReservoir r; restir_reservoirReset(r);
    ReSTIRSurface   surf{};
    surf.valid = 0.0f;

    // Dedicated RNG stream (matches the CUDA kernel's seeding salt).
    // Mix camera.frameIndex so the canonical sample changes every frame
    // even when sampleIndex is reset to 0 by camera motion.
    uint32_t seedSalt = params.sampleIndex + camera.frameIndex * 0x9E3779B9u;
    uint32_t rng = pcg32_seed(pixelIdx * 0xA1B2C3D4u + seedSalt,
                              seedSalt * 0xDEADBEEFu + 1u);

    // Primary ray — same jitter as the main kernel so the reservoir lines up
    // with the shading point that will consume it.
    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;
    Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

    RadiancePayload rp = traceRadianceRay(
        handle, ray.origin, ray.direction, ray.tmin, ray.tmax);

    bool eligible = (rp.hit != 0) && scene.d_areaLights &&
                    scene.areaLightCount > 0 && scene.d_lightBVHNodes;

    if (eligible) {
        uint32_t triIdx = rp.primIdx;
        uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
        uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
        uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
        float baryU = rp.baryU;
        float baryV = rp.baryV;
        float baryW = 1.0f - baryU - baryV;
        float3 v0 = scene.d_positions[i0];
        float3 v1 = scene.d_positions[i1];
        float3 v2 = scene.d_positions[i2];
        float3 hitPos = v0 * baryW + v1 * baryU + v2 * baryV;

        int matIdx = scene.d_materialIndices ? scene.d_materialIndices[triIdx] : -1;
        if (matIdx < 0 || (uint32_t)matIdx >= scene.materialCount) {
            eligible = false;
        }
        if (eligible) {
            GPUMaterial mat = scene.d_materials[matIdx];

            // Shading normal — interpolate vertex normals, flip to face the ray.
            float3 N;
            if (scene.d_normals) {
                N = normalize(scene.d_normals[i0] * baryW +
                              scene.d_normals[i1] * baryU +
                              scene.d_normals[i2] * baryV);
            } else {
                N = normalize(cross(v1 - v0, v2 - v0));
            }
            if (dot(N, ray.direction) > 0.0f) N = -N;

            // Texture UV + albedo / metallic / roughness resolve.
            float2 uv = make_float2(0.0f, 0.0f);
            if (scene.d_uvs) {
                uv = scene.d_uvs[i0] * baryW +
                     scene.d_uvs[i1] * baryU +
                     scene.d_uvs[i2] * baryV;
            }
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

            surf.position    = hitPos;
            surf.normal      = N;
            surf.albedo      = albedo;
            // Floor roughness so pHat's GGX term doesn't spike near zero —
            // keeps RIS well-conditioned on near-mirror surfaces. Matches
            // the CUDA kernel.
            surf.roughness   = fmaxf(mat.roughness, 0.04f);
            surf.metallic    = mat.metallic;
            surf.pureDiffuse = mat.pureDiffuse ? 1u : 0u;
            surf.viewDir     = -ray.direction;
            surf.valid       = 1.0f;

            // Cached specProb for downstream passes.
            {
                float NdotV = fmaxf(dot(surf.normal, surf.viewDir), 0.0f);
                float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), surf.albedo, surf.metallic);
                float t = 1.0f - fminf(fmaxf(NdotV, 0.0f), 1.0f);
                float t5 = t*t*t*t*t;
                float3 F = F0 + (make_float3(1,1,1) - F0) * t5;
                float specW = restirLuminance(F);
                float3 kd = (make_float3(1,1,1) - F) * (1.0f - surf.metallic);
                float diffW = restirLuminance(kd * surf.albedo);
                float p = specW / fmaxf(specW + diffW, 1e-7f);
                surf.specProb = fminf(fmaxf(p, 0.1f), 0.9f);
            }

            float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hitPos);
            surf.prevPixel = make_float2(
                (clipPrev.x + 1.0f) * 0.5f * (float)params.width,
                (1.0f - clipPrev.y) * 0.5f * (float)params.height);

            // ── RIS: draw M candidates from the light BVH, stream them ──
            uint32_t M = params.restirNumCandidates;
            float wSum = 0.0f;
            for (uint32_t i = 0; i < M; i++) {
                float u = pcg32_float(rng);
                uint32_t slot = 0;
                float pSelect = 0.0f;
                if (!lightBVH_sample(scene.d_lightBVHNodes,
                                     scene.lightBVHRootIndex,
                                     surf.position, u, slot, pSelect) ||
                    !(pSelect > 0.0f)) {
                    // Parity with render/ReSTIR.cu kReSTIR_InitCandidates:
                    // count failed BVH descents toward M so finalize divides
                    // by the true number of candidates considered. Otherwise
                    // shading points where the BVH frequently rejects (above
                    // / behind a tight light cluster) get an inflated W.
                    r.M += 1.0f;
                    continue;
                }
                uint32_t lightIdx = scene.d_lightOrderedIndices[slot];
                GPUAreaLight light = scene.d_areaLights[lightIdx];

                float r1 = pcg32_float(rng);
                float r2 = pcg32_float(rng);
                float su = sqrtf(r1);
                float cb1 = su * (1.0f - r2);
                float cb2 = su * r2;

                float areaPdf = pSelect / fmaxf(light.area, 1e-7f);
                float pHat = restirEvalTargetPdf(surf, light, cb1, cb2);
                float wCand = (areaPdf > 0.0f) ? (pHat / areaPdf) : 0.0f;

                restir_reservoirUpdate(r, wSum, lightIdx, cb1, cb2, pHat,
                                       wCand, pcg32_float(rng));
            }
            restir_reservoirFinalize(r, wSum);
        }
    }

    if (params.restirReservoirsCurr) params.restirReservoirsCurr[pixelIdx] = r;
    if (params.restirSurfacesCurr)   params.restirSurfacesCurr[pixelIdx]   = surf;
}

// ─────────────────────────────────────────────────────────────────────────
// ReSTIR DI visibility-reuse raygen (OptiX path).
//
// Mirrors `kReSTIR_Visibility` from render/ReSTIR.cu but traces the shadow
// ray against the OptiX GAS instead of the CUDA SAH BVH. One ray per pixel
// toward the held sample's point on the light; zeros W on occlusion so the
// subsequent temporal / spatial reuse can't propagate occluded samples
// (Bitterli 2020 Alg. 5 lines 6-9).
//
// Reuses the existing shadow SBT (offset 1, miss 1) so we get the same
// glass-transparency handling as the main shading shadow ray.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__ void __raygen__restir_visibility()
{
    uint3 idx = optixGetLaunchIndex();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    if (!params.restirReservoirsCurr || !params.restirSurfacesCurr) return;
    ReSTIRReservoir r = params.restirReservoirsCurr[pixelIdx];
    if (r.lightIndex == 0xFFFFFFFFu || r.W <= 0.0f) return;

    ReSTIRSurface s = params.restirSurfacesCurr[pixelIdx];
    if (s.valid < 0.5f) return;

    const DeviceSceneData& scene = params.scene;
    GPUAreaLight light = scene.d_areaLights[r.lightIndex];
    float b0 = 1.0f - r.baryB1 - r.baryB2;
    float3 pOnLight = light.v0 * b0
                    + (light.v0 + light.e1) * r.baryB1
                    + (light.v0 + light.e2) * r.baryB2;

    float3 origin = s.position + s.normal * 1e-3f;
    float3 toL    = pOnLight - origin;
    float  dist   = sqrtf(fmaxf(dot(toL, toL), 1e-12f));
    float3 dir    = toL * (1.0f / dist);
    // Pull tmax in slightly so the light triangle itself doesn't register.
    float  tmax   = fmaxf(dist - 2e-3f, 1e-4f);

    float3 transmittance = traceShadowRay(
        params.handle, origin, dir, 1e-3f, tmax);
    // Treat fully-occluded (transmittance == 0) as a kill. Glass paths return
    // partial transmittance > 0; we keep those samples — they're what the
    // final-shading shadow ray will also accumulate against.
    float lum = restirLuminance(transmittance);
    if (lum <= 1e-6f) {
        r.W = 0.0f;
        params.restirReservoirsCurr[pixelIdx] = r;
    }
}

// ── ReSTIR GI raygen (lives in OptiXProgramsGI.inl to keep this file
// readable). Compiled into the same module as the rest of the OptiX
// programs so it sees `params`, `traceRadianceRay`, etc.
#include "backend/OptiXProgramsGI.inl"

// ── ReSTIR PT raygen — multi-bounce postfix random walk past the
// reconnection vertex. Same idea as the GI raygen but with a longer path.
#include "backend/OptiXProgramsPT.inl"
