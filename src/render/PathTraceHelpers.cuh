#pragma once
// Shared __device__ helpers previously defined inline in PathTraceKernel.cu.
// Extracted so the split-output kernel can reuse them without relying on
// separable device compilation. Contents are byte-identical to the originals;
// PathTraceKernel.cu now includes this header to pick up the same definitions.

#include "core/Math.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/RayTypes.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Sampling.h"

#ifndef M_PI_F
#  define M_PI_F 3.14159265358979323846f
#endif

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
