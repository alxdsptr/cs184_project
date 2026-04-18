// ── OptiX device programs ─────────────────────────────────────
// Raygen: outer bounce loop + shading (ported from PathTraceKernel.cu).
// Closest-hit: records primitive index + barycentrics into payload.
// Miss (radiance): marks "no hit".
// Any-hit (shadow): accumulates glass transmittance, ignores or terminates.
// Miss (shadow): no-op (transmittance slots carry final value).

#include <optix.h>

#include "backend/OptiXLaunchParams.h"
#include "core/Math.h"
#include "core/Halton.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "gpu/Sampling.h"
#include "gpu/BRDF.h"
#include "gpu/RayTypes.h"

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

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
        uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                                  params.sampleIndex * 0x85EBCA6Bu + s);

        float jx = pcg32_float(rng) - 0.5f;
        float jy = pcg32_float(rng) - 0.5f;
        jx += camera.jitterOffset.x;
        jy += camera.jitterOffset.y;

        Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

        float3 throughput = make_float3(1, 1, 1);
        float3 radiance   = make_float3(0, 0, 0);
        bool firstBounce  = true;
        bool lastBounceSpecular = false;
        bool havePrevSurface = false;
        float3 prevSurfacePos = make_float3(0, 0, 0);
        float prevBsdfPdf = 1.0f;

        for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
            RadiancePayload rp = traceRadianceRay(
                handle, ray.origin, ray.direction, ray.tmin, ray.tmax);

            bool didHit = (rp.hit != 0);

            if (!didHit) {
                if (enableEnvironment) {
                    float3 envColor = sampleEnvironment(ray.direction, scene.envMapTex);
                    float envLum = 0.2126f * envColor.x + 0.7152f * envColor.y + 0.0722f * envColor.z;
                    float envClamp = 100.0f;
                    if (envLum > envClamp) envColor = envColor * (envClamp / envLum);
                    radiance += throughput * envColor;
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
            if (mat.transmission <= 0.0f) {
                if (dot(N, ray.direction) > 0) N = -N;
            }

            if (firstBounce) {
                if (!gbufferWritten) {
                    if (params.aux.d_linearDepth)
                        params.aux.d_linearDepth[pixelIdx] = dot(hit.position - camera.position, camera.forward);
                    if (params.aux.d_albedo)
                        params.aux.d_albedo[pixelIdx] = albedo;
                    if (params.aux.d_normal)
                        params.aux.d_normal[pixelIdx] = N;
                    if (params.aux.d_motionVectors) {
                        float3 clipCurr = mat4_transformPoint(camera.viewProjMatrix, hit.position);
                        float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hit.position);
                        float2 screenCurr = make_float2((clipCurr.x + 1.0f) * 0.5f * params.width,
                                                         (1.0f - clipCurr.y) * 0.5f * params.height);
                        float2 screenPrev = make_float2((clipPrev.x + 1.0f) * 0.5f * params.width,
                                                         (1.0f - clipPrev.y) * 0.5f * params.height);
                        params.aux.d_motionVectors[pixelIdx] = screenCurr - screenPrev;
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

                lastBounceSpecular = true;
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
                if (bounce > 0 && havePrevSurface && !lastBounceSpecular && scene.d_triangleAreaLightIndex) {
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
                uint32_t lightIndex = sampleAreaLightIndex(
                    scene.d_areaLightCDF, scene.areaLightCount,
                    pcg32_float(rng));
                GPUAreaLight light = scene.d_areaLights[lightIndex];

                float r1 = pcg32_float(rng);
                float r2 = pcg32_float(rng);
                float su = sqrtf(r1);
                float b0 = 1.0f - su;
                float b1 = su * (1.0f - r2);
                float b2 = su * r2;

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
                        float pTri = light.weight / scene.areaLightTotalWeight;
                        float pArea = pTri / fmaxf(light.area, 1e-7f);
                        float pdfOmega = pArea * dist2 / fmaxf(lightNdot, 1e-7f);

                        float3 V = -ray.direction;
                        float3 brdf = bsdfEvaluate(N, V, Ld, albedo, mat.roughness, mat.metallic);
                        float neeSpecProb = computeSpecProb(N, V, albedo, mat.metallic);
                        float pdfBsdf = bsdfMixturePdf(N, V, Ld, mat.roughness, neeSpecProb);
                        float weight = powerHeuristic(pdfOmega, pdfBsdf);

                        radiance += throughput * shadowTransmittance * brdf *
                                    light.emission *
                                    (NdotL / fmaxf(pdfOmega, 1e-7f)) * weight;
                    }
                }
            } else if (scene.d_pointLights && scene.pointLightCount > 0) {
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
                    float3 brdf = bsdfEvaluate(N, V, Ld, albedo, mat.roughness, mat.metallic);
                    direct += brdf * shadowTransmittancePL * Li * NdotL;
                }
                radiance += throughput * direct;
            }

            // BRDF sampling
            float3 V = -ray.direction;
            float specProb = computeSpecProb(N, V, albedo, mat.metallic);
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
                lastBounceSpecular = true;
            } else {
                float u1 = pcg32_float(rng);
                float u2 = pcg32_float(rng);
                float dummyPdf;
                float3 localDir = sampleCosineHemisphere(u1, u2, dummyPdf);
                float3 T, B;
                buildONB(N, T, B);
                newDir = localToWorld(localDir, T, N, B);
                lastBounceSpecular = false;
            }

            float NdotL_new = dot(N, newDir);
            if (NdotL_new < 1e-6f) break;

            float pdf = bsdfMixturePdf(N, V, newDir, mat.roughness, specProb);
            if (pdf < 1e-7f) break;

            float3 brdf = bsdfEvaluate(N, V, newDir, albedo, mat.roughness, mat.metallic);
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
    params.output[pixelIdx] = params.accum[pixelIdx] * invN;
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
