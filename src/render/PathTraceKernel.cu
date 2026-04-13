#include "render/PathTraceKernel.h"
#include "core/Math.h"
#include "core/Halton.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/RayTypes.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "gpu/Sampling.h"
#include "accel/BVH.h"
#include "util/CudaCheck.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ── Environment ──────────────────────────────────────────────
__device__ inline float3 sampleEnvironment(float3 dir) {
    // Constant sky color with slight gradient
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

    // Convert to [-1,1] NDC (y flipped for screen coords)
    float ndcX = 2.0f * u - 1.0f;
    float ndcY = 1.0f - 2.0f * v;

    // Scale by FOV and aspect
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
    return a2 / (M_PI_F * denom * denom + 1e-7f);
}

__device__ inline float3 fresnelSchlick_local(float cosTheta, float3 F0) {
    float t = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    return F0 + (make_float3(1,1,1) - F0) * t5;
}

__device__ inline float smithG1_local(float NdotX, float roughness) {
    float a = roughness * roughness;
    float k = a * 0.5f;
    return NdotX / (NdotX * (1.0f - k) + k + 1e-7f);
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
    if (NdotH <= 0.0f || VdotH <= 0.0f) {
        return 0.0f;
    }

    float a = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    float D_val = a2 / (M_PI_F * denom * denom + 1e-7f);
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
    const float3& N,
    const float3& V,
    const float3& L,
    float roughness,
    float specProb)
{
    float diffusePdf = bsdfDiffusePdf(dot(N, L));
    float specPdf = bsdfSpecularPdf(N, V, L, roughness);
    return specProb * specPdf + (1.0f - specProb) * diffusePdf;
}

__device__ inline float3 bsdfEvaluate(
    const float3& N,
    const float3& V,
    const float3& L,
    const float3& albedo,
    float roughness,
    float metallic)
{
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float NdotV = fmaxf(dot(N, V), 0.0f);
    if (NdotL <= 0.0f || NdotV <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float3 H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float LdotH = fmaxf(dot(L, H), 0.0f);

    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float3 F = fresnelSchlick_local(LdotH, F0);
    float D_val = ggxD_local(NdotH, roughness);
    float G_val = smithG1_local(NdotL, roughness) * smithG1_local(NdotV, roughness);

    float3 specular = F * (D_val * G_val / (4.0f * NdotL * NdotV + 1e-7f));
    float3 kd = (make_float3(1, 1, 1) - F) * (1.0f - metallic);
    float3 diffuse = kd * albedo * (1.0f / M_PI_F);
    return diffuse + specular;
}

__device__ inline uint32_t sampleAreaLightIndex(
    const float* cdf,
    uint32_t count,
    float target)
{
    uint32_t low = 0;
    uint32_t high = count;
    while (low < high) {
        uint32_t mid = (low + high) / 2;
        if (target <= cdf[mid]) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return (low >= count) ? (count - 1) : low;
}

// ── Path Trace Kernel ────────────────────────────────────────
__global__ void pathTraceKernel(
    DeviceSceneData scene,
    CameraParams    camera,
    float4*         d_accumBuffer,
    float4*         d_outputBuffer,
    AuxBufferPtrs   auxBuffers,
    uint32_t        width,
    uint32_t        height,
    uint32_t        sampleIndex,
    bool            enableEnvironment,
    uint32_t        maxBounces)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint32_t pixelIdx = y * width + x;

    // Per-pixel RNG
    uint32_t rng = pcg32_seed(pixelIdx, sampleIndex);

    // Sub-pixel jitter
    float jx = pcg32_float(rng) - 0.5f;
    float jy = pcg32_float(rng) - 0.5f;

    // Add Halton jitter for DLSS
    jx += camera.jitterOffset.x;
    jy += camera.jitterOffset.y;

    Ray ray = generateRay(x, y, width, height, camera, jx, jy);

    float3 throughput = make_float3(1, 1, 1);
    float3 radiance   = make_float3(0, 0, 0);
    bool firstBounce  = true;
    bool lastBounceSpecular = false;
    bool havePrevSurface = false;
    float3 prevSurfacePos = make_float3(0.0f, 0.0f, 0.0f);
    float prevBsdfPdf = 1.0f;

    for (uint32_t bounce = 0; bounce < maxBounces; bounce++) {
        HitRecord hit;
        hit.t = ray.tmax;

        bool didHit = false;
        if (scene.d_bvhNodes && scene.totalTriangles > 0) {
            didHit = bvh_closestHit(
                ray, scene.d_bvhNodes, scene.bvhRootIndex,
                scene.d_positions, scene.d_indices, scene.d_materialIndices,
                hit);
        }

        if (!didHit) {
            if (enableEnvironment) {
                radiance += throughput * sampleEnvironment(ray.direction);
            }
            break;
        }

        // Fetch material
        GPUMaterial mat;
        if (hit.materialIndex >= 0 && (uint32_t)hit.materialIndex < scene.materialCount)
            mat = scene.d_materials[hit.materialIndex];
        else {
            mat.albedo = make_float3(0.8f, 0.2f, 0.8f);
            mat.roughness = 0.5f;
            mat.metallic = 0.0f;
            mat.emission = make_float3(0,0,0);
            mat.emissionStrength = 0.0f;
        }

        // Clamp roughness to avoid singularities in GGX sampling/evaluation
        mat.roughness = fmaxf(mat.roughness, 0.045f);

        // Fetch vertex indices and barycentric coords for interpolation
        uint32_t triIdx = (uint32_t)hit.primitiveIndex;
        uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
        uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
        uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
        float baryU = hit.uv.x, baryV = hit.uv.y;
        float baryW = 1.0f - baryU - baryV;

        // Interpolate actual texture UVs from vertex data
        float2 texUV = make_float2(0.0f, 0.0f);
        if (scene.d_uvs) {
            float2 uv0 = scene.d_uvs[i0];
            float2 uv1 = scene.d_uvs[i1];
            float2 uv2 = scene.d_uvs[i2];
            texUV = uv0 * baryW + uv1 * baryU + uv2 * baryV;
        }

        // Sample albedo texture if available
        float3 albedo = mat.albedo;
        if (mat.albedoTex != 0) {
            float4 texColor = tex2D<float4>(mat.albedoTex, texUV.x, texUV.y);
            albedo = make_float3(texColor.x, texColor.y, texColor.z);
        }

        // Interpolate vertex normals if available
        float3 N = hit.shadingNormal;
        if (scene.d_normals) {
            float3 n0 = scene.d_normals[i0];
            float3 n1 = scene.d_normals[i1];
            float3 n2 = scene.d_normals[i2];
            N = normalize(n0 * baryW + n1 * baryU + n2 * baryV);
            if (dot(N, ray.direction) > 0) N = -N;
        }

        // Write aux buffers on first bounce
        if (firstBounce) {
            if (auxBuffers.d_linearDepth)
                auxBuffers.d_linearDepth[pixelIdx] = dot(hit.position - camera.position, camera.forward);
            if (auxBuffers.d_albedo)
                auxBuffers.d_albedo[pixelIdx] = albedo;
            if (auxBuffers.d_normal)
                auxBuffers.d_normal[pixelIdx] = N;
            // Motion vectors (simplified: only camera motion)
            if (auxBuffers.d_motionVectors) {
                float3 clipCurr = mat4_transformPoint(camera.viewProjMatrix, hit.position);
                float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hit.position);
                float2 screenCurr = make_float2((clipCurr.x + 1.0f) * 0.5f * width,
                                                 (1.0f - clipCurr.y) * 0.5f * height);
                float2 screenPrev = make_float2((clipPrev.x + 1.0f) * 0.5f * width,
                                                 (1.0f - clipPrev.y) * 0.5f * height);
                auxBuffers.d_motionVectors[pixelIdx] = screenCurr - screenPrev;
            }
            firstBounce = false;
        }

        bool isEmissive = mat.emissionStrength > 0.0f &&
                          (mat.emission.x > 0.0f || mat.emission.y > 0.0f || mat.emission.z > 0.0f);
        if (isEmissive) {
            float3 Le = mat.emission * mat.emissionStrength;
            float weight = 1.0f;

            if (bounce > 0 && havePrevSurface && !lastBounceSpecular && scene.d_triangleAreaLightIndex) {
                int areaLightIndex = scene.d_triangleAreaLightIndex[(uint32_t)hit.primitiveIndex];
                if (areaLightIndex >= 0 && scene.d_areaLights && scene.areaLightCount > 0) {
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
            break;
        }

        // Direct lighting from emissive triangle lights (next-event estimation).
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
                bool occluded = false;
                if (scene.d_bvhNodes && scene.totalTriangles > 0) {
                    Ray shadowRay;
                    shadowRay.origin = hit.position + N * 0.001f;
                    shadowRay.direction = Ld;
                    shadowRay.tmin = 0.001f;
                    shadowRay.tmax = fmaxf(dist - 0.002f, 0.001f);

                    HitRecord shadowHit;
                    shadowHit.t = shadowRay.tmax;
                    occluded = bvh_closestHit(
                        shadowRay, scene.d_bvhNodes, scene.bvhRootIndex,
                        scene.d_positions, scene.d_indices, scene.d_materialIndices,
                        shadowHit);
                }

                if (!occluded) {
                    float pTri = light.weight / scene.areaLightTotalWeight;
                    float pArea = pTri / fmaxf(light.area, 1e-7f);
                    float pdfOmega = pArea * dist2 / fmaxf(lightNdot, 1e-7f);

                    float3 V = -ray.direction;
                    float3 brdf = bsdfEvaluate(N, V, Ld, albedo, mat.roughness, mat.metallic);
                    float neeSpecProb = computeSpecProb(N, V, albedo, mat.metallic);
                    float pdfBsdf = bsdfMixturePdf(N, V, Ld, mat.roughness, neeSpecProb);
                    float weight = powerHeuristic(pdfOmega, pdfBsdf);

                    radiance += throughput * brdf * light.emission * (NdotL / fmaxf(pdfOmega, 1e-7f)) * weight;
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

                bool occluded = false;
                if (scene.d_bvhNodes && scene.totalTriangles > 0) {
                    Ray shadowRay;
                    shadowRay.origin = hit.position + N * 0.001f;
                    shadowRay.direction = Ld;
                    shadowRay.tmin = 0.001f;
                    shadowRay.tmax = fmaxf(dist - 0.002f, 0.001f);

                    HitRecord shadowHit;
                    shadowHit.t = shadowRay.tmax;
                    occluded = bvh_closestHit(
                        shadowRay, scene.d_bvhNodes, scene.bvhRootIndex,
                        scene.d_positions, scene.d_indices, scene.d_materialIndices,
                        shadowHit);
                }

                if (occluded) continue;

                float attenDen = light.constantAttenuation
                               + light.linearAttenuation * dist
                               + light.quadraticAttenuation * dist2;
                float attenuation = 1.0f / fmaxf(attenDen, 1e-4f);
                float3 Li = light.color * (light.intensity * attenuation);
                float3 brdf = bsdfEvaluate(N, V, Ld, albedo, mat.roughness, mat.metallic);

                direct += brdf * Li * NdotL;
            }

            radiance += throughput * direct;
        }

        // BRDF sampling: Fresnel-weighted blend between diffuse and specular
        float3 V = -ray.direction;
        float specProb = computeSpecProb(N, V, albedo, mat.metallic);

        float3 newDir;

        if (pcg32_float(rng) < specProb) {
            // GGX importance sampling
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
            // Cosine-weighted hemisphere sampling (diffuse)
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

        // Compute full mixture PDF for the sampled direction
        float pdf = bsdfMixturePdf(N, V, newDir, mat.roughness, specProb);
        if (pdf < 1e-7f) break;

        // Evaluate BRDF
        float3 brdf = bsdfEvaluate(N, V, newDir, albedo, mat.roughness, mat.metallic);

        throughput = throughput * brdf * (NdotL_new / (pdf + 1e-7f));

        prevSurfacePos = hit.position;
            prevBsdfPdf = pdf;
        havePrevSurface = true;

        // Russian roulette
        if (bounce >= 2) {
            float lum = 0.2126f * throughput.x + 0.7152f * throughput.y + 0.0722f * throughput.z;
            float p = fminf(fmaxf(lum, 0.05f), 0.95f);
            if (pcg32_float(rng) >= p) break;
            throughput = throughput * (1.0f / p);
        }

        // Next ray
        ray.origin    = hit.position + N * 0.001f;
        ray.direction = newDir;
        ray.tmin      = 0.001f;
        ray.tmax      = 1e30f;
    }

    // Clamp fireflies and reject NaN/inf before accumulation
    if (isnan(radiance.x) || isnan(radiance.y) || isnan(radiance.z) ||
        isinf(radiance.x) || isinf(radiance.y) || isinf(radiance.z)) {
        radiance = make_float3(0.0f, 0.0f, 0.0f);
    }
    float luminance = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
    float clampMax = 1000.0f;
    if (luminance > clampMax) {
        float scale = clampMax / luminance;
        radiance = radiance * scale;
    }

    // Accumulate
    float4 sample = make_float4(radiance.x, radiance.y, radiance.z, 1.0f);
    d_accumBuffer[pixelIdx] = d_accumBuffer[pixelIdx] + sample;
    float invN = 1.0f / (float)(sampleIndex + 1);
    d_outputBuffer[pixelIdx] = d_accumBuffer[pixelIdx] * invN;
}

void launchPathTraceKernel(
    const DeviceSceneData& scene,
    const CameraParams& camera,
    float4* d_accumBuffer,
    float4* d_outputBuffer,
    AuxBufferPtrs auxBuffers,
    uint32_t width,
    uint32_t height,
    uint32_t sampleIndex,
    bool enableEnvironment,
    uint32_t maxBounces)
{
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    pathTraceKernel<<<grid, block>>>(
        scene, camera, d_accumBuffer, d_outputBuffer, auxBuffers,
        width, height, sampleIndex, enableEnvironment, maxBounces);
    CUDA_CHECK(cudaGetLastError());
}
