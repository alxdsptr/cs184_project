#include "render/PathTraceKernel.h"
#include "core/Math.h"
#include "core/Halton.h"
#include "gpu/RayTypes.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "gpu/Sampling.h"
#include "accel/BVH.h"
#include "util/CudaCheck.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

static constexpr int MAX_BOUNCES = 8;

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
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotX / (NdotX * (1.0f - k) + k + 1e-7f);
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
    uint32_t        sampleIndex)
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

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
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
            // Environment
            radiance += throughput * sampleEnvironment(ray.direction);
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

        // Sample albedo texture if available
        float3 albedo = mat.albedo;
        if (mat.albedoTex != 0) {
            float4 texColor = tex2D<float4>(mat.albedoTex, hit.uv.x, hit.uv.y);
            albedo = make_float3(texColor.x, texColor.y, texColor.z);
        }

        // Interpolate vertex normals if available
        float3 N = hit.shadingNormal;
        if (scene.d_normals) {
            uint32_t triIdx = (uint32_t)hit.primitiveIndex;
            uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
            uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
            uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
            float3 n0 = scene.d_normals[i0];
            float3 n1 = scene.d_normals[i1];
            float3 n2 = scene.d_normals[i2];
            float u = hit.uv.x, v = hit.uv.y;
            N = normalize(n0 * (1.0f - u - v) + n1 * u + n2 * v);
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

        // Emission
        if (mat.emissionStrength > 0.0f) {
            radiance += throughput * mat.emission * mat.emissionStrength;
        }

        // BRDF sampling: metallic blend between diffuse and specular
        float specProb = 0.5f * (1.0f + mat.metallic);
        float3 V = -ray.direction;

        float3 newDir;
        float pdf;

        if (pcg32_float(rng) < specProb) {
            // GGX importance sampling (simplified: sample around reflection)
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

            float NdotH = fmaxf(dot(N, H), 0.0f);
            float VdotH = fmaxf(dot(V, H), 0.0f);
            float D_val = ggxD_local(NdotH, mat.roughness);
            pdf = D_val * NdotH / (4.0f * VdotH + 1e-7f);
            pdf *= specProb;
        } else {
            // Cosine-weighted hemisphere sampling (diffuse)
            float u1 = pcg32_float(rng);
            float u2 = pcg32_float(rng);
            float cosTheta;
            float3 localDir = sampleCosineHemisphere(u1, u2, pdf);
            float3 T, B;
            buildONB(N, T, B);
            newDir = localToWorld(localDir, T, N, B);
            pdf *= (1.0f - specProb);
        }

        if (pdf < 1e-7f || dot(newDir, N) < 0.0f) break;

        // Evaluate BRDF
        float3 H = normalize(V + newDir);
        float NdotL = fmaxf(dot(N, newDir), 0.0f);
        float NdotV = fmaxf(dot(N, V), 0.0f);
        float NdotH = fmaxf(dot(N, H), 0.0f);
        float LdotH = fmaxf(dot(newDir, H), 0.0f);

        float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, mat.metallic);
        float3 F = fresnelSchlick_local(LdotH, F0);
        float D_val = ggxD_local(NdotH, mat.roughness);
        float G_val = smithG1_local(NdotL, mat.roughness) * smithG1_local(NdotV, mat.roughness);

        float3 specular = F * (D_val * G_val / (4.0f * NdotL * NdotV + 1e-7f));
        float3 kd = (make_float3(1,1,1) - F) * (1.0f - mat.metallic);
        float3 diffuse = kd * albedo * (1.0f / M_PI_F);
        float3 brdf = (diffuse + specular) * NdotL;

        throughput = throughput * brdf * (1.0f / (pdf + 1e-7f));

        // Russian roulette after bounce 3
        if (bounce >= 3) {
            float p = fminf(fmaxf(fmaxf(throughput.x, throughput.y), throughput.z), 0.95f);
            if (pcg32_float(rng) >= p) break;
            throughput = throughput * (1.0f / p);
        }

        // Next ray
        ray.origin    = hit.position + N * 0.001f;
        ray.direction = newDir;
        ray.tmin      = 0.001f;
        ray.tmax      = 1e30f;
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
    uint32_t sampleIndex)
{
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    pathTraceKernel<<<grid, block>>>(
        scene, camera, d_accumBuffer, d_outputBuffer, auxBuffers,
        width, height, sampleIndex);
    CUDA_CHECK(cudaGetLastError());
}
