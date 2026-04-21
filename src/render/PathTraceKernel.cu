#include "render/PathTraceKernel.h"
#include "core/Math.h"
#include "core/Halton.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/RayTypes.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "gpu/Sampling.h"
#include "gpu/BRDF.h"
#include "accel/BVH.h"
#include "util/CudaCheck.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ── Environment ──────────────────────────────────────────────
__device__ inline float3 sampleEnvironment(float3 dir, cudaTextureObject_t envMap) {
    if (envMap != 0) {
        // Equirectangular HDR mapping: direction -> (u, v)
        float theta = acosf(fminf(fmaxf(dir.y, -1.0f), 1.0f)); // [0, pi]
        float phi   = atan2f(dir.z, dir.x);                      // [-pi, pi]
        float u = (phi + M_PI_F) * (0.5f / M_PI_F);             // [0, 1]
        float v = theta / M_PI_F;                                 // [0, 1]
        float4 texel = tex2D<float4>(envMap, u, v);
        return make_float3(texel.x, texel.y, texel.z);
    }
    // Fallback: procedural sky gradient
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
    return a2 / (M_PI_F * denom * denom + 1e-14f);
}

__device__ inline float3 fresnelSchlick_local(float cosTheta, float3 F0) {
    float t = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    return F0 + (make_float3(1,1,1) - F0) * t5;
}

__device__ inline float smithG1_GGX(float NdotX, float alpha) {
    // Smith G1 for GGX (exact form, not Schlick approximation)
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
    if (NdotH <= 0.0f || VdotH <= 0.0f) {
        return 0.0f;
    }

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
    float alpha = roughness * roughness;
    float G_val = smithG1_GGX(NdotL, alpha) * smithG1_GGX(NdotV, alpha);

    float3 specular = F * (D_val * G_val / (4.0f * NdotL * NdotV + 1e-7f));
    float3 kd = (make_float3(1, 1, 1) - F) * (1.0f - metallic);
    float3 diffuse = kd * albedo * (1.0f / M_PI_F);
    return diffuse + specular;
}

// ── Specular-Glossiness BRDF ─────────────────────────────────
// Same Cook-Torrance form as bsdfEvaluate but takes F0 directly (no
// metallic→F0 substitution). The diffuse term still uses (1-F) for energy
// conservation but is not darkened by metallic.
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
    if (NdotL <= 0.0f || NdotV <= 0.0f) return make_float3(0, 0, 0);

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

// Material-aware wrappers — respect GPUMaterial::pureDiffuse to render legacy
// Collada Phong materials as a pure Lambertian BRDF (no dielectric F0 lobe),
// and dispatch to the SG path when useSpecularGlossiness is set.
__device__ inline float materialSpecProb(
    const GPUMaterial& mat,
    const float3& N, const float3& V, const float3& albedo)
{
    if (mat.pureDiffuse) return 0.0f;
    // SG materials are remapped per-pixel into MR (metallic from spec.rgb
    // luminance, albedo blended with spec colour) before BRDF evaluation, so
    // the standard MR path is correct here.
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

// Fetch the radiance Le emitted at a barycentric point on an area light,
// handling the textured/untextured cases. For textured emitters we sample the
// emissive texture via the UVs stored on the light and multiply by the
// per-light `emission` (= albedo × emissionStrength, set at load time).
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
    uint32_t        maxBounces,
    uint32_t        samplesPerPixel)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint32_t pixelIdx = y * width + x;

    if (samplesPerPixel < 1) samplesPerPixel = 1;

    // Sum of per-sample radiance over this frame's spp. Added to the accum
    // buffer as one batch; caller advances the sample counter by `spp`.
    float3 radianceSum = make_float3(0, 0, 0);
    bool gbufferWritten = false;

    for (uint32_t s = 0; s < samplesPerPixel; s++) {
    // Unique RNG subseed per (pixel, frame, sample-in-frame).
    uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B9u + s,
                              sampleIndex * 0x85EBCA6Bu + s);

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
    // True only when the previous bounce used a *delta* BSDF (perfect mirror /
    // glass refraction) whose sampling pdf is a Dirac. In that case we cannot
    // MIS the next emissive hit with light-sampling (the latter has pdf = 0 in
    // the delta direction), so weight defaults to 1. Cook-Torrance specular
    // lobes are NOT delta — they have a finite roughness and a real pdf — so
    // this flag stays false after GGX sampling, letting MIS do its job.
    bool lastBounceDelta = false;
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
                float3 envColor = sampleEnvironment(ray.direction, scene.envMapTex);
                // Clamp extremely bright HDR texels (sun etc.) to prevent
                // fireflies when a path through glass hits a hot pixel.
                float envLum = 0.2126f * envColor.x + 0.7152f * envColor.y + 0.0722f * envColor.z;
                float envClamp = 100.0f;
                if (envLum > envClamp) {
                    envColor = envColor * (envClamp / envLum);
                }
                radiance += throughput * envColor;
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
            mat.useSpecularGlossiness = 0;
            mat.specularGlossAlphaIsGlossiness = 0;
            mat.specularColor = make_float3(1.0f, 1.0f, 1.0f);
            mat.glossiness = 0.5f;
            mat.specularGlossTex = 0;
        }

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

        // Sample metallic-roughness texture (glTF convention: G=roughness, B=metallic)
        if (mat.metallicRoughTex != 0) {
            float4 mrTexel = tex2D<float4>(mat.metallicRoughTex, texUV.x, texUV.y);
            mat.roughness = mat.roughness * mrTexel.y;
            mat.metallic = mat.metallic * mrTexel.z;
        }

        // Specular-Glossiness "soft" interpretation: the spec map's chromaticity
        // is unreliable across assets (saturated magentas / yellows that aren't
        // physical F0), but its *luminance* is a meaningful "how reflective is
        // this pixel" signal. We use spec luminance to drive both roughness and
        // F0 strength so that bright-spec areas (MEASURE_SEVEN's polished floor)
        // get visible mirror-like highlights, while dark-spec areas (matte
        // walls, fabric) stay properly diffuse. Albedo keeps its BaseColor
        // chromaticity — only its weighting in the BRDF changes.
        //
        // Encoding the target F0 through (metallic, albedo) so we don't fork
        // the BRDF: setting metallic = m and albedo = a gives F0 =
        //   lerp(0.04, a, m) = 0.04*(1-m) + a*m.
        // We pick m so that F0 (white) = 0.04 + (Ftarget - 0.04) = Ftarget when
        // a is treated as white at the F0-mixing site. To keep diffuse colour
        // intact we apply this ONLY to F0; the diffuse term still uses the
        // original BaseColor through kd = (1-F)*(1-m), which gracefully fades
        // diffuse as the surface becomes more metallic — the exact behaviour
        // we want for polished stone / brushed metal.
        if (mat.useSpecularGlossiness) {
            float3 specRGB = mat.specularColor;
            float  alphaG  = 1.0f;
            if (mat.specularGlossTex != 0) {
                float4 sg = tex2D<float4>(mat.specularGlossTex, texUV.x, texUV.y);
                specRGB = mat.specularColor * make_float3(sg.x, sg.y, sg.z);
                alphaG  = sg.w;
            }
            float specLum = 0.2126f * specRGB.x + 0.7152f * specRGB.y + 0.0722f * specRGB.z;
            specLum = clampf(specLum, 0.0f, 1.0f);

            // sqrt curve: even mid-luminance spec gives a noticeable polish,
            // matching how artists usually paint these masks.
            float specStrength = sqrtf(specLum);

            // Target F0 ranges from dielectric baseline (0.04) to ~0.6 for the
            // brightest spec values — well into "polished metal" territory but
            // capped below pure mirror to leave headroom for the BaseColor tint
            // visible in the diffuse lobe.
            float F0_target = 0.04f + 0.56f * specStrength;
            mat.metallic = clampf((F0_target - 0.04f) / 0.96f, 0.0f, 1.0f);

            // When alpha carries glossiness data, use it (modulated by the
            // material factor). When it doesn't, let spec luminance directly
            // drive glossiness so bright-spec areas show clear reflections;
            // the scalar `glossiness` factor only acts as an upper bound /
            // global trim.
            float gloss;
            if (mat.specularGlossAlphaIsGlossiness) {
                gloss = mat.glossiness * alphaG;
            } else {
                gloss = specStrength;
            }
            mat.roughness = 1.0f - clampf(gloss, 0.0f, 0.95f);
        }

        // Clamp roughness/metallic to stable ranges for BRDF sampling/evaluation
        mat.roughness = fmaxf(mat.roughness, 0.045f);
        mat.metallic = clampf(mat.metallic, 0.0f, 1.0f);

        // Sample emissive texture if available
        float3 emissiveColor = mat.emission;
        if (mat.emissiveTex != 0) {
            float4 emissiveTexel = tex2D<float4>(mat.emissiveTex, texUV.x, texUV.y);
            emissiveColor = make_float3(emissiveTexel.x, emissiveTexel.y, emissiveTexel.z);
        }

        // Interpolate vertex normals if available
        float3 N = hit.shadingNormal;
        if (scene.d_normals) {
            float3 n0 = scene.d_normals[i0];
            float3 n1 = scene.d_normals[i1];
            float3 n2 = scene.d_normals[i2];
            N = normalize(n0 * baryW + n1 * baryU + n2 * baryV);
        }
        // Normal mapping: skip for glass so refraction stays on the geometric
        // surface. For opaque surfaces, perturb before the back-face flip so
        // the flip considers the perturbed orientation.
        if (mat.transmission <= 0.0f && mat.normalTex != 0 && scene.d_tangents) {
            float4 t0 = scene.d_tangents[i0];
            float4 t1 = scene.d_tangents[i1];
            float4 t2 = scene.d_tangents[i2];
            float4 tangent = t0 * baryW + t1 * baryU + t2 * baryV;
            N = applyNormalMap(N, tangent, mat.normalTex, texUV);
        }
        // For glass, preserve the original normal orientation (frontFace is the ground truth).
        // For opaque surfaces, flip to face the ray as before.
        if (mat.transmission <= 0.0f) {
            if (dot(N, ray.direction) > 0) N = -N;
        }

        // Write aux buffers from the first sample that produces a primary hit.
        // `firstBounce` guards within a path; `!gbufferWritten` guards across
        // the samples-per-pixel loop so we don't overwrite on subsequent spp.
        if (firstBounce) {
            if (!gbufferWritten) {
                if (auxBuffers.d_linearDepth)
                    auxBuffers.d_linearDepth[pixelIdx] = dot(hit.position - camera.position, camera.forward);
                if (auxBuffers.d_albedo)
                    auxBuffers.d_albedo[pixelIdx] = albedo;
                if (auxBuffers.d_normal)
                    auxBuffers.d_normal[pixelIdx] = N;
                if (auxBuffers.d_motionVectors) {
                    float3 clipCurr = mat4_transformPoint(camera.viewProjMatrix, hit.position);
                    float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hit.position);
                    float2 screenCurr = make_float2((clipCurr.x + 1.0f) * 0.5f * width,
                                                     (1.0f - clipCurr.y) * 0.5f * height);
                    float2 screenPrev = make_float2((clipPrev.x + 1.0f) * 0.5f * width,
                                                     (1.0f - clipPrev.y) * 0.5f * height);
                    auxBuffers.d_motionVectors[pixelIdx] = screenCurr - screenPrev;
                }
                gbufferWritten = true;
            }
            firstBounce = false;
        }

        // ── Glass / transmissive material ───────────────────────
        bool handledAsGlass = false;
        if (mat.transmission > 0.0f) {
            // Determine if we're entering or exiting the medium
            bool entering = hit.frontFace;

            // Nglass: the outward-facing normal (always points to the side the ray came from)
            // For glass we did NOT flip N above, so use frontFace to orient it.
            float3 Nglass = entering ? N : -N;
            // Make sure Nglass faces the incoming ray
            if (dot(Nglass, ray.direction) > 0.0f) Nglass = -Nglass;

            float etaI = entering ? 1.0f : mat.ior;
            float etaT = entering ? mat.ior : 1.0f;
            float eta = etaI / etaT;

            float cosThetaI = fmaxf(dot(-ray.direction, Nglass), 0.0f);

            // Exact Fresnel for dielectrics
            float Fr = fresnelDielectric(cosThetaI, eta);

            float3 newDirGlass;
            if (pcg32_float(rng) < Fr) {
                // Reflection
                newDirGlass = ray.direction - Nglass * (2.0f * dot(ray.direction, Nglass));
                newDirGlass = normalize(newDirGlass);
            } else {
                // Refraction (Snell's law)
                if (!refractDir(ray.direction, Nglass, eta, newDirGlass)) {
                    // Total internal reflection fallback
                    newDirGlass = ray.direction - Nglass * (2.0f * dot(ray.direction, Nglass));
                    newDirGlass = normalize(newDirGlass);
                }
            }

            // Glass tint: colored glass absorbs light based on albedo.
            // Only apply tint when exiting the medium AND the color is
            // intentionally non-white (skip near-white to keep clear glass clear).
            if (!entering) {
                float albedoLum = 0.2126f * albedo.x + 0.7152f * albedo.y + 0.0722f * albedo.z;
                if (albedoLum < 0.9f) {
                    throughput = throughput * albedo;
                }
            }

            // Delta BSDF: throughput unchanged by pdf (delta distribution cancels)
            // Offset origin in the direction of travel to avoid self-intersection
            float3 offsetN = (dot(newDirGlass, Nglass) > 0.0f) ? Nglass : -Nglass;
            ray.origin    = hit.position + offsetN * 0.002f;
            ray.direction = newDirGlass;
            ray.tmin      = 0.001f;
            ray.tmax      = 1e30f;

            lastBounceDelta = true;   // glass refraction/reflection is delta
            prevSurfacePos = hit.position;
            prevBsdfPdf = 1.0f;
            havePrevSurface = true;
            handledAsGlass = true;
        }
        if (handledAsGlass) {
            // For glass, also flip N for aux buffers (ensure outward-facing for denoiser)
            if (dot(N, ray.direction) > 0) N = -N;
            // Glass Russian roulette: only terminate after many bounces to
            // prevent infinite TIR loops, but do NOT boost throughput (delta
            // BSDF doesn't lose energy so boosting causes fireflies).
            if (bounce >= 6) {
                if (pcg32_float(rng) > 0.9f) break;
            }
            continue; // Skip NEE and opaque BRDF — glass is a delta BSDF
        }

        bool isEmissive = mat.emissionStrength > 0.0f &&
                          (emissiveColor.x > 0.0f || emissiveColor.y > 0.0f || emissiveColor.z > 0.0f);
        if (isEmissive) {
            float3 Le = emissiveColor * mat.emissionStrength;
            float weight = 1.0f;

            // Check if this triangle is a registered area light (NEE-sampleable
            // via the area light CDF). Texture-emitter triangles are now
            // registered too, so MIS applies uniformly.
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

            // For textured emissives (e.g. light bulbs): continue the path so the
            // surface also reflects light and shows specular highlights / depth.
            // For pure area lights (uniform emission, no texture): terminate as before.
            if (mat.emissiveTex != 0) {
                // fall through to BRDF sampling below
            } else {
                break;
            }
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
                // Shadow ray with glass transparency
                float3 shadowTransmittance = make_float3(1.0f, 1.0f, 1.0f);
                bool occluded = false;
                if (scene.d_bvhNodes && scene.totalTriangles > 0) {
                    Ray shadowRay;
                    shadowRay.origin = hit.position + N * 0.001f;
                    shadowRay.direction = Ld;
                    shadowRay.tmin = 0.001f;
                    shadowRay.tmax = fmaxf(dist - 0.002f, 0.001f);

                    for (int shadowStep = 0; shadowStep < 8; shadowStep++) {
                        HitRecord shadowHit;
                        shadowHit.t = shadowRay.tmax;
                        bool didHitShadow = bvh_closestHit(
                            shadowRay, scene.d_bvhNodes, scene.bvhRootIndex,
                            scene.d_positions, scene.d_indices, scene.d_materialIndices,
                            shadowHit);
                        if (!didHitShadow) break;

                        // Check if the hit surface is glass
                        GPUMaterial shadowMat;
                        if (shadowHit.materialIndex >= 0 && (uint32_t)shadowHit.materialIndex < scene.materialCount)
                            shadowMat = scene.d_materials[shadowHit.materialIndex];
                        else { occluded = true; break; }

                        if (shadowMat.transmission > 0.0f) {
                            // Attenuate by glass color for colored glass;
                            // near-white glass is treated as fully transparent.
                            float sAlbLum = 0.2126f * shadowMat.albedo.x + 0.7152f * shadowMat.albedo.y + 0.0722f * shadowMat.albedo.z;
                            if (sAlbLum < 0.9f) {
                                shadowTransmittance = shadowTransmittance * shadowMat.albedo;
                            }
                            // Continue shadow ray past the glass surface
                            shadowRay.origin = shadowHit.position + Ld * 0.002f;
                            shadowRay.tmax = fmaxf(dist - length(shadowRay.origin - (hit.position + N * 0.001f)) - 0.002f, 0.001f);
                        } else {
                            occluded = true;
                            break;
                        }
                    }
                }

                float shadowLum = 0.2126f * shadowTransmittance.x + 0.7152f * shadowTransmittance.y + 0.0722f * shadowTransmittance.z;
                if (!occluded && shadowLum > 1e-6f) {
                    float pTri = light.weight / scene.areaLightTotalWeight;
                    float pArea = pTri / fmaxf(light.area, 1e-7f);
                    float pdfOmega = pArea * dist2 / fmaxf(lightNdot, 1e-7f);

                    float3 V = -ray.direction;
                    float3 brdf = materialBsdfEvaluate(mat, N, V, Ld, albedo);
                    float neeSpecProb = materialSpecProb(mat, N, V, albedo);
                    float pdfBsdf = materialMixturePdf(mat, N, V, Ld, neeSpecProb);
                    float weight = powerHeuristic(pdfOmega, pdfBsdf);

                    float3 Le = sampleAreaLightLe(light, b0, b1, b2);
                    radiance += throughput * shadowTransmittance * brdf * Le * (NdotL / fmaxf(pdfOmega, 1e-7f)) * weight;
                }
            }
        }

        // Point lights are delta emitters: BSDF-sampling can never hit them,
        // so they are always sampled independently (no MIS, no area-light
        // exclusivity). Bistro puts its main illumination on 4 point lights
        // in addition to emissive mesh geometry — gating this branch behind
        // "no area lights" would drop those entirely.
        if (scene.d_pointLights && scene.pointLightCount > 0) {
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

                // Shadow ray with glass transparency
                float3 shadowTransmittancePL = make_float3(1.0f, 1.0f, 1.0f);
                bool occluded = false;
                if (scene.d_bvhNodes && scene.totalTriangles > 0) {
                    Ray shadowRay;
                    shadowRay.origin = hit.position + N * 0.001f;
                    shadowRay.direction = Ld;
                    shadowRay.tmin = 0.001f;
                    shadowRay.tmax = fmaxf(dist - 0.002f, 0.001f);

                    for (int shadowStep = 0; shadowStep < 8; shadowStep++) {
                        HitRecord shadowHit;
                        shadowHit.t = shadowRay.tmax;
                        bool didHitShadow = bvh_closestHit(
                            shadowRay, scene.d_bvhNodes, scene.bvhRootIndex,
                            scene.d_positions, scene.d_indices, scene.d_materialIndices,
                            shadowHit);
                        if (!didHitShadow) break;

                        GPUMaterial shadowMat;
                        if (shadowHit.materialIndex >= 0 && (uint32_t)shadowHit.materialIndex < scene.materialCount)
                            shadowMat = scene.d_materials[shadowHit.materialIndex];
                        else { occluded = true; break; }

                        if (shadowMat.transmission > 0.0f) {
                            float sAlbLumPL = 0.2126f * shadowMat.albedo.x + 0.7152f * shadowMat.albedo.y + 0.0722f * shadowMat.albedo.z;
                            if (sAlbLumPL < 0.9f) {
                                shadowTransmittancePL = shadowTransmittancePL * shadowMat.albedo;
                            }
                            shadowRay.origin = shadowHit.position + Ld * 0.002f;
                            shadowRay.tmax = fmaxf(dist - length(shadowRay.origin - (hit.position + N * 0.001f)) - 0.002f, 0.001f);
                        } else {
                            occluded = true;
                            break;
                        }
                    }
                }

                float shadowLumPL = 0.2126f * shadowTransmittancePL.x + 0.7152f * shadowTransmittancePL.y + 0.0722f * shadowTransmittancePL.z;
                if (occluded || shadowLumPL < 1e-6f) continue;

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

        // BRDF sampling: Fresnel-weighted blend between diffuse and specular
        float3 V = -ray.direction;
        float specProb = materialSpecProb(mat, N, V, albedo);

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
            // Cook-Torrance specular lobe is NOT a delta — MIS is valid.
            lastBounceDelta = false;
        } else {
            // Cosine-weighted hemisphere sampling (diffuse)
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

        // Compute full mixture PDF for the sampled direction
        float pdf = materialMixturePdf(mat, N, V, newDir, specProb);
        if (pdf < 1e-7f) break;

        // Evaluate BRDF
        float3 brdf = materialBsdfEvaluate(mat, N, V, newDir, albedo);

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
    float clampMax = 200.0f;
    if (luminance > clampMax) {
        float scale = clampMax / luminance;
        radiance = radiance * scale;
    }

        radianceSum = radianceSum + radiance;
    } // end spp loop

    // Accumulate: add all spp samples at once. The caller advances the sample
    // counter by `samplesPerPixel`, so the divisor below stays correct.
    float4 sumTexel = make_float4(radianceSum.x, radianceSum.y, radianceSum.z, (float)samplesPerPixel);
    d_accumBuffer[pixelIdx] = d_accumBuffer[pixelIdx] + sumTexel;
    float invN = 1.0f / (float)(sampleIndex + samplesPerPixel);
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
    uint32_t maxBounces,
    uint32_t samplesPerPixel)
{
    if (samplesPerPixel < 1) samplesPerPixel = 1;
    dim3 block(8, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    pathTraceKernel<<<grid, block>>>(
        scene, camera, d_accumBuffer, d_outputBuffer, auxBuffers,
        width, height, sampleIndex, enableEnvironment, maxBounces, samplesPerPixel);
    CUDA_CHECK(cudaGetLastError());
}
