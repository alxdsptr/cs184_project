#include "render/ReSTIR.h"
#include "core/Math.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "accel/BVH.h"
#include "accel/LightBVHSample.h"
#include "util/CudaCheck.h"

#include <cuda_runtime.h>

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// ─────────────────────────────────────────────────────────────────────────
// ReSTIR DI — primary-hit only, biased spatial reuse.
//
// The whole pipeline for one frame runs in three kernels below. Each pixel
// owns a ReSTIRReservoir that holds *one* light sample along with the
// probability-weighting needed to turn it into an unbiased estimator:
//
//   E[ f / target * W ]  =  integral( f )                       [RIS]
//
// where target(x) = luminance(Le) * |f_r(wi)| * G(x,y) is evaluated without
// visibility — shadow ray is cast once at final shading.
// ─────────────────────────────────────────────────────────────────────────

// ── Minimal BRDF helpers used only for the target-pdf evaluation ──
// We intentionally mirror the main kernel's behavior (materialSpecProb,
// bsdfEvaluate, bsdfDiffusePdf) rather than reuse its file — PathTraceKernel.cu
// is a megakernel translation unit and its symbols are static. The target pdf
// only needs a scalar luminance so small approximations are fine: we use the
// same mixture as the main kernel but read the cached specProb off the
// surface record, avoiding recomputation.

__device__ inline float luminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ inline float ggx_D(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / (M_PI_F * denom * denom + 1e-14f);
}
__device__ inline float smith_G1(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    float cos2 = NdotX * NdotX;
    return 2.0f * NdotX / (NdotX + sqrtf(a2 + (1.0f - a2) * cos2) + 1e-7f);
}
__device__ inline float3 fresnelSchlick(float cosTheta, float3 F0) {
    float t = 1.0f - fminf(fmaxf(cosTheta, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    return F0 + (make_float3(1,1,1) - F0) * t5;
}

__device__ inline float3 evalBrdf_local(
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
    float3 F = fresnelSchlick(LdotH, F0);
    float Dt = ggx_D(NdotH, s.roughness);
    float alpha = s.roughness * s.roughness;
    float Gt = smith_G1(NdotL, alpha) * smith_G1(NdotV, alpha);

    float3 spec = F * (Dt * Gt / (4.0f * NdotL * NdotV + 1e-7f));
    float3 kd = (make_float3(1,1,1) - F) * (1.0f - s.metallic);
    float3 diff = kd * s.albedo * (1.0f / M_PI_F);
    return diff + spec;
}

// Target pdf = luminance(Le) * |BRDF * NdotL| * geometry, NO visibility.
// Returns 0 if the sample is back-facing on either surface.
__device__ inline float evalTargetPdf(
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

    // Use the on-triangle emission as a proxy. Textured emitters still work —
    // we slightly under-weight bright texels, but the final shadow-ray pass
    // re-fetches the texel so no energy is lost.
    float Lum = luminance(light.emission);
    if (Lum <= 0.0f) return 0.0f;

    float3 brdf = evalBrdf_local(s, L);
    float  fLum = luminance(brdf) * NdotL;
    if (fLum <= 0.0f) return 0.0f;

    // Geometric factor = lightNdot / dist^2  (NdotL already in fLum)
    float geom = lightNdot / dist2;
    return Lum * fLum * geom;
}

// Duplicate of generateRay from PathTraceKernel.cu — that file's static
// definitions aren't visible here. Keeping it identical in math so reservoirs
// line up with the main kernel's ray (same jitter, same FOV).
__device__ inline Ray generateRayReSTIR(
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

// ── Reservoir primitives (Alg. 2 in Bitterli 2020) ──────────────
// updateReservoir:      stream one candidate with weight w_i (= p_hat / pdf).
// combineReservoir:     stream an entire other reservoir (used for reuse).
// Both share the same code modulo how they aggregate M.

__device__ inline void reservoir_reset(ReSTIRReservoir& r) {
    r.lightIndex = 0xFFFFFFFFu;
    r.baryB1 = 0.0f;
    r.baryB2 = 0.0f;
    r.pHat   = 0.0f;
    r.W      = 0.0f;
    r.M      = 0.0f;
}

// Returns true if the incoming candidate was accepted as the held sample.
// wSum is the running sum of candidate weights; pass &wSum across updates.
__device__ inline bool reservoir_update(
    ReSTIRReservoir& r, float& wSum,
    uint32_t lightIdx, float b1, float b2, float pHat,
    float wCandidate, float u01)
{
    if (!(wCandidate > 0.0f)) return false;
    wSum += wCandidate;
    r.M  += 1.0f;
    if (u01 * wSum < wCandidate) {
        r.lightIndex = lightIdx;
        r.baryB1     = b1;
        r.baryB2     = b2;
        r.pHat       = pHat;
        return true;
    }
    return false;
}

// After all streaming, convert (pHat, wSum, M) into the unbiased contribution
// weight W.  W = wSum / (M * pHat)  (Bitterli eq. 9).
__device__ inline void reservoir_finalize(ReSTIRReservoir& r, float wSum) {
    if (r.lightIndex == 0xFFFFFFFFu || r.pHat <= 0.0f || r.M <= 0.0f) {
        r.W = 0.0f;
        return;
    }
    r.W = wSum / (r.M * r.pHat);
}

// Stream an entire neighbor reservoir into `dst`. This is the spatial/temporal
// combine operator: pretend we drew `src.M` candidates with effective weight
// `src.pHat * src.W * src.M` (so that the RIS invariant holds), re-evaluated
// at `dst`'s surface.
__device__ inline bool reservoir_combine(
    ReSTIRReservoir& dst, float& wSum,
    const ReSTIRReservoir& src, float pHatAtDst, float u01)
{
    if (src.lightIndex == 0xFFFFFFFFu || src.M <= 0.0f) {
        // Still count M so temporal cap logic sees the history length.
        dst.M += src.M;
        return false;
    }
    // Candidate weight contributed by this reservoir: pHatAtDst * W * M.
    // When pHatAtDst = 0 (sample is behind the dst surface, or self-shadowed
    // by geometry at dst's orientation) we still add to M but do not replace.
    float w = pHatAtDst * src.W * src.M;
    bool accepted = false;
    if (w > 0.0f) {
        wSum += w;
        if (u01 * wSum < w) {
            dst.lightIndex = src.lightIndex;
            dst.baryB1     = src.baryB1;
            dst.baryB2     = src.baryB2;
            dst.pHat       = pHatAtDst;
            accepted = true;
        }
    }
    dst.M += src.M;
    return accepted;
}

// ── Kernel 1: initial candidate generation ──────────────────────
// For each pixel: cast the primary ray, resolve the material (a cut-down
// version of the main kernel's logic — enough for pHat), draw M light
// candidates via the source-pdf (flat weight / totalWeight, or the BVH's
// position-aware pdf), RIS them into a reservoir.
__global__ void kReSTIR_InitCandidates(
    DeviceSceneData scene,
    CameraParams    camera,
    ReSTIRReservoir* outReservoirs,
    ReSTIRSurface*   outSurfaces,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    uint32_t numCandidates)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    // Dedicated RNG stream so ReSTIR and the main kernel don't correlate.
    // A different salt (0xA1B2C3D4u) ensures independent trajectories even
    // when sampleIndex matches.
    uint32_t rng = pcg32_seed(pixelIdx * 0xA1B2C3D4u + sampleIndex,
                              sampleIndex * 0xDEADBEEFu + 1u);

    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;
    Ray ray = generateRayReSTIR(x, y, width, height, camera, jx, jy);

    // Primary hit — fall back to empty reservoir on miss.
    HitRecord hit;
    hit.t = 1e30f;
    bool didHit = false;
    if (scene.d_bvhNodes && scene.totalTriangles > 0) {
        didHit = bvh_closestHit(ray, scene.d_bvhNodes, scene.bvhRootIndex,
                                scene.d_positions, scene.d_indices,
                                scene.d_materialIndices, hit);
    }

    ReSTIRReservoir r; reservoir_reset(r);
    ReSTIRSurface surf{};
    surf.valid = 0.0f;

    if (didHit && hit.materialIndex >= 0 && (uint32_t)hit.materialIndex < scene.materialCount
        && scene.d_areaLights && scene.areaLightCount > 0
        && scene.d_lightBVHNodes)
    {
        GPUMaterial mat = scene.d_materials[hit.materialIndex];
        // Interpolate normal / tangent / uv using the hit's barycentrics.
        uint32_t i0 = scene.d_indices[hit.primitiveIndex * 3 + 0];
        uint32_t i1 = scene.d_indices[hit.primitiveIndex * 3 + 1];
        uint32_t i2 = scene.d_indices[hit.primitiveIndex * 3 + 2];
        float b1 = hit.uv.x, b2 = hit.uv.y;
        float b0 = 1.0f - b1 - b2;
        float3 N = normalize(scene.d_normals[i0] * b0 +
                             scene.d_normals[i1] * b1 +
                             scene.d_normals[i2] * b2);
        if (dot(N, ray.direction) > 0.0f) N = -N;

        float2 uv = scene.d_uvs[i0] * b0 + scene.d_uvs[i1] * b1 + scene.d_uvs[i2] * b2;
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

        surf.position  = hit.position;
        surf.normal    = N;
        surf.albedo    = albedo;
        surf.roughness = fmaxf(mat.roughness, 0.04f);
        surf.metallic  = mat.metallic;
        surf.pureDiffuse = mat.pureDiffuse ? 1u : 0u;
        surf.viewDir   = -ray.direction;
        surf.valid     = 1.0f;

        // Precompute cached specProb (matches main kernel's heuristic).
        {
            float NdotV = fmaxf(dot(surf.normal, surf.viewDir), 0.0f);
            float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), surf.albedo, surf.metallic);
            float t = 1.0f - fminf(fmaxf(NdotV, 0.0f), 1.0f);
            float t5 = t*t*t*t*t;
            float3 F = F0 + (make_float3(1,1,1) - F0) * t5;
            float specW = luminance(F);
            float3 kd = (make_float3(1,1,1) - F) * (1.0f - surf.metallic);
            float diffW = luminance(kd * surf.albedo);
            float p = specW / fmaxf(specW + diffW, 1e-7f);
            surf.specProb = fminf(fmaxf(p, 0.1f), 0.9f);
        }

        // Screen-space motion for temporal lookup next frame.
        float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hit.position);
        float2 prevPx = make_float2((clipPrev.x + 1.0f) * 0.5f * width,
                                     (1.0f - clipPrev.y) * 0.5f * height);
        surf.prevPixel = prevPx;

        // ── RIS: draw M candidates from the light BVH, stream them ──
        float wSum = 0.0f;
        for (uint32_t i = 0; i < numCandidates; i++) {
            float u = pcg32_float(rng);
            uint32_t slot = 0;
            float pSelect = 0.0f;
            if (!lightBVH_sample(scene.d_lightBVHNodes,
                                 scene.lightBVHRootIndex,
                                 surf.position, u, slot, pSelect) || !(pSelect > 0.0f))
                continue;
            uint32_t lightIdx = scene.d_lightOrderedIndices[slot];
            GPUAreaLight light = scene.d_areaLights[lightIdx];

            // Sample a uniform point on the triangle (same mapping the main kernel uses).
            float r1 = pcg32_float(rng);
            float r2 = pcg32_float(rng);
            float su = sqrtf(r1);
            float cb1 = su * (1.0f - r2);
            float cb2 = su * r2;

            // Source pdf (area-sampling): pSelect / area_tri.
            float areaPdf = pSelect / fmaxf(light.area, 1e-7f);
            float pHat = evalTargetPdf(surf, light, cb1, cb2);
            float wCand = (areaPdf > 0.0f) ? (pHat / areaPdf) : 0.0f;

            reservoir_update(r, wSum, lightIdx, cb1, cb2, pHat,
                             wCand, pcg32_float(rng));
        }
        reservoir_finalize(r, wSum);
    }

    outReservoirs[pixelIdx] = r;
    outSurfaces[pixelIdx]   = surf;
}

// ── Kernel 2: temporal reuse ────────────────────────────────────
// For each pixel: look up the reprojected previous-frame reservoir, reject
// on normal/depth mismatch, combine if accepted. Cap M to bound bias from
// temporal correlation.
__global__ void kReSTIR_Temporal(
    DeviceSceneData scene,
    ReSTIRReservoir* curr,
    const ReSTIRReservoir* prev,
    const ReSTIRSurface*   surfCurr,
    const ReSTIRSurface*   surfPrev,
    uint32_t width, uint32_t height,
    uint32_t prevWidth, uint32_t prevHeight,
    uint32_t sampleIndex,
    uint32_t mCap)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    ReSTIRReservoir r = curr[pixelIdx];
    ReSTIRSurface   s = surfCurr[pixelIdx];
    if (s.valid < 0.5f) { return; }

    uint32_t rng = pcg32_seed(pixelIdx * 0x5F356495u + sampleIndex,
                              sampleIndex * 0xB5297A4Du + 2u);

    // Reproject. `s.prevPixel` holds the pixel coordinate in the previous
    // frame for this shading point.
    int px = (int)floorf(s.prevPixel.x);
    int py = (int)floorf(s.prevPixel.y);
    if (px < 0 || py < 0 || px >= (int)prevWidth || py >= (int)prevHeight) return;
    uint32_t prevIdx = (uint32_t)py * prevWidth + (uint32_t)px;

    ReSTIRSurface sp = surfPrev[prevIdx];
    if (sp.valid < 0.5f) return;
    // Reject on large normal or depth disparity (standard ReSTIR heuristics).
    // The 0.9 cosine threshold (~25°) keeps a surface and its neighbor in the
    // reuse pool but rejects history across hard normal discontinuities.
    if (dot(s.normal, sp.normal) < 0.9f) return;
    // Position-drift gate: 10% of the shading point's distance from origin is
    // a coarse proxy for camera distance, sufficient to reject history across
    // foreground/background transitions without requiring viewZ in the SoA.
    float drift = length(s.position - sp.position);
    if (drift > 0.1f * fmaxf(length(s.position), 1.0f)) return;

    ReSTIRReservoir pr = prev[prevIdx];
    if (pr.lightIndex == 0xFFFFFFFFu) return;
    // Cap the previous M so temporal bleed is bounded (Bitterli: 20*M_init).
    if (pr.M > (float)mCap) pr.M = (float)mCap;

    // Re-evaluate pHat of pr's sample at the current surface.
    GPUAreaLight light = scene.d_areaLights[pr.lightIndex];
    float pHatAtCurr = evalTargetPdf(s, light, pr.baryB1, pr.baryB2);

    // Rebuild wSum for the current reservoir so we can continue streaming.
    // Before combine: wSum = r.pHat * r.M * r.W  (inverse of finalize).
    float wSum = r.pHat * r.M * r.W;
    reservoir_combine(r, wSum, pr, pHatAtCurr, pcg32_float(rng));
    reservoir_finalize(r, wSum);

    curr[pixelIdx] = r;
}

// ── Kernel 3: spatial reuse ─────────────────────────────────────
// Each pixel samples k neighbors in a disk and combines their reservoirs into
// a scratch buffer. Biased formulation (no Jacobian) — visually matches
// ground truth in most scenes and is ~2x faster than the unbiased form.
__global__ void kReSTIR_Spatial(
    DeviceSceneData scene,
    const ReSTIRReservoir* inRes,
    ReSTIRReservoir*       outRes,
    const ReSTIRSurface*   surf,
    uint32_t width, uint32_t height,
    uint32_t sampleIndex,
    uint32_t numNeighbors,
    float    radiusPixels)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    ReSTIRReservoir r = inRes[pixelIdx];
    ReSTIRSurface   s = surf[pixelIdx];
    if (s.valid < 0.5f) { outRes[pixelIdx] = r; return; }

    uint32_t rng = pcg32_seed(pixelIdx * 0x71937573u + sampleIndex,
                              sampleIndex * 0x9E3779B1u + 3u);

    float wSum = r.pHat * r.M * r.W;

    for (uint32_t i = 0; i < numNeighbors; i++) {
        // Uniform disk sample in pixel space.
        float u1 = pcg32_float(rng);
        float u2 = pcg32_float(rng);
        float rr = sqrtf(u1) * radiusPixels;
        float th = 2.0f * M_PI_F * u2;
        int nx = (int)x + (int)(rr * cosf(th));
        int ny = (int)y + (int)(rr * sinf(th));
        if (nx < 0 || ny < 0 || nx >= (int)width || ny >= (int)height) continue;
        uint32_t nIdx = (uint32_t)ny * width + (uint32_t)nx;

        ReSTIRSurface ns = surf[nIdx];
        if (ns.valid < 0.5f) continue;
        // Same geometry gate as the temporal pass.
        if (dot(s.normal, ns.normal) < 0.9f) continue;
        float dz = length(s.position - ns.position);
        if (dz > 0.1f * fmaxf(length(s.position), 1.0f)) continue;

        ReSTIRReservoir nr = inRes[nIdx];
        if (nr.lightIndex == 0xFFFFFFFFu) continue;

        GPUAreaLight light = scene.d_areaLights[nr.lightIndex];
        float pHatAtCurr = evalTargetPdf(s, light, nr.baryB1, nr.baryB2);
        reservoir_combine(r, wSum, nr, pHatAtCurr, pcg32_float(rng));
    }
    reservoir_finalize(r, wSum);
    outRes[pixelIdx] = r;
}

// ── Host launchers ──────────────────────────────────────────────
static inline dim3 makeGrid(uint32_t w, uint32_t h, dim3 block) {
    return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

void launchReSTIRInitialCandidates(
    const DeviceSceneData& scene,
    const CameraParams&    camera,
    ReSTIRBuffers          buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               numCandidates)
{
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIR_InitCandidates<<<grid, block>>>(
        scene, camera,
        buffers.d_reservoirsCurr, buffers.d_surfaceCurr,
        width, height, sampleIndex, numCandidates);
}

void launchReSTIRTemporalReuse(
    const DeviceSceneData& scene,
    ReSTIRBuffers          buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               temporalMCap)
{
    if (!buffers.historyValid) return;
    if (buffers.prevWidth == 0 || buffers.prevHeight == 0) return;
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIR_Temporal<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr,
        buffers.d_reservoirsPrev,
        buffers.d_surfaceCurr,
        buffers.d_surfacePrev,
        width, height,
        buffers.prevWidth, buffers.prevHeight,
        sampleIndex, temporalMCap);
}

void launchReSTIRSpatialReuse(
    const DeviceSceneData& scene,
    ReSTIRBuffers          buffers,
    uint32_t               width,
    uint32_t               height,
    uint32_t               sampleIndex,
    uint32_t               numNeighbors,
    float                  radiusPixels)
{
    // Writes scratch only; the caller (ReSTIRContext::runFrame) is in charge
    // of swapping d_reservoirsSpatial ↔ d_reservoirsCurr so the main path
    // tracer reads the spatial-pass output.
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIR_Spatial<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr,
        buffers.d_reservoirsSpatial,
        buffers.d_surfaceCurr,
        width, height, sampleIndex,
        numNeighbors, radiusPixels);
}

// ── ReSTIRContext implementation ────────────────────────────────
static void allocReservoirs(ReSTIRReservoir** p, uint32_t count) {
    CUDA_CHECK(cudaMalloc(p, count * sizeof(ReSTIRReservoir)));
    CUDA_CHECK(cudaMemset(*p, 0, count * sizeof(ReSTIRReservoir)));
}
static void allocSurfaces(ReSTIRSurface** p, uint32_t count) {
    CUDA_CHECK(cudaMalloc(p, count * sizeof(ReSTIRSurface)));
    CUDA_CHECK(cudaMemset(*p, 0, count * sizeof(ReSTIRSurface)));
}

void ReSTIRContext::init(uint32_t width, uint32_t height) {
    free();
    const uint32_t count = width * height;
    allocReservoirs(&m_buffers.d_reservoirsCurr,    count);
    allocReservoirs(&m_buffers.d_reservoirsPrev,    count);
    allocReservoirs(&m_buffers.d_reservoirsSpatial, count);
    allocSurfaces(&m_buffers.d_surfaceCurr, count);
    allocSurfaces(&m_buffers.d_surfacePrev, count);
    m_buffers.width      = width;
    m_buffers.height     = height;
    m_buffers.prevWidth  = width;
    m_buffers.prevHeight = height;
    m_buffers.historyValid = false;
}

void ReSTIRContext::resize(uint32_t width, uint32_t height) {
    if (width == m_buffers.width && height == m_buffers.height) return;
    init(width, height);
}

void ReSTIRContext::free() {
    if (m_buffers.d_reservoirsCurr)    cudaFree(m_buffers.d_reservoirsCurr);
    if (m_buffers.d_reservoirsPrev)    cudaFree(m_buffers.d_reservoirsPrev);
    if (m_buffers.d_reservoirsSpatial) cudaFree(m_buffers.d_reservoirsSpatial);
    if (m_buffers.d_surfaceCurr)       cudaFree(m_buffers.d_surfaceCurr);
    if (m_buffers.d_surfacePrev)       cudaFree(m_buffers.d_surfacePrev);
    m_buffers = ReSTIRBuffers{};
}

void ReSTIRContext::swapHistory() {
    ReSTIRReservoir* r = m_buffers.d_reservoirsCurr;
    m_buffers.d_reservoirsCurr = m_buffers.d_reservoirsPrev;
    m_buffers.d_reservoirsPrev = r;
    ReSTIRSurface* s = m_buffers.d_surfaceCurr;
    m_buffers.d_surfaceCurr = m_buffers.d_surfacePrev;
    m_buffers.d_surfacePrev = s;
    m_buffers.prevWidth  = m_buffers.width;
    m_buffers.prevHeight = m_buffers.height;
    m_buffers.historyValid = true;
}

void ReSTIRContext::invalidateHistory() {
    m_buffers.historyValid = false;
}

void ReSTIRContext::runFrame(
    const DeviceSceneData& scene, const CameraParams& camera,
    uint32_t width, uint32_t height, uint32_t sampleIndex)
{
    if (!m_enabled) return;
    if (!scene.d_lightBVHNodes || !scene.d_areaLights ||
        scene.areaLightCount == 0 || !scene.d_bvhNodes) return;

    launchReSTIRInitialCandidates(
        scene, camera, m_buffers,
        width, height, sampleIndex, m_numCandidates);
    launchReSTIRTemporalReuse(
        scene, m_buffers,
        width, height, sampleIndex, m_temporalMCap);
    launchReSTIRSpatialReuse(
        scene, m_buffers,
        width, height, sampleIndex, m_numNeighbors, m_spatialRadius);
    // Spatial pass's output lives in d_reservoirsSpatial — swap with
    // d_reservoirsCurr so the main kernel reads the post-spatial result.
    ReSTIRReservoir* t = m_buffers.d_reservoirsCurr;
    m_buffers.d_reservoirsCurr    = m_buffers.d_reservoirsSpatial;
    m_buffers.d_reservoirsSpatial = t;
}
