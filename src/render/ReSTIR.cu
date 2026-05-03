#include "render/ReSTIR.h"
#include "render/ReSTIRDevice.cuh"  // shared target-pdf + reservoir primitives
#include "backend/RayTracingBackend.h"
#include "core/Math.h"
#include "gpu/AreaLightGPU.h"
#include "gpu/MaterialGPU.h"
#include "gpu/Random.h"
#include "accel/BVH.h"
#include "accel/LightBVHSample.h"
#include "util/CudaCheck.h"

#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────
// ReSTIR DI — primary-hit only, biased spatial reuse.
//
// The whole pipeline for one frame runs in three kernels below. Each pixel
// owns a ReSTIRReservoir that holds *one* light sample along with the
// probability-weighting needed to turn it into an unbiased estimator:
//
//   E[ f / target * W ]  =  integral( f )                       [RIS]
//
// where target(x) = restirLuminance(Le) * |f_r(wi)| * G(x,y) is evaluated without
// visibility — shadow ray is cast once at final shading.
// ─────────────────────────────────────────────────────────────────────────

// (BRDF + target-pdf helpers live in ReSTIRDevice.cuh so OptiX raygen can
// share them verbatim.)

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

// (Reservoir primitives live in ReSTIRDevice.cuh.)

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

    ReSTIRReservoir r; restir_reservoirReset(r);
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
            float specW = restirLuminance(F);
            float3 kd = (make_float3(1,1,1) - F) * (1.0f - surf.metallic);
            float diffW = restirLuminance(kd * surf.albedo);
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
            float pHat = restirEvalTargetPdf(surf, light, cb1, cb2);
            float wCand = (areaPdf > 0.0f) ? (pHat / areaPdf) : 0.0f;

            restir_reservoirUpdate(r, wSum, lightIdx, cb1, cb2, pHat,
                             wCand, pcg32_float(rng));
        }
        restir_reservoirFinalize(r, wSum);
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
    float pHatAtCurr = restirEvalTargetPdf(s, light, pr.baryB1, pr.baryB2);

    // Rebuild wSum for the current reservoir so we can continue streaming.
    // Before combine: wSum = r.pHat * r.M * r.W  (inverse of finalize).
    float wSum = r.pHat * r.M * r.W;
    restir_reservoirCombine(r, wSum, pr, pHatAtCurr, pcg32_float(rng));
    restir_reservoirFinalize(r, wSum);

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
        float pHatAtCurr = restirEvalTargetPdf(s, light, nr.baryB1, nr.baryB2);
        restir_reservoirCombine(r, wSum, nr, pHatAtCurr, pcg32_float(rng));
    }
    restir_reservoirFinalize(r, wSum);
    outRes[pixelIdx] = r;
}

// ── Kernel 1.5: visibility reuse ────────────────────────────────
// Bitterli 2020 Alg. 5 lines 6-9: trace one shadow ray per pixel toward the
// held sample. If occluded, zero W so spatial/temporal reuse won't propagate
// the occluded sample to neighbors. This keeps RIS from being dominated by
// visibility noise once M grows large (paper §3.2 "Visibility Reuse").
__global__ void kReSTIR_Visibility(
    DeviceSceneData scene,
    ReSTIRReservoir* reservoirs,
    const ReSTIRSurface* surfaces,
    uint32_t width, uint32_t height)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t pixelIdx = y * width + x;

    ReSTIRReservoir r = reservoirs[pixelIdx];
    if (r.lightIndex == 0xFFFFFFFFu || r.W <= 0.0f) return;

    ReSTIRSurface s = surfaces[pixelIdx];
    if (s.valid < 0.5f) return;

    GPUAreaLight light = scene.d_areaLights[r.lightIndex];
    float b0 = 1.0f - r.baryB1 - r.baryB2;
    float3 pOnLight = light.v0 * b0 + (light.v0 + light.e1) * r.baryB1
                                    + (light.v0 + light.e2) * r.baryB2;
    // Origin nudged off the surface along its normal to avoid self-hits;
    // bvh_anyHit subtracts a tmax epsilon so the light triangle itself
    // doesn't register as an occluder.
    float3 origin = s.position + s.normal * 1e-3f;
    bool occluded = bvh_anyHit(origin, pOnLight,
                               scene.d_bvhNodes, scene.bvhRootIndex,
                               scene.d_positions, scene.d_indices);
    if (occluded) {
        r.W = 0.0f;
        reservoirs[pixelIdx] = r;
    }
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

void launchReSTIRVisibilityReuse(
    const DeviceSceneData& scene,
    ReSTIRBuffers          buffers,
    uint32_t               width,
    uint32_t               height)
{
    // No CUDA-traversable BVH (e.g. OptiX backend without patch) — skip.
    // Without visibility reuse the algorithm is still correct, just noisier
    // in heavily-occluded scenes.
    if (!scene.d_bvhNodes || scene.totalTriangles == 0) return;
    dim3 block(8, 8);
    dim3 grid = makeGrid(width, height, block);
    kReSTIR_Visibility<<<grid, block>>>(
        scene,
        buffers.d_reservoirsCurr,
        buffers.d_surfaceCurr,
        width, height);
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

bool ReSTIRContext::runFrame(
    const DeviceSceneData& scene, const CameraParams& camera,
    uint32_t width, uint32_t height, uint32_t sampleIndex,
    RayTracingBackend* backend)
{
    if (!m_enabled) return false;
    if (!scene.d_lightBVHNodes || !scene.d_areaLights ||
        scene.areaLightCount == 0) return false;

    // Initial-candidates pass: prefer the backend's native implementation
    // (OptiX raygen → GAS) when available; otherwise fall back to the CUDA
    // kernel, which needs scene.d_bvhNodes populated by backend->patchScene.
    bool initRan = false;
    if (backend) {
        initRan = backend->runReSTIRInitCandidates(
            scene, camera,
            (void*)m_buffers.d_reservoirsCurr,
            (void*)m_buffers.d_surfaceCurr,
            width, height, sampleIndex, m_numCandidates);
    }
    if (!initRan) {
        if (!scene.d_bvhNodes) return false;
        launchReSTIRInitialCandidates(
            scene, camera, m_buffers,
            width, height, sampleIndex, m_numCandidates);
    }

    // Visibility reuse: kill occluded samples before they leak through
    // spatial/temporal reuse. Bitterli 2020 Alg. 5. Prefer the backend's
    // native implementation (OptiX raygen → GAS, hardware-accelerated)
    // when available; fall back to the CUDA SAH-BVH kernel otherwise.
    bool visRan = false;
    if (backend) {
        visRan = backend->runReSTIRVisibilityReuse(
            scene,
            (void*)m_buffers.d_reservoirsCurr,
            (const void*)m_buffers.d_surfaceCurr,
            width, height);
    }
    if (!visRan) {
        launchReSTIRVisibilityReuse(scene, m_buffers, width, height);
    }

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
    return true;
}
