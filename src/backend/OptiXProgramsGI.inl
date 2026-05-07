// ── ReSTIR GI: initial-candidates raygen ──────────────────────
// Mirrors src/render/ReSTIRGI.cu's kReSTIRGI_InitCandidates, but uses OptiX
// hardware-traced rays (radiance for primary + indirect, shadow for the
// indirect-bounce NEE). Output layout matches the CUDA kernel's reservoir
// + surface buffers byte-for-byte so the downstream temporal/spatial CUDA
// passes consume either backend's output transparently.

namespace gi_optix {

__device__ inline float3 giDirectLightingAtSampleOptiX(
    const DeviceSceneData& scene,
    OptixTraversableHandle handle,
    const float3& pos, const float3& normal,
    const float3& albedo, float roughness, float metallic, bool pureDiffuse,
    const float3& viewDir,
    uint32_t& rng)
{
    // Pack the raw shading attributes into a temporary ReSTIRSurface so
    // restirEvalBrdf sees the same fields used at pHat / reservoir-merge time.
    ReSTIRSurface tmp = ptMakeSurface(pos, normal, albedo,
                                       fmaxf(roughness, 0.04f), metallic,
                                       pureDiffuse, viewDir, /*specProb=*/0.0f);

    // Glass-aware shadow trace via the OptiX __anyhit__shadow program.
    auto traceShadow = [&](float3 origin, float3 dir, float dist) -> float3 {
        float tmax = (dist >= 1.0e29f) ? 1.0e30f : fmaxf(dist - 0.002f, 0.001f);
        return traceShadowRay(handle, origin, dir, 1.0e-3f, tmax);
    };
    // M7 with 9759 small emissive triangles produces grazing NEE samples whose
    // 1/pdfOmega term spikes Li — without the source-side cap the bright Lo
    // gets stored in the GI reservoir's `sampleRadiance` and persists for
    // ~mCap frames. 50 matches the CUDA kernel; PT uses 25 (tighter, since
    // the postfix random walk amplifies Li through more throughput stages).
    float3 Li = restirAreaLightNEE(scene, tmp, rng, traceShadow, /*fireflyClamp=*/50.0f);
    Li = Li + restirDirectionalLightsNEE(scene, tmp, traceShadow);
    return Li;
}

} // namespace gi_optix

extern "C" __global__ void __raygen__restir_gi_init_candidates()
{
    uint3 idx = optixGetLaunchIndex();
    uint32_t x = idx.x;
    uint32_t y = idx.y;
    if (x >= params.width || y >= params.height) return;
    uint32_t pixelIdx = y * params.width + x;

    const DeviceSceneData& scene  = params.scene;
    const CameraParams&    camera = params.camera;
    OptixTraversableHandle handle = params.handle;
    bool enableEnvironment = (params.giEnableEnvironment != 0);

    GIReservoir r; giReservoirReset(r);
    ReSTIRSurface surf{};
    surf.valid = 0.0f;

    // Mix camera.frameIndex into salt so canonical sample changes every
    // frame even when sampleIndex is pinned to 0 by camera motion.
    uint32_t seedSalt = params.sampleIndex + camera.frameIndex * 0x9E3779B9u;
    uint32_t rng = pcg32_seed(pixelIdx * 0x517CC1B7u + seedSalt,
                              seedSalt * 0xCAFEF00Du + 0x67u);

    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;
    Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

    RadiancePayload rp = traceRadianceRay(
        handle, ray.origin, ray.direction, ray.tmin, ray.tmax);

    bool primaryHit = (rp.hit != 0);
    ReSTIRHitDecode hPrim{};
    if (primaryHit) {
        hPrim = restirDecodeHit(scene, rp.primIdx, rp.baryU, rp.baryV, ray.direction);
    }
    bool eligible = hPrim.valid;

    if (eligible) {
        surf.position    = hPrim.pos;
        surf.normal      = hPrim.normal;
        surf.albedo      = hPrim.albedo;
        surf.roughness   = fmaxf(hPrim.mat.roughness, 0.04f);
        surf.metallic    = hPrim.mat.metallic;
        surf.pureDiffuse = hPrim.pureDiffuse ? 1u : 0u;
        surf.viewDir     = -ray.direction;
        surf.valid       = 1.0f;
        surf.specProb    = computeSpecProb(hPrim.normal, surf.viewDir, hPrim.albedo, hPrim.mat.metallic);

        float3 hitPosPrevGI = hPrim.pos;
        if (scene.d_positionsPrev) {
            uint32_t triIdx2 = rp.primIdx;
            uint32_t pi0 = scene.d_indices[triIdx2 * 3 + 0];
            uint32_t pi1 = scene.d_indices[triIdx2 * 3 + 1];
            uint32_t pi2 = scene.d_indices[triIdx2 * 3 + 2];
            float bU = rp.baryU, bV = rp.baryV, bW = 1.0f - bU - bV;
            hitPosPrevGI = scene.d_positionsPrev[pi0] * bW
                         + scene.d_positionsPrev[pi1] * bU
                         + scene.d_positionsPrev[pi2] * bV;
        }
        float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hitPosPrevGI);
        surf.prevPixel  = make_float2((clipPrev.x + 1.0f) * 0.5f * (float)params.width,
                                       (1.0f - clipPrev.y) * 0.5f * (float)params.height);

        // ── Generate `numCandidates` independent paths and stream into RIS ──
        // Mirrors the CUDA kernel (render/ReSTIRGI.cu) so OptiX produces
        // visually equivalent reservoirs. Single-candidate degenerates ReSTIR
        // to "1-bounce path tracing with reservoir reuse" — visibly noisier
        // than the CUDA path. The loop here restores parity.
        uint32_t numCandidates = params.giNumCandidates;
        if (numCandidates < 1) numCandidates = 1;
        float wSum = 0.0f;
        float xrRoughnessKept = 1.0f;

        for (uint32_t k = 0; k < numCandidates; k++) {
            float3 wi;
            float  pdfBsdf = 0.0f;
            if (!restirSampleBsdfDir(surf, rng, wi, pdfBsdf)) {
                r.M += 1.0f;   // failed attempt still counts toward |R|
                continue;
            }

            bool   isEnvCand    = false;
            float3 candPos      = make_float3(0,0,0);
            float3 candNormal   = make_float3(0,1,0);
            float3 Lo           = make_float3(0,0,0);
            float  candXrRough  = 0.0f;
            bool   ok           = false;

            float3 sec_origin = hPrim.pos + hPrim.normal * 0.001f;
            RadiancePayload rp2 = traceRadianceRay(
                handle, sec_origin, wi, 1e-3f, 1e30f);
            if (rp2.hit == 0) {
                if (enableEnvironment) {
                    float3 envColor = sampleEnvironment(wi, scene.envMapTex);
                    float envLum = luminance(envColor);
                    const float clampLum = 100.0f;
                    if (envLum > clampLum) envColor = envColor * (clampLum / envLum);
                    isEnvCand   = true;
                    candPos     = wi;
                    candNormal  = -wi;
                    Lo          = envColor;
                    candXrRough = 0.0f;
                    ok          = (envLum > 0.0f);
                }
            } else {
                ReSTIRHitDecode hSec = restirDecodeHit(
                    scene, rp2.primIdx, rp2.baryU, rp2.baryV, wi);
                if (hSec.valid) {
                    float3 viewDir2 = -wi;
                    float3 direct = gi_optix::giDirectLightingAtSampleOptiX(
                        scene, handle, hSec.pos, hSec.normal, hSec.albedo,
                        fmaxf(hSec.mat.roughness, 0.04f), hSec.mat.metallic,
                        hSec.pureDiffuse, viewDir2, rng);
                    Lo          = hSec.emission + direct;
                    candPos     = hSec.pos;
                    candNormal  = hSec.normal;
                    isEnvCand   = false;
                    candXrRough = fmaxf(hSec.mat.roughness, 0.04f);
                    ok          = (luminance(Lo) > 0.0f);
                }
            }

            float pHat = 0.0f;
            if (ok) {
                GIReservoir candR{};
                candR.visiblePos     = surf.position;
                candR.visibleNormal  = surf.normal;
                candR.samplePos      = candPos;
                candR.sampleNormal   = candNormal;
                candR.sampleRadiance = Lo;
                candR.isEnv          = isEnvCand ? 1u : 0u;
                candR.valid          = 1u;
                candR.xrRoughness    = candXrRough;
                float3 wiOut;
                pHat = giEvalTargetPdf(surf, candR, wiOut);
            }

            bool replaced = gris_streamCandidate(
                r, wSum,
                surf.position, surf.normal,
                isEnvCand, candPos, candNormal, Lo,
                pHat, pdfBsdf, pcg32_float(rng));
            if (replaced) xrRoughnessKept = candXrRough;
        }

        if (r.valid) {
            gris_cHat(r) = r.pHat;
            r.xrRoughness = xrRoughnessKept;
        }
        giReservoirFinalize(r, wSum);
    }

    if (params.giReservoirsCurr) params.giReservoirsCurr[pixelIdx] = r;
    if (params.giSurfacesCurr)   params.giSurfacesCurr[pixelIdx]   = surf;
}
