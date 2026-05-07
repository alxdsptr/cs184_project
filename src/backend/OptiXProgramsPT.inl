// ── ReSTIR PT: initial-candidates raygen ──────────────────────
// Mirrors the CUDA `kReSTIRPT_InitCandidates` kernel from src/render/ReSTIRPT.cu
// but uses OptiX hardware ray tracing (radiance for primary + walk + bounce
// scatter, shadow for the per-vertex NEE). Output reservoir + surface layout
// is byte-for-byte identical to the CUDA kernel's output so the CUDA-side
// temporal/spatial/shade passes consume either backend's output transparently.

namespace pt_optix {

// One NEE shadow-ray bounce at the given vertex via the OptiX shadow SBT.
// Glass-aware (traceShadowRay returns float3 attenuation through the
// __anyhit__shadow program). Source-side firefly cap matches the CUDA PT
// (25, tighter than GI's 50 due to throughput stacking through the
// postfix random walk).
__device__ inline float3 ptDirectLightingAtVertexOptiX(
    const DeviceSceneData& scene,
    OptixTraversableHandle handle,
    const ReSTIRSurface& s,
    uint32_t& rng)
{
    auto traceShadow = [&](float3 origin, float3 dir, float dist) -> float3 {
        float tmax = (dist >= 1.0e29f) ? 1.0e30f : fmaxf(dist - 0.002f, 0.001f);
        return traceShadowRay(handle, origin, dir, 1.0e-3f, tmax);
    };
    float3 Li = restirAreaLightNEE(scene, s, rng, traceShadow, /*fireflyClamp=*/25.0f);
    Li = Li + restirDirectionalLightsNEE(scene, s, traceShadow);
    return Li;
}

// Resolve a hit record (from rp = traceRadianceRay) into shading attributes
// via the canonical restirDecodeHit (ReSTIRDevice.cuh).
__device__ inline bool ptShadeHitOptiX(
    const DeviceSceneData& scene,
    const RadiancePayload& rp,
    const float3& rayDir,
    float3& outPos, float3& outN, float3& outAlbedo, float3& outEmission,
    float& outRoughness, float& outMetallic, bool& outPureDiffuse)
{
    if (rp.hit == 0) return false;
    ReSTIRHitDecode h = restirDecodeHit(scene, rp.primIdx, rp.baryU, rp.baryV, rayDir);
    if (!h.valid) return false;
    outPos         = h.pos;
    outN           = h.normal;
    outAlbedo      = h.albedo;
    outEmission    = h.emission;
    outRoughness   = fmaxf(h.mat.roughness, 0.04f);
    outMetallic    = h.mat.metallic;
    outPureDiffuse = h.pureDiffuse;
    return true;
}

// OptiX twin of ptPathPostfix. Same algorithm (NEE at every vertex past x_r,
// Russian roulette after bounce 1, firefly clamp on the total).
__device__ inline float3 ptPathPostfixOptiX(
    const DeviceSceneData& scene,
    OptixTraversableHandle handle,
    const float3& xrPos, const float3& xrN,
    const float3& xrAlbedo, const float3& xrEmis,
    float xrRoughness, float xrMetallic, bool xrPureDiffuse,
    const float3& viewDir,
    bool  enableEnvironment,
    uint32_t bounces,
    uint32_t& rng)
{
    float3 L = xrEmis;
    float specProb_xr = computeSpecProb(xrN, viewDir, xrAlbedo, xrMetallic);
    ReSTIRSurface curr = ptMakeSurface(xrPos, xrN, xrAlbedo,
                                        xrRoughness, xrMetallic, xrPureDiffuse,
                                        viewDir, specProb_xr);
    L = L + ptDirectLightingAtVertexOptiX(scene, handle, curr, rng);

    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    for (uint32_t i = 0; i < bounces; i++) {
        float3 wi;
        float  pdfBsdf = 0.0f;
        if (!restirSampleBsdfDir(curr, rng, wi, pdfBsdf)) break;

        float3 brdf = restirEvalBrdf(curr, wi);
        float NdotL = fmaxf(dot(curr.normal, wi), 0.0f);
        if (NdotL <= 0.0f) break;
        float3 weight = brdf * (NdotL / pdfBsdf);
        throughput = throughput * weight;

        float3 origin = curr.position + curr.normal * 0.001f;
        RadiancePayload rp = traceRadianceRay(handle, origin, wi, 1e-3f, 1e30f);
        if (rp.hit == 0) {
            if (enableEnvironment) {
                float3 envColor = sampleEnvironment(wi, scene.envMapTex);
                float envLum = luminance(envColor);
                const float clampLum = 100.0f;
                if (envLum > clampLum) envColor = envColor * (clampLum / envLum);
                L = L + throughput * envColor;
            }
            break;
        }

        float3 hPos, hN, hAlbedo, hEmis;
        float  hRoughness, hMetallic;
        bool   hPure;
        if (!ptShadeHitOptiX(scene, rp, wi,
                             hPos, hN, hAlbedo, hEmis,
                             hRoughness, hMetallic, hPure)) break;
        L = L + throughput * hEmis;

        float3 nViewDir = -wi;
        float specProb = computeSpecProb(hN, nViewDir, hAlbedo, hMetallic);
        curr = ptMakeSurface(hPos, hN, hAlbedo, hRoughness, hMetallic, hPure,
                             nViewDir, specProb);

        L = L + throughput * ptDirectLightingAtVertexOptiX(scene, handle, curr, rng);

        if (i >= 1) {
            float maxC = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            float pCont = fminf(fmaxf(maxC, 0.05f), 0.95f);
            if (pcg32_float(rng) > pCont) break;
            throughput = throughput * (1.0f / pCont);
        }
    }

    float lum = luminance(L);
    const float clampMax = 200.0f;
    if (lum > clampMax) L = L * (clampMax / lum);
    return L;
}

} // namespace pt_optix

extern "C" __global__ void __raygen__restir_pt_init_candidates()
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
    uint32_t pathLength    = params.ptPathLength;

    GIReservoir r; giReservoirReset(r);
    ReSTIRSurface surf{}; surf.valid = 0.0f;

    // Mix camera.frameIndex into salt so canonical sample changes every
    // frame even when sampleIndex is pinned to 0 by camera motion.
    uint32_t seedSalt = params.sampleIndex + camera.frameIndex * 0x9E3779B9u;
    uint32_t rng = pcg32_seed(pixelIdx * 0x9E3779B1u + seedSalt,
                              seedSalt * 0x85EBCA6Bu + 0xB7u);

    float jx = camera.jitterOffset.x;
    float jy = camera.jitterOffset.y;
    Ray ray = generateRay(x, y, params.width, params.height, camera, jx, jy);

    RadiancePayload rp = traceRadianceRay(
        handle, ray.origin, ray.direction, ray.tmin, ray.tmax);

    if (rp.hit == 0) {
        if (params.ptReservoirsCurr) params.ptReservoirsCurr[pixelIdx] = r;
        if (params.ptSurfacesCurr)   params.ptSurfacesCurr[pixelIdx]   = surf;
        return;
    }

    float3 hPos, hN, hAlbedo, hEmis;
    float  hRoughness, hMetallic;
    bool   hPure;
    if (!pt_optix::ptShadeHitOptiX(scene, rp, ray.direction,
                                    hPos, hN, hAlbedo, hEmis,
                                    hRoughness, hMetallic, hPure)) {
        if (params.ptReservoirsCurr) params.ptReservoirsCurr[pixelIdx] = r;
        if (params.ptSurfacesCurr)   params.ptSurfacesCurr[pixelIdx]   = surf;
        return;
    }

    surf.position    = hPos;
    surf.normal      = hN;
    surf.albedo      = hAlbedo;
    surf.roughness   = hRoughness;
    surf.metallic    = hMetallic;
    surf.pureDiffuse = hPure ? 1u : 0u;
    surf.viewDir     = -ray.direction;
    surf.valid       = 1.0f;
    surf.specProb    = computeSpecProb(hN, surf.viewDir, hAlbedo, hMetallic);

    float3 hPosPrev = hPos;
    if (scene.d_positionsPrev) {
        uint32_t triIdx = rp.primIdx;
        uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
        uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
        uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
        float bU = rp.baryU, bV = rp.baryV, bW = 1.0f - bU - bV;
        hPosPrev = scene.d_positionsPrev[i0] * bW
                 + scene.d_positionsPrev[i1] * bU
                 + scene.d_positionsPrev[i2] * bV;
    }
    float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hPosPrev);
    surf.prevPixel  = make_float2((clipPrev.x + 1.0f) * 0.5f * (float)params.width,
                                   (1.0f - clipPrev.y) * 0.5f * (float)params.height);

    // ── Generate `numCandidates` independent paths and stream into RIS ──
    // Mirrors the CUDA kernel (render/ReSTIR PT.cu kReSTIRPT_InitCandidates):
    // each candidate samples a BSDF direction, traces secondary + postfix
    // walk to gather Lo, then streams into the canonical reservoir via
    // gris_streamCandidate (paper §4.1 RIS, Eq. 5). M is bumped on every
    // attempt (including failed ones) so the convergence proof of §5.7 holds.
    //
    // This used to be a single-candidate loop, which made OptiX's ReSTIR PT
    // visibly noisier than the CUDA path's. Multi-candidate restores parity.
    uint32_t numCandidates = params.ptNumCandidates;
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

        bool   isEnvCand   = false;
        float3 candPos     = make_float3(0,0,0);
        float3 candNormal  = make_float3(0,1,0);
        float3 Lo          = make_float3(0,0,0);
        float  candXrRough = 0.0f;
        bool   ok          = false;

        float3 sec_origin = hPos + hN * 0.001f;
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
                candXrRough = 0.0f;          // env: roughness gate disabled
                ok          = (envLum > 0.0f);
            }
        } else {
            float3 xPos, xN, xAlbedo, xEmis;
            float  xRoughness, xMetallic;
            bool   xPure;
            if (pt_optix::ptShadeHitOptiX(scene, rp2, wi,
                                           xPos, xN, xAlbedo, xEmis,
                                           xRoughness, xMetallic, xPure)) {
                float3 viewAtXr = -wi;
                Lo = pt_optix::ptPathPostfixOptiX(scene, handle,
                                                   xPos, xN, xAlbedo, xEmis,
                                                   xRoughness, xMetallic, xPure,
                                                   viewAtXr,
                                                   enableEnvironment,
                                                   pathLength,
                                                   rng);
                candPos     = xPos;
                candNormal  = xN;
                isEnvCand   = false;
                candXrRough = xRoughness;
                ok          = (luminance(Lo) > 0.0f);
            }
        }

        // Evaluate target p̂ at the visible surface.
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

    if (params.ptReservoirsCurr) params.ptReservoirsCurr[pixelIdx] = r;
    if (params.ptSurfacesCurr)   params.ptSurfacesCurr[pixelIdx]   = surf;
}
