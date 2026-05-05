// ── ReSTIR PT: initial-candidates raygen ──────────────────────
// Mirrors the CUDA `kReSTIRPT_InitCandidates` kernel from src/render/ReSTIRPT.cu
// but uses OptiX hardware ray tracing (radiance for primary + walk + bounce
// scatter, shadow for the per-vertex NEE). Output reservoir + surface layout
// is byte-for-byte identical to the CUDA kernel's output so the CUDA-side
// temporal/spatial/shade passes consume either backend's output transparently.

namespace pt_optix {

__device__ inline float ptComputeSpecProb(
    const float3& N, const float3& V, const float3& albedo, float metallic)
{
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float t = 1.0f - fminf(fmaxf(NdotV, 0.0f), 1.0f);
    float t5 = t*t*t*t*t;
    float3 F = F0 + (make_float3(1,1,1) - F0) * t5;
    float specW  = restirLuminance(F);
    float3 kd    = (make_float3(1,1,1) - F) * (1.0f - metallic);
    float diffW  = restirLuminance(kd * albedo);
    float p = specW / fmaxf(specW + diffW, 1e-7f);
    return fminf(fmaxf(p, 0.1f), 0.9f);
}

__device__ inline float ptDiffusePdf(float NdotL) {
    return fmaxf(NdotL, 0.0f) * (1.0f / M_PI_F);
}

__device__ inline float ptSpecularPdf(
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

__device__ inline float ptMixturePdf(
    bool pureDiffuse,
    const float3& N, const float3& V, const float3& L,
    float roughness, float specProb)
{
    float diffPdf = ptDiffusePdf(dot(N, L));
    if (pureDiffuse) return diffPdf;
    float specPdf = ptSpecularPdf(N, V, L, roughness);
    return specProb * specPdf + (1.0f - specProb) * diffPdf;
}

__device__ inline ReSTIRSurface ptMakeSurface(
    const float3& pos, const float3& N, const float3& albedo,
    float roughness, float metallic, bool pureDiffuse, const float3& viewDir,
    float specProb)
{
    ReSTIRSurface s{};
    s.position    = pos;
    s.normal      = N;
    s.albedo      = albedo;
    s.roughness   = fmaxf(roughness, 0.04f);
    s.metallic    = metallic;
    s.pureDiffuse = pureDiffuse ? 1u : 0u;
    s.viewDir     = viewDir;
    s.specProb    = specProb;
    s.valid       = 1.0f;
    return s;
}

__device__ inline bool ptSampleBsdfDir(
    const ReSTIRSurface& s, uint32_t& rng,
    float3& outDir, float& outPdf)
{
    bool pureDiffuse = (s.pureDiffuse != 0u);
    float specProb = pureDiffuse ? 0.0f : s.specProb;
    float u = pcg32_float(rng);
    float3 dir;
    if (!pureDiffuse && u < specProb) {
        float a = s.roughness * s.roughness;
        float u1 = pcg32_float(rng);
        float u2 = pcg32_float(rng);
        float cosTheta = sqrtf((1.0f - u1) / (1.0f + (a*a - 1.0f) * u1 + 1e-7f));
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
        float phi = 2.0f * M_PI_F * u2;
        float3 localH = make_float3(sinTheta * cosf(phi), cosTheta, sinTheta * sinf(phi));
        float3 T, B;
        buildONB(s.normal, T, B);
        float3 H = localToWorld(localH, T, s.normal, B);
        float3 inDir = -s.viewDir;
        dir = inDir - H * (2.0f * dot(inDir, H));
        dir = normalize(dir);
    } else {
        float u1 = pcg32_float(rng);
        float u2 = pcg32_float(rng);
        float dummy;
        float3 local = sampleCosineHemisphere(u1, u2, dummy);
        float3 T, B;
        buildONB(s.normal, T, B);
        dir = localToWorld(local, T, s.normal, B);
    }
    if (dot(s.normal, dir) <= 1e-6f) return false;
    outDir = dir;
    outPdf = ptMixturePdf(pureDiffuse, s.normal, s.viewDir, dir,
                          s.roughness, specProb);
    return outPdf > 1e-7f;
}

// One NEE shadow-ray bounce at the given vertex via the OptiX shadow SBT.
__device__ inline float3 ptDirectLightingAtVertexOptiX(
    const DeviceSceneData& scene,
    OptixTraversableHandle handle,
    const ReSTIRSurface& s,
    uint32_t& rng)
{
    if (!scene.d_areaLights || scene.areaLightCount == 0 ||
        !scene.d_lightBVHNodes) return make_float3(0, 0, 0);

    uint32_t slot = 0;
    float    pSelect = 0.0f;
    if (!lightBVH_sample(scene.d_lightBVHNodes, scene.lightBVHRootIndex,
                         s.position, pcg32_float(rng), slot, pSelect) ||
        !(pSelect > 0.0f))
        return make_float3(0, 0, 0);
    uint32_t lightIdx = scene.d_lightOrderedIndices[slot];
    GPUAreaLight light = scene.d_areaLights[lightIdx];

    float r1 = pcg32_float(rng);
    float r2 = pcg32_float(rng);
    float su = sqrtf(r1);
    float b0 = 1.0f - su;
    float b1 = su * (1.0f - r2);
    float b2 = su * r2;
    float3 lp = light.v0 * b0 + (light.v0 + light.e1) * b1 + (light.v0 + light.e2) * b2;
    float3 toL = lp - s.position;
    float  d2  = fmaxf(dot(toL, toL), 1e-6f);
    float  d   = sqrtf(d2);
    float3 L   = toL * (1.0f / d);
    float NdotL = fmaxf(dot(s.normal, L), 0.0f);
    float lightCos = fmaxf(dot(light.normal, -L), 0.0f);
    if (NdotL <= 0.0f || lightCos <= 0.0f) return make_float3(0, 0, 0);

    float3 origin = s.position + s.normal * 0.001f;
    float tmax = fmaxf(d - 0.002f, 0.001f);
    float3 trans = traceShadowRay(handle, origin, L, 1e-3f, tmax);
    if (restirLuminance(trans) <= 1e-6f) return make_float3(0, 0, 0);

    float3 Le;
    if (light.emissiveTex == 0) {
        Le = light.emission;
    } else {
        float texU = light.uv0.x * b0 + light.uv1.x * b1 + light.uv2.x * b2;
        float texV = light.uv0.y * b0 + light.uv1.y * b1 + light.uv2.y * b2;
        float4 et = tex2D<float4>(light.emissiveTex, texU, texV);
        Le = make_float3(et.x, et.y, et.z) * light.emission;
    }

    float3 brdf = restirEvalBrdf(s, L);
    float pTri  = pSelect;
    float pArea = pTri / fmaxf(light.area, 1e-7f);
    float pdfOmega = pArea * d2 / fmaxf(lightCos, 1e-7f);
    float3 Li = brdf * Le * trans * (NdotL / fmaxf(pdfOmega, 1e-7f));
    // Source-side firefly clamp — see render/ReSTIRPT.cu and
    // OptiXProgramsGI.inl for the M7 flash-and-decay rationale.
    // Match the CUDA kernel's PT-tightened cap (25, vs GI's 50).
    float lumLi = restirLuminance(Li);
    const float liCap = 25.0f;
    if (lumLi > liCap) Li = Li * (liCap / lumLi);
    return Li;
}

// Resolve a hit record (from rp = traceRadianceRay) into shading attributes.
__device__ inline bool ptShadeHitOptiX(
    const DeviceSceneData& scene,
    const RadiancePayload& rp,
    const float3& rayDir,
    float3& outPos, float3& outN, float3& outAlbedo, float3& outEmission,
    float& outRoughness, float& outMetallic, bool& outPureDiffuse)
{
    if (rp.hit == 0) return false;
    int matIdx = scene.d_materialIndices ? scene.d_materialIndices[rp.primIdx] : -1;
    if (matIdx < 0 || (uint32_t)matIdx >= scene.materialCount) return false;
    GPUMaterial mat = scene.d_materials[matIdx];
    uint32_t i0 = scene.d_indices[rp.primIdx * 3 + 0];
    uint32_t i1 = scene.d_indices[rp.primIdx * 3 + 1];
    uint32_t i2 = scene.d_indices[rp.primIdx * 3 + 2];
    float c1 = rp.baryU, c2 = rp.baryV;
    float c0 = 1.0f - c1 - c2;
    float3 v0b = scene.d_positions[i0];
    float3 v1b = scene.d_positions[i1];
    float3 v2b = scene.d_positions[i2];
    outPos = v0b * c0 + v1b * c1 + v2b * c2;
    float3 N = scene.d_normals
        ? normalize(scene.d_normals[i0] * c0 +
                    scene.d_normals[i1] * c1 +
                    scene.d_normals[i2] * c2)
        : normalize(cross(v1b - v0b, v2b - v0b));
    if (dot(N, rayDir) > 0.0f) N = -N;
    outN = N;
    float2 uv = scene.d_uvs
        ? (scene.d_uvs[i0] * c0 + scene.d_uvs[i1] * c1 + scene.d_uvs[i2] * c2)
        : make_float2(0.0f, 0.0f);
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
    float3 emis = mat.emission * mat.emissionStrength;
    if (mat.emissiveTex != 0) {
        float4 et = tex2D<float4>(mat.emissiveTex, uv.x, uv.y);
        emis = make_float3(et.x, et.y, et.z) * mat.emissionStrength;
    }
    outAlbedo   = albedo;
    outEmission = emis;
    outRoughness = fmaxf(mat.roughness, 0.04f);
    outMetallic  = mat.metallic;
    outPureDiffuse = (mat.pureDiffuse != 0);
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
    float specProb_xr = ptComputeSpecProb(xrN, viewDir, xrAlbedo, xrMetallic);
    ReSTIRSurface curr = ptMakeSurface(xrPos, xrN, xrAlbedo,
                                        xrRoughness, xrMetallic, xrPureDiffuse,
                                        viewDir, specProb_xr);
    L = L + ptDirectLightingAtVertexOptiX(scene, handle, curr, rng);

    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

    for (uint32_t i = 0; i < bounces; i++) {
        float3 wi;
        float  pdfBsdf = 0.0f;
        if (!ptSampleBsdfDir(curr, rng, wi, pdfBsdf)) break;

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
                float envLum = restirLuminance(envColor);
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
        float specProb = ptComputeSpecProb(hN, nViewDir, hAlbedo, hMetallic);
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

    float lum = restirLuminance(L);
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
    surf.specProb    = pt_optix::ptComputeSpecProb(hN, surf.viewDir, hAlbedo, hMetallic);

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
        if (!pt_optix::ptSampleBsdfDir(surf, rng, wi, pdfBsdf)) {
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
                float envLum = restirLuminance(envColor);
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
                ok          = (restirLuminance(Lo) > 0.0f);
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
