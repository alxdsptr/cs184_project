// ── ReSTIR GI: initial-candidates raygen ──────────────────────
// Mirrors src/render/ReSTIRGI.cu's kReSTIRGI_InitCandidates, but uses OptiX
// hardware-traced rays (radiance for primary + indirect, shadow for the
// indirect-bounce NEE). Output layout matches the CUDA kernel's reservoir
// + surface buffers byte-for-byte so the downstream temporal/spatial CUDA
// passes consume either backend's output transparently.

namespace gi_optix {

__device__ inline float giComputeSpecProb(
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

__device__ inline float giDiffusePdf(float NdotL) {
    return fmaxf(NdotL, 0.0f) * (1.0f / M_PI_F);
}

__device__ inline float giSpecularPdfLocal(
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

__device__ inline float giMixturePdfLocal(
    bool pureDiffuse,
    const float3& N, const float3& V, const float3& L,
    float roughness, float specProb)
{
    float diffPdf = giDiffusePdf(dot(N, L));
    if (pureDiffuse) return diffPdf;
    float specPdf = giSpecularPdfLocal(N, V, L, roughness);
    return specProb * specPdf + (1.0f - specProb) * diffPdf;
}

__device__ inline bool giSampleBsdfDir(
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
    outPdf = giMixturePdfLocal(pureDiffuse, s.normal, s.viewDir, dir,
                               s.roughness, specProb);
    return outPdf > 1e-7f;
}

__device__ inline float3 giDirectLightingAtSampleOptiX(
    const DeviceSceneData& scene,
    OptixTraversableHandle handle,
    const float3& pos, const float3& normal,
    const float3& albedo, float roughness, float metallic, bool pureDiffuse,
    const float3& viewDir,
    uint32_t& rng)
{
    if (!scene.d_areaLights || scene.areaLightCount == 0 ||
        !scene.d_lightBVHNodes) return make_float3(0, 0, 0);

    uint32_t slot = 0;
    float    pSelect = 0.0f;
    if (!lightBVH_sample(scene.d_lightBVHNodes, scene.lightBVHRootIndex,
                         pos, pcg32_float(rng), slot, pSelect) ||
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
    float3 toL = lp - pos;
    float  d2  = fmaxf(dot(toL, toL), 1e-6f);
    float  d   = sqrtf(d2);
    float3 L   = toL * (1.0f / d);
    float NdotL = fmaxf(dot(normal, L), 0.0f);
    float lightCos = fmaxf(dot(light.normal, -L), 0.0f);
    if (NdotL <= 0.0f || lightCos <= 0.0f) return make_float3(0, 0, 0);

    float3 origin = pos + normal * 0.001f;
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

    float3 brdf;
    if (pureDiffuse) {
        brdf = albedo * (1.0f / M_PI_F);
    } else {
        ReSTIRSurface tmp{};
        tmp.position    = pos;
        tmp.normal      = normal;
        tmp.albedo      = albedo;
        tmp.roughness   = fmaxf(roughness, 0.04f);
        tmp.metallic    = metallic;
        tmp.viewDir     = viewDir;
        tmp.pureDiffuse = 0u;
        brdf = restirEvalBrdf(tmp, L);
    }

    float pTri  = pSelect;
    float pArea = pTri / fmaxf(light.area, 1e-7f);
    float pdfOmega = pArea * d2 / fmaxf(lightCos, 1e-7f);
    return brdf * Le * trans * (NdotL / fmaxf(pdfOmega, 1e-7f));
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
    bool eligible = primaryHit;
    int  matIdx = -1;
    GPUMaterial mat;
    float3 hitPos = make_float3(0,0,0);
    float3 N      = make_float3(0,1,0);
    float2 uv     = make_float2(0,0);
    float3 albedo = make_float3(1,1,1);

    if (eligible) {
        uint32_t triIdx = rp.primIdx;
        uint32_t i0 = scene.d_indices[triIdx * 3 + 0];
        uint32_t i1 = scene.d_indices[triIdx * 3 + 1];
        uint32_t i2 = scene.d_indices[triIdx * 3 + 2];
        float baryU = rp.baryU, baryV = rp.baryV;
        float baryW = 1.0f - baryU - baryV;
        float3 v0 = scene.d_positions[i0];
        float3 v1 = scene.d_positions[i1];
        float3 v2 = scene.d_positions[i2];
        hitPos = v0 * baryW + v1 * baryU + v2 * baryV;
        matIdx = scene.d_materialIndices ? scene.d_materialIndices[triIdx] : -1;
        if (matIdx < 0 || (uint32_t)matIdx >= scene.materialCount) {
            eligible = false;
        }
        if (eligible) {
            mat = scene.d_materials[matIdx];
            if (scene.d_normals) {
                N = normalize(scene.d_normals[i0] * baryW +
                              scene.d_normals[i1] * baryU +
                              scene.d_normals[i2] * baryV);
            } else {
                N = normalize(cross(v1 - v0, v2 - v0));
            }
            if (dot(N, ray.direction) > 0.0f) N = -N;
            if (scene.d_uvs) {
                uv = scene.d_uvs[i0] * baryW +
                     scene.d_uvs[i1] * baryU +
                     scene.d_uvs[i2] * baryV;
            }
            albedo = mat.albedo;
            if (mat.albedoTex != 0) {
                float4 t = tex2D<float4>(mat.albedoTex, uv.x, uv.y);
                albedo = albedo * make_float3(t.x, t.y, t.z);
            }
            if (mat.metallicRoughTex != 0) {
                float4 mrT = tex2D<float4>(mat.metallicRoughTex, uv.x, uv.y);
                mat.roughness *= mrT.y;
                mat.metallic  *= mrT.z;
            }
        }
    }

    if (eligible) {
        surf.position    = hitPos;
        surf.normal      = N;
        surf.albedo      = albedo;
        surf.roughness   = fmaxf(mat.roughness, 0.04f);
        surf.metallic    = mat.metallic;
        surf.pureDiffuse = mat.pureDiffuse ? 1u : 0u;
        surf.viewDir     = -ray.direction;
        surf.valid       = 1.0f;
        surf.specProb    = gi_optix::giComputeSpecProb(N, surf.viewDir, albedo, mat.metallic);

        float3 clipPrev = mat4_transformPoint(camera.prevViewProjMatrix, hitPos);
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
            if (!gi_optix::giSampleBsdfDir(surf, rng, wi, pdfBsdf)) {
                r.M += 1.0f;   // failed attempt still counts toward |R|
                continue;
            }

            bool   isEnvCand    = false;
            float3 candPos      = make_float3(0,0,0);
            float3 candNormal   = make_float3(0,1,0);
            float3 Lo           = make_float3(0,0,0);
            float  candXrRough  = 0.0f;
            bool   ok           = false;

            float3 sec_origin = hitPos + N * 0.001f;
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
                    candXrRough = 0.0f;
                    ok          = (envLum > 0.0f);
                }
            } else {
                uint32_t t2 = rp2.primIdx;
                uint32_t j0 = scene.d_indices[t2 * 3 + 0];
                uint32_t j1 = scene.d_indices[t2 * 3 + 1];
                uint32_t j2 = scene.d_indices[t2 * 3 + 2];
                float c1 = rp2.baryU, c2 = rp2.baryV;
                float c0 = 1.0f - c1 - c2;
                int matIdx2 = scene.d_materialIndices ? scene.d_materialIndices[t2] : -1;
                if (matIdx2 >= 0 && (uint32_t)matIdx2 < scene.materialCount) {
                    GPUMaterial mat2 = scene.d_materials[matIdx2];
                    float3 v0b = scene.d_positions[j0];
                    float3 v1b = scene.d_positions[j1];
                    float3 v2b = scene.d_positions[j2];
                    float3 sp  = v0b * c0 + v1b * c1 + v2b * c2;
                    float3 N2 = scene.d_normals
                        ? normalize(scene.d_normals[j0] * c0 +
                                    scene.d_normals[j1] * c1 +
                                    scene.d_normals[j2] * c2)
                        : normalize(cross(v1b - v0b, v2b - v0b));
                    if (dot(N2, wi) > 0.0f) N2 = -N2;
                    float2 uv2 = scene.d_uvs
                        ? (scene.d_uvs[j0] * c0 + scene.d_uvs[j1] * c1 + scene.d_uvs[j2] * c2)
                        : make_float2(0.0f, 0.0f);
                    float3 albedo2 = mat2.albedo;
                    if (mat2.albedoTex != 0) {
                        float4 t = tex2D<float4>(mat2.albedoTex, uv2.x, uv2.y);
                        albedo2 = albedo2 * make_float3(t.x, t.y, t.z);
                    }
                    if (mat2.metallicRoughTex != 0) {
                        float4 mrT = tex2D<float4>(mat2.metallicRoughTex, uv2.x, uv2.y);
                        mat2.roughness *= mrT.y;
                        mat2.metallic  *= mrT.z;
                    }
                    float3 emis = mat2.emission * mat2.emissionStrength;
                    if (mat2.emissiveTex != 0) {
                        float4 et = tex2D<float4>(mat2.emissiveTex, uv2.x, uv2.y);
                        emis = make_float3(et.x, et.y, et.z) * mat2.emissionStrength;
                    }
                    float3 viewDir2 = -wi;
                    float3 direct = gi_optix::giDirectLightingAtSampleOptiX(
                        scene, handle, sp, N2, albedo2,
                        fmaxf(mat2.roughness, 0.04f), mat2.metallic,
                        mat2.pureDiffuse != 0, viewDir2, rng);
                    Lo          = emis + direct;
                    candPos     = sp;
                    candNormal  = N2;
                    isEnvCand   = false;
                    candXrRough = fmaxf(mat2.roughness, 0.04f);
                    ok          = (restirLuminance(Lo) > 0.0f);
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
