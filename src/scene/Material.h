#pragma once
#include <string>
#include <cuda_runtime.h>

struct PBRMaterial {
    float3 albedo      = {0.8f, 0.8f, 0.8f};
    float  roughness   = 0.5f;
    float  metallic    = 0.0f;
    float3 emission    = {0.0f, 0.0f, 0.0f};
    float  emissionStrength = 0.0f;
    float  ior         = 1.5f;
    float  transmission = 0.0f;

    // Set for materials that should be rendered as a pure Lambertian diffuse
    // BRDF — bypasses the Cook-Torrance specular lobe entirely (including the
    // F0 = 0.04 dielectric term). Used for legacy Collada Phong materials with
    // negligible specular, to match classic path tracers that only read the
    // <diffuse> term.
    bool   pureDiffuse = false;

    // ── Specular-Glossiness workflow (legacy FBX, e.g. MEASURE_SEVEN). ──
    // When enabled, the kernel reads F0 from `specularColor` (× specularTex.rgb
    // if bound) and roughness from `1 - glossiness` (× specularTex.a if bound),
    // bypassing the metallic-roughness `lerp(0.04, albedo, metallic)` path.
    bool   useSpecularGlossiness = false;
    float3 specularColor = {1.0f, 1.0f, 1.0f}; // F0 multiplier (linear)
    float  glossiness    = 0.5f;               // 1 - roughness multiplier
    // True when the spec/gloss texture's alpha channel actually carries
    // glossiness data (variance > noise floor). When false the kernel uses
    // only the scalar glossiness factor and does not multiply by tex.a — this
    // prevents 3-channel specular maps (alpha implicitly 1) from collapsing
    // every shaded pixel to a perfect mirror.
    bool   specularGlossAlphaIsGlossiness = false;

    std::string albedoTexPath;
    std::string normalTexPath;
    std::string metallicRoughTexPath;
    std::string emissiveTexPath;
    std::string specularGlossTexPath; // RGB = F0 color, A = glossiness

    // Runtime CUDA texture handles (0 means no texture bound).
    cudaTextureObject_t albedoTexObj = 0;
    cudaTextureObject_t normalTexObj = 0;
    cudaTextureObject_t metallicRoughTexObj = 0;
    cudaTextureObject_t emissiveTexObj = 0;
    cudaTextureObject_t specularGlossTexObj = 0;
};
