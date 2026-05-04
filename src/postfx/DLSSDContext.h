#pragma once

// Wraps NVIDIA NGX DLSS Ray-Reconstruction (DLSS-RR / DLSSD) for Vulkan.
// Sibling of DLSSContext — RR's eval struct is `NVSDK_NGX_VK_DLSSD_Eval_Params`
// (different from `..._VK_DLSS_Eval_Params`) and consumes a richer set of
// guide buffers (diffuse/specular albedo, normals + packed roughness, specular
// hit distance, view→clip matrices). See external/DLSS/doc/DLSS-RR Integration
// Guide.pdf §3.4 for the full input list.

#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

class DLSSDContext {
public:
    enum QualityMode {
        PERFORMANCE,
        BALANCED,
        QUALITY,
        DLAA,
    };

    DLSSDContext();
    ~DLSSDContext();
    DLSSDContext(const DLSSDContext&) = delete;
    DLSSDContext& operator=(const DLSSDContext&) = delete;

    // Init reuses the same NGX init that DLSSContext does — DLSS-RR is just a
    // different feature ID on the same NGX runtime. Caller may already have
    // brought NGX up via DLSSContext; this class allocates its own parameter
    // block and feature handle, so init() is safe to call alongside DLSS-SR.
    bool init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);

    // Returns the DLSS-RR-suggested render resolution for the given output +
    // quality. Internally calls `NGX_DLSSD_GET_OPTIMAL_SETTINGS` so it picks
    // up RR-specific clamps when the SDK starts to differ from DLSS-SR.
    bool getOptimalRenderResolution(
        uint32_t outputW, uint32_t outputH, QualityMode quality,
        uint32_t& renderW, uint32_t& renderH);

    // Create the DLSSD feature handle. Call once per render+output res tuple.
    bool createFeature(
        VkCommandBuffer cmd,
        uint32_t renderW, uint32_t renderH,
        uint32_t outputW, uint32_t outputH,
        QualityMode quality);

    // Per-frame upscale + denoise.
    //   inColor:        noisy combined HDR linear RGBA16F @ render res
    //   outColor:       upscaled HDR linear RGBA16F      @ output res
    //   motion:         RG16F screen-space pixel deltas   @ render res
    //   depth:          NDC depth (clip.z/clip.w)         @ render res
    //   diffAlbedo:     RGBA8 / RGBA16F unorm-ish albedo @ render res
    //   specAlbedo:     RGBA16F EnvBRDFApprox2           @ render res
    //   normalRoughness: RGBA16F (xyz=worldN, w=roughness packed)
    //   specHitT:       R32F world-space scalar          @ render res
    //   worldToView/viewToClip: float[16] row-major matrices (left-mul)
    void evaluate(
        VkCommandBuffer cmd,
        VkImageView inColor,           VkImage inColorImage,
        VkImageView outColor,          VkImage outColorImage,
        VkImageView motion,            VkImage motionImage,
        VkImageView depth,             VkImage depthImage,
        VkImageView diffAlbedo,        VkImage diffAlbedoImage, VkFormat diffAlbedoFormat,
        VkImageView specAlbedo,        VkImage specAlbedoImage,
        VkImageView normalRoughness,   VkImage normalRoughnessImage,
        VkImageView specHitT,          VkImage specHitTImage,
        VkFormat colorFormat, VkFormat motionFormat, VkFormat depthFormat,
        uint32_t renderW, uint32_t renderH,
        uint32_t outputW, uint32_t outputH,
        float    jitterX, float jitterY,
        float    worldToView[16], float viewToClip[16],
        bool     reset);

    void shutdown();
    bool isValid() const;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};
