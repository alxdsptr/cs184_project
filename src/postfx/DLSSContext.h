#pragma once

// Wraps NVIDIA NGX DLSS-SR for Vulkan. Pre-device-creation helpers expose the
// required VkInstance/VkDevice extensions; post-device methods init NGX, query
// optimal render resolution for a quality mode, and evaluate the upscaler.

#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

class DLSSContext {
public:
    enum QualityMode {
        PERFORMANCE,
        BALANCED,
        QUALITY,
        DLAA,
    };

    DLSSContext();
    ~DLSSContext();
    DLSSContext(const DLSSContext&) = delete;
    DLSSContext& operator=(const DLSSContext&) = delete;

    // Static: call before VkInstance / VkDevice creation. Populates the
    // extension lists NGX requires. Returns false if the NGX DLL couldn't be
    // probed — in which case the caller should skip NGX init.
    static bool queryRequiredExtensions(
        std::vector<const char*>& instanceExts,
        std::vector<const char*>& deviceExts);

    // Call once after VkDevice is up.
    bool init(VkInstance instance, VkPhysicalDevice physicalDevice, VkDevice device);

    // Returns the DLSS-suggested render resolution for the given output + quality.
    // On return `renderW`/`renderH` contain the internal resolution to use.
    bool getOptimalRenderResolution(
        uint32_t outputW, uint32_t outputH, QualityMode quality,
        uint32_t& renderW, uint32_t& renderH);

    // Create the DLSS feature handle. Must be called at least once per
    // (renderRes, outputRes, quality) tuple. `cmd` is used for internal
    // resource transitions; pass a command buffer in recording state.
    bool createFeature(
        VkCommandBuffer cmd,
        uint32_t renderW, uint32_t renderH,
        uint32_t outputW, uint32_t outputH,
        QualityMode quality,
        bool isHDR);

    // Per-frame upscale.
    //   inColor:   linear HDR RGBA16F @ render res
    //   outColor:  linear HDR RGBA16F @ output res (writeable)
    //   motion:    RG16F screen-space MVs @ render res (pixels; MV scale = 1,1 by default)
    //   depth:     linear or HW depth @ render res
    //   jitterPx:  Halton jitter applied to this frame's primary rays (pixel units)
    //   reset:     true on scene/camera teleport
    void evaluate(
        VkCommandBuffer cmd,
        VkImageView  inColor,  VkImage inColorImage,
        VkImageView  outColor, VkImage outColorImage,
        VkImageView  motion,   VkImage motionImage,
        VkImageView  depth,    VkImage depthImage,
        VkFormat     colorFormat, VkFormat motionFormat, VkFormat depthFormat,
        uint32_t     renderW, uint32_t renderH,
        uint32_t     outputW, uint32_t outputH,
        float        jitterX, float jitterY,
        bool         reset);

    void shutdown();
    bool isValid() const;

private:
    struct Impl;
    Impl* m_impl = nullptr;
};
