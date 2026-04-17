#pragma once

// Wraps NRD (Real-Time Denoisers) + NRI (NVIDIA Render Interface) to denoise
// a split diffuse+specular path-traced frame into Vulkan images. Only used in
// RenderMode::NRDOnly / NRDDLSS; Native mode never constructs one.

#include <vulkan/vulkan.h>
#include <cstdint>

// NRD and NRI use PIMPL-style forward types; wrap them with opaque void*
// handles to keep SDK headers out of this header.
class VulkanSharedAuxBuffers;
class VulkanDisplay;

class NRDContext {
public:
    NRDContext();
    ~NRDContext();
    NRDContext(const NRDContext&) = delete;
    NRDContext& operator=(const NRDContext&) = delete;

    // Must be called after VulkanDisplay has finished its own init (device
    // + queues + enabled extensions all ready).
    bool init(const VulkanDisplay& display, uint32_t renderW, uint32_t renderH);

    // Re-create internal NRD resources at a new render resolution. Cheaper
    // than destroying/reinitializing because NRI device is preserved.
    bool resize(uint32_t renderW, uint32_t renderH);

    // One-shot per-frame: feed common settings (matrices, jitter, mvec scale).
    // `frameIndex` must increment monotonically across frames.
    void setCommonSettings(
        const float viewToClip[16],      // current, unjittered, col-major
        const float viewToClipPrev[16],  // previous, unjittered
        const float worldToView[16],     // current
        const float worldToViewPrev[16], // previous
        float cameraJitter[2],
        float cameraJitterPrev[2],
        float motionVectorScalePx[2],    // typically {1/renderW, 1/renderH} for pixel-space MVs
        uint32_t renderW, uint32_t renderH,
        uint32_t frameIndex,
        bool reset);

    // Dispatch the denoiser onto `cmd`. Inputs come from `aux` (diffuse/spec
    // radiance + hit-dist, normal+roughness, viewZ, MVs).
    // After return, the output images `outDiffuseView()` / `outSpecularView()`
    // contain the denoised demodulated radiance.
    // The app owns state transitions for the input images BEFORE this call:
    //   - they must be in VK_IMAGE_LAYOUT_GENERAL with STORAGE access.
    void denoise(VkCommandBuffer cmd, const VulkanSharedAuxBuffers& aux);

    // Read access to the denoised outputs (render-res VkImage / VkImageView).
    // Valid after a successful denoise() call within the same frame.
    VkImage     outDiffuseImage()  const;
    VkImageView outDiffuseView()   const;
    VkImage     outSpecularImage() const;
    VkImageView outSpecularView() const;

    void shutdown();
    bool isValid() const;

private:
    // Opaque impl to keep NRD / NRI headers out of this header.
    struct Impl;
    Impl* m_impl = nullptr;
};
