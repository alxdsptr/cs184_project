#pragma once
#include "render/AccumulationBuffer.h"
#include "render/AuxBuffers.h"
#include "render/Tonemapping.h"
#include "core/Camera.h"
#include "gpu/DeviceScene.h"
#include <cstdint>
#include <memory>
#ifdef PATHTRACER_NRD_DLSS_ENABLED
#  include <vulkan/vulkan.h>
#endif

class RayTracingBackend;
class VulkanDisplay;

#ifdef PATHTRACER_NRD_DLSS_ENABLED
class NRDContext;
class DLSSContext;
class CompositePass;
class VulkanSharedAuxBuffers;
#endif

class Renderer {
public:
    enum class Mode {
        Native,    // = today's behaviour (CUDA tonemap, direct blit)
        NRDOnly,   // denoise with NRD, composite+tonemap in Vulkan
        NRDDLSS,   // NRD → composite linear HDR → DLSS upscale → tonemap
    };

    Renderer();
    ~Renderer();

    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void resetAccumulation();

    // Per-frame entry.
    // `d_ldrOutput` is the CUDA interop buffer from VulkanDisplay::mapForCUDA;
    // used only in Native mode. `display` is only required for non-Native
    // modes (it provides Vulkan device / command buffer plumbing).
    void renderFrame(
        const CameraParams& camera,
        const DeviceSceneData& scene,
        RayTracingBackend* backend,
        uchar4* d_ldrOutput,
        bool enableEnvironment,
        uint32_t maxBounces,
        VulkanDisplay* display,
        uint32_t frameIndex
    );

    // Mode plumbing (safe to call after init()).
    Mode getMode() const { return m_mode; }
    bool setMode(Mode newMode, VulkanDisplay* display);   // returns the mode actually set (may demote)

    uint32_t getSampleCount() const { return m_accumBuffer.getSampleCount(); }
    float getExposure() const { return m_exposure; }
    void setExposure(float exposure) { m_exposure = exposure; }
    ToneMappingMode getToneMappingMode() const { return m_toneMappingMode; }
    void setToneMappingMode(ToneMappingMode mode) { m_toneMappingMode = mode; }

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    enum class DLSSQuality { Performance, Balanced, Quality, DLAA };
    void setDLSSQuality(DLSSQuality q);
    DLSSQuality getDLSSQuality() const { return m_dlssQuality; }
    uint32_t getRenderWidth()  const { return m_renderWidth; }
    uint32_t getRenderHeight() const { return m_renderHeight; }
#endif

    void shutdown();

private:
#ifdef PATHTRACER_NRD_DLSS_ENABLED
    // Mode init / teardown helpers.
    bool initNrdPath(VulkanDisplay* display, uint32_t renderW, uint32_t renderH);
    void shutdownNrdPath();
    bool initDlssPath(VulkanDisplay* display, uint32_t outputW, uint32_t outputH);
    void shutdownDlssPath();
    // Frame hook registered with VulkanDisplay — records NRD/composite/DLSS
    // into the active command buffer just before the swapchain blit.
    static void prePresentTrampoline(VkCommandBuffer cmd, void* user);
    void        recordPrePresent(VkCommandBuffer cmd);
#endif

    AccumulationBuffer m_accumBuffer;
    AuxBuffers         m_auxBuffers;
    uint32_t m_width = 0, m_height = 0;
    float    m_exposure = 1.0f;
    ToneMappingMode m_toneMappingMode = ToneMappingMode::ACES;

    Mode m_mode = Mode::Native;

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    // Display backend we were last bound to (needed for clean shutdown).
    VulkanDisplay* m_display = nullptr;
    // Render resolution (may be smaller than display size in NRDDLSS mode).
    uint32_t m_renderWidth  = 0;
    uint32_t m_renderHeight = 0;
    uint32_t m_frameIndex   = 0;
    DLSSQuality m_dlssQuality = DLSSQuality::Balanced;

    std::unique_ptr<VulkanSharedAuxBuffers> m_sharedAux;
    std::unique_ptr<NRDContext>     m_nrd;
    std::unique_ptr<DLSSContext>    m_dlss;
    std::unique_ptr<CompositePass>  m_compositeRender;  // writes to sampledImage (NRDOnly) or linear HDR (NRDDLSS)
    std::unique_ptr<CompositePass>  m_tonemap;          // NRDDLSS mode only (post-DLSS)

    // Render pass + framebuffer targeting VulkanDisplay::sampledImage for the
    // final LDR composite — shared by NRDOnly (direct composite) and NRDDLSS
    // (post-DLSS tonemap).
    VkRenderPass  m_ldrRenderPass = VK_NULL_HANDLE;
    VkFramebuffer m_ldrFramebuffer = VK_NULL_HANDLE;

    // Linear-HDR intermediate for NRDDLSS (render res) and upscaled output (output res).
    VkImage        m_hdrRenderImage = VK_NULL_HANDLE;
    VkDeviceMemory m_hdrRenderMem   = VK_NULL_HANDLE;
    VkImageView    m_hdrRenderView  = VK_NULL_HANDLE;
    VkFramebuffer  m_hdrRenderFb    = VK_NULL_HANDLE;
    VkRenderPass   m_hdrRenderPass  = VK_NULL_HANDLE;

    VkImage        m_hdrOutputImage = VK_NULL_HANDLE;
    VkDeviceMemory m_hdrOutputMem   = VK_NULL_HANDLE;
    VkImageView    m_hdrOutputView  = VK_NULL_HANDLE;

    // Cached for the pre-present recorder; set by renderFrame() each call.
    const CameraParams* m_lastCamera = nullptr;
    bool m_lastCameraMoved = false;
#endif
};
