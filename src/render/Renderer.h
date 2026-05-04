#pragma once
#include "render/AccumulationBuffer.h"
#include "render/AuxBuffers.h"
#include "render/Tonemapping.h"
#include "render/ReSTIR.h"
#include "render/ReSTIRGI.h"
#include "render/ReSTIRPT.h"
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
class DLSSDContext;
class CompositePass;
class VulkanSharedAuxBuffers;
#endif

class Renderer {
public:
    enum class Mode {
        Native,     // = today's behaviour (CUDA tonemap, direct blit)
        NRDOnly,    // denoise with NRD, composite+tonemap in Vulkan
        NRDDLSS,    // NRD → composite linear HDR → DLSS upscale → tonemap
        DLSSOnly,   // path tracer → HDR interop → DLSS upscale → tonemap (no NRD)
        DLSSRR,     // DLSS Ray Reconstruction: noisy color + guides → DLSS-RR (denoise + upscale) → tonemap
    };

    Renderer();
    ~Renderer();

    void init(uint32_t width, uint32_t height);
    void resize(uint32_t width, uint32_t height);
    void resetAccumulation();
    // Forces ReSTIR DI + GI to drop their reservoir history. Call this for
    // events that actually invalidate the cached samples — scene reload,
    // resize, toggling a pass — NOT for plain camera motion (the temporal
    // pass re-projects + gates that itself).
    void invalidateReSTIRHistory();
    // Tell DLSS/NRD to drop their temporal history on the NEXT pre-present.
    // One-shot: cleared after the upscaler/denoiser consumes it. Call this
    // for genuine pipeline transitions (mode change, scene reload, resize,
    // camera teleport via file load) — NEVER on continuous camera motion,
    // which is what motion vectors are designed to handle.
    void markPipelineNeedsReset();

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
        uint32_t samplesPerFrame,
        VulkanDisplay* display,
        uint32_t frameIndex,
        // True when the camera moved this frame. Tells the ReSTIR passes to
        // clamp temporal M-cap aggressively so a 20-frame-old reservoir from
        // before the motion doesn't outweigh the fresh canonical sample.
        bool cameraMoved = false
    );

    // Mode plumbing (safe to call after init()).
    Mode getMode() const { return m_mode; }
    bool setMode(Mode newMode, VulkanDisplay* display);   // returns the mode actually set (may demote)

    uint32_t getSampleCount() const { return m_accumBuffer.getSampleCount(); }
    float getExposure() const { return m_exposure; }
    void setExposure(float exposure) { m_exposure = exposure; }
    ToneMappingMode getToneMappingMode() const { return m_toneMappingMode; }
    void setToneMappingMode(ToneMappingMode mode) { m_toneMappingMode = mode; }

    // ReSTIR DI controls (applied to primary-hit direct lighting in Native mode).
    bool  isReSTIREnabled() const { return m_restir.enabled(); }
    void  setReSTIREnabled(bool on) { m_restir.setEnabled(on); m_restir.invalidateHistory(); }
    ReSTIRContext&       restir()       { return m_restir; }
    const ReSTIRContext& restir() const { return m_restir; }

    // ReSTIR GI controls (replaces continuation bounces at the primary hit
    // with a resampled 1-bounce indirect estimate). Native mode only.
    bool  isReSTIRGIEnabled() const { return m_restirGI.enabled(); }
    void  setReSTIRGIEnabled(bool on) { m_restirGI.setEnabled(on); m_restirGI.invalidateHistory(); }
    ReSTIRGIContext&       restirGI()       { return m_restirGI; }
    const ReSTIRGIContext& restirGI() const { return m_restirGI; }

    // ReSTIR PT controls (Lin et al. 2022). Generalises GI to arbitrary path
    // length: stored radiance is the result of a multi-bounce random walk
    // past the reconnection vertex. Native mode only. Mutually exclusive
    // with GI at the consumption site (PT wins when both are on).
    bool  isReSTIRPTEnabled() const { return m_restirPT.enabled(); }
    void  setReSTIRPTEnabled(bool on) { m_restirPT.setEnabled(on); m_restirPT.invalidateHistory(); }
    ReSTIRPTContext&       restirPT()       { return m_restirPT; }
    const ReSTIRPTContext& restirPT() const { return m_restirPT; }

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
    // `withNrd=false` skips NRD itself but still allocates the shared aux
    // images + LDR framebuffer + composite pass — used by DLSSOnly mode where
    // the path tracer writes HDR directly into the shared interop image and
    // DLSS upscales it without going through any denoiser.
    bool initNrdPath(VulkanDisplay* display, uint32_t renderW, uint32_t renderH,
                     bool withNrd = true);
    void shutdownNrdPath();
    bool initDlssPath(VulkanDisplay* display, uint32_t outputW, uint32_t outputH);
    void shutdownDlssPath();
    // Frame hook registered with VulkanDisplay — records NRD/composite/DLSS
    // into the active command buffer just before the swapchain blit.
    static void prePresentTrampoline(VkCommandBuffer cmd, void* user);
    void        recordPrePresent(VkCommandBuffer cmd);
    // DLSSOnly fast path: HDR interop image → DLSS upscale → tonemap.
    void        recordDlssOnlyPrePresent(VkCommandBuffer cmd);
    // DLSSRR fast path: noisy color + RR guides → DLSS-RR upscale+denoise → tonemap.
    void        recordDlssRRPrePresent(VkCommandBuffer cmd);
#endif

    AccumulationBuffer m_accumBuffer;
    AuxBuffers         m_auxBuffers;
    ReSTIRContext      m_restir;
    ReSTIRGIContext    m_restirGI;
    ReSTIRPTContext    m_restirPT;
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
    std::unique_ptr<DLSSDContext>   m_dlssd;   // DLSS-RR (Mode::DLSSRR)
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
    // MUST be a full copy, not a pointer: the CameraParams fed into
    // renderFrame() is a stack local in Application::renderSceneSample(),
    // which goes out of scope before present() runs the pre-present hook.
    // Holding a pointer was dangling-read territory — NRD then saw garbage
    // matrices/jitter and produced pure noise.
    CameraParams m_lastCamera{};
    bool m_lastCameraValid = false;
    bool m_lastCameraMoved = false;
    // Previous-frame jitter, in pixel units. NRD needs `cameraJitterPrev` to
    // correctly align sub-pixel positions when reprojecting history.
    float m_prevJitter[2] = {0.0f, 0.0f};

    // True if NRD/DLSS/DLSS-RR should treat the next pre-present as a fresh
    // pipeline start (history-reset). Set by mode change, scene reload,
    // resize and explicit camera teleport (loadFromFile). NEVER set by
    // continuous WASD/dolly/orbit motion — these are the case temporal
    // accumulation is *designed* to handle via motion vectors, and resetting
    // every frame causes severe shimmer because the upscaler/denoiser must
    // rebuild history from zero each frame.
    //
    // Decoupled from `AccumulationBuffer::sampleCount`: that buffer averages
    // path-traced radiance across frames and *must* reset on any camera
    // motion (different views can't be averaged without ghosting). DLSS/NRD
    // history is a different beast that survives motion via reprojection.
    bool m_pipelineNeedsReset = true;
#endif
};
