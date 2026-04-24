#include "render/Renderer.h"
#include "render/Tonemapping.h"
#include "backend/RayTracingBackend.h"
#include "display/VulkanDisplay.h"
#include "util/Log.h"

#ifdef PATHTRACER_NRD_DLSS_ENABLED
#include "render/PathTraceKernel.h"
#include "render/VulkanSharedAuxBuffers.h"
#include "postfx/NRDContext.h"
#include "postfx/DLSSContext.h"
#include "postfx/CompositePass.h"
#include "interop/VulkanImageInterop.h"
#include "core/Math.h"   // mat4_inverse
#include <cstring>
#include <cmath>
#endif

Renderer::Renderer()  = default;
Renderer::~Renderer() = default;

void Renderer::init(uint32_t width, uint32_t height) {
    m_width  = width;
    m_height = height;
    m_accumBuffer.init(width, height);
    m_auxBuffers.init(width, height);
    m_restir.init(width, height);
#ifdef PATHTRACER_NRD_DLSS_ENABLED
    m_renderWidth  = width;
    m_renderHeight = height;
#endif
}

void Renderer::resize(uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height) return;
    m_width  = width;
    m_height = height;
    m_accumBuffer.resize(width, height);
    m_auxBuffers.resize(width, height);
    m_restir.resize(width, height);
    m_restir.invalidateHistory();

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    // Non-Native modes rely on shared VkImages sized to the render resolution;
    // Native mode uses the CUDA-only aux buffers above. Re-init the non-Native
    // pipeline if it was active.
    if (m_mode != Mode::Native && m_display) {
        const uint32_t rw = (m_mode == Mode::NRDDLSS) ? m_renderWidth : width;
        const uint32_t rh = (m_mode == Mode::NRDDLSS) ? m_renderHeight : height;
        (void)rw; (void)rh;
        // Keep the current render resolution policy on display-size change:
        //   - NRDOnly: render res == display res, follows `width/height`.
        //   - NRDDLSS: ask DLSS for optimal render res at the new output res.
        Mode m = m_mode;
        setMode(Mode::Native, m_display);  // tears down the pipeline safely
        setMode(m,              m_display);
    }
#endif
}

void Renderer::resetAccumulation() {
    m_accumBuffer.reset();
    // The reservoir history samples light transport that is now stale —
    // keeping it would smear the previous camera's shading into the new one.
    m_restir.invalidateHistory();
}

void Renderer::renderFrame(
    const CameraParams& camera,
    const DeviceSceneData& scene,
    RayTracingBackend* backend,
    uchar4* d_ldrOutput,
    bool enableEnvironment,
    uint32_t maxBounces,
    uint32_t samplesPerFrame,
    VulkanDisplay* display,
    uint32_t frameIndex)
{
    uint32_t sampleIndex = m_accumBuffer.getSampleCount();
    if (samplesPerFrame < 1) samplesPerFrame = 1;

    if (m_mode == Mode::Native) {
        // ── ReSTIR DI prepass (Bitterli et al. 2020) ─────────────
        // Runs before the main path tracer when enabled and the scene has
        // an area-light BVH; writes a per-pixel reservoir that the main
        // kernel consumes at bounce-0 NEE.
        DeviceSceneData sceneWithBVH = scene;
        backend->patchScene(sceneWithBVH);
        const bool restirRan = m_restir.enabled() &&
            sceneWithBVH.d_lightBVHNodes &&
            sceneWithBVH.d_areaLights &&
            sceneWithBVH.areaLightCount > 0 &&
            sceneWithBVH.d_bvhNodes;
        if (restirRan) {
            m_restir.runFrame(sceneWithBVH, camera,
                              m_width, m_height, sampleIndex);
        }

        DeviceSceneData scenePatched = scene;
        if (restirRan) {
            scenePatched.d_restirReservoirs = m_restir.getBuffers().d_reservoirsCurr;
            scenePatched.restirEnabled      = 1;
        }

        backend->launchPathTrace(
            scenePatched, camera,
            m_accumBuffer.getAccumBuffer(),
            m_accumBuffer.getOutputBuffer(),
            m_auxBuffers.getPtrs(),
            m_width, m_height, sampleIndex,
            enableEnvironment,
            maxBounces,
            samplesPerFrame
        );
        launchTonemapKernel(
            m_accumBuffer.getOutputBuffer(),
            d_ldrOutput,
            m_width, m_height,
            m_exposure,
            m_toneMappingMode
        );
        m_accumBuffer.addSamples(samplesPerFrame);
        if (restirRan) m_restir.swapHistory();
        return;
    }

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    // ── DLSSOnly: backend writes HDR + motion + viewZ directly into the
    // shared interop image. No diff/spec split, no NRD. The same backend
    // (CUDA or OptiX) used for Native is used here.
    if (m_mode == Mode::DLSSOnly) {
        PrimaryHitSurfaces gb{};
        if (m_sharedAux) {
            SharedAuxSurfaces s = m_sharedAux->surfaces();
            gb.motionVectors = s.motionVectors;
            gb.viewZ         = s.viewZ;
            gb.hdrColor      = s.hdrColor;
            gb.ndcDepth      = s.ndcDepth;
        }
        backend->launchPathTrace(
            scene, camera,
            m_accumBuffer.getAccumBuffer(),
            // d_outputBuffer not needed (gb.hdrColor publishes the result),
            // but kernels still use it as a scratch / debug write. Pass the
            // accum-buffer's CUDA-side output to keep the existing path alive.
            m_accumBuffer.getOutputBuffer(),
            m_auxBuffers.getPtrs(),
            m_renderWidth, m_renderHeight, sampleIndex,
            enableEnvironment, maxBounces, samplesPerFrame,
            gb);
        m_accumBuffer.addSamples(samplesPerFrame);

        m_lastCamera = camera;
        m_lastCameraValid = true;
        m_frameIndex = frameIndex;
        if (display && display != m_display) {
            display->setPrePresentRecorder(&Renderer::prePresentTrampoline, this);
            m_display = display;
        }
        return;
    }

    // ── NRDOnly / NRDDLSS: split-output path-trace into Vulkan-shared aux images.
    // Goes through the backend (CUDA SAH-BVH / OptiX GAS) so OptiX-only scenes
    // (no CUDA BVH built) still work.
    SplitSurfaceOutputs surf{};
    if (m_sharedAux) {
        SharedAuxSurfaces s = m_sharedAux->surfaces();
        surf.diffuseRadianceHitDist  = s.diffuseRadianceHitDist;
        surf.specularRadianceHitDist = s.specularRadianceHitDist;
        surf.normalRoughness         = s.normalRoughness;
        surf.viewZ                   = s.viewZ;
        surf.motionVectors           = s.motionVectors;
        surf.albedo                  = s.albedo;
        surf.emissive                = s.emissive;
        surf.ndcDepth                = s.ndcDepth;
    }
    // NRDOnly has no AA resolver behind it (no DLSS, no TAA) — see NRD README:
    // "NRD tries to preserve jittering at least on geometrical edges ... moves
    // the problem of anti-aliasing to the application side." Feeding Halton
    // sub-pixel jitter with no final resolver produces a persistent shimmer on
    // edges even when the camera is static. Zero the jitter in this mode.
    // NRDDLSS keeps Halton — DLSS consumes it for super-sampling AA.
    CameraParams cameraForSplit = camera;
    if (m_mode == Mode::NRDOnly) {
        cameraForSplit.jitterOffset = make_float2(0.0f, 0.0f);
    }
    backend->launchPathTraceSplit(
        scene, cameraForSplit, surf,
        m_renderWidth, m_renderHeight, sampleIndex,
        enableEnvironment, maxBounces, samplesPerFrame);
    m_accumBuffer.incrementSamples();

    // Cache what the pre-present recorder needs; it runs inside present(),
    // long after the `camera` argument (a stack local in the caller) has
    // gone out of scope — so take a deep copy, don't stash a pointer.
    // IMPORTANT: cache the *kernel-visible* camera (including the zeroed
    // jitter for NRDOnly) so NRD sees the same jitter the ray-gen used.
    m_lastCamera = cameraForSplit;
    m_lastCameraValid = true;
    m_frameIndex = frameIndex;
    // Register the pre-present hook idempotently.
    if (display && display != m_display) {
        // Defensive: this shouldn't normally happen — setMode should have
        // wired the hook already.
        display->setPrePresentRecorder(&Renderer::prePresentTrampoline, this);
        m_display = display;
    }
#else
    (void)display; (void)frameIndex;
#endif
}

#ifdef PATHTRACER_NRD_DLSS_ENABLED
// Forward decls — defined below in the NRD+DLSS impl section.
static bool initHdrIntermediates(
    VkDevice dev, VkPhysicalDevice phys,
    uint32_t renderW, uint32_t renderH,
    uint32_t outputW, uint32_t outputH,
    VkRenderPass& hdrRenderPass,
    VkImage& renderImg, VkDeviceMemory& renderMem, VkImageView& renderView, VkFramebuffer& renderFb,
    VkImage& outputImg, VkDeviceMemory& outputMem, VkImageView& outputView);
static bool createHdrTarget(VkDevice dev, VkPhysicalDevice phys,
    uint32_t w, uint32_t h, VkImageUsageFlags extraUsage,
    VkImage& outImg, VkDeviceMemory& outMem, VkImageView& outView);
#endif

bool Renderer::setMode(Mode newMode, VulkanDisplay* display) {
#ifndef PATHTRACER_NRD_DLSS_ENABLED
    (void)display;
    if (newMode != Mode::Native) {
        LOG_WARN("NRD+DLSS was not compiled in; staying in Native mode");
        m_mode = Mode::Native;
        return false;
    }
    m_mode = Mode::Native;
    return true;
#else
    if (newMode == m_mode && display == m_display) return true;

    // Always tear down non-Native resources first to reach a clean state.
    shutdownDlssPath();
    shutdownNrdPath();
    if (m_display) {
        m_display->setPrePresentRecorder(nullptr, nullptr);
    }

    if (newMode == Mode::Native) {
        // Restore accum/aux buffers to output resolution if the previous mode
        // had shrunk them (DLSSOnly resizes them to renderW × renderH).
        if (m_renderWidth != m_width || m_renderHeight != m_height) {
            m_accumBuffer.resize(m_width, m_height);
            m_auxBuffers.resize(m_width, m_height);
        }
        m_mode = Mode::Native;
        m_display = display;
        m_renderWidth = m_width;
        m_renderHeight = m_height;
        return true;
    }

    if (!display) {
        LOG_WARN("Renderer::setMode requires a VulkanDisplay; staying Native");
        m_mode = Mode::Native;
        return false;
    }

    // For NRDDLSS / DLSSOnly, ask DLSS for optimal render resolution first.
    uint32_t renderW = m_width, renderH = m_height;
    const bool wantsDlss = (newMode == Mode::NRDDLSS || newMode == Mode::DLSSOnly);
    if (wantsDlss) {
        if (!initDlssPath(display, m_width, m_height)) {
            LOG_WARN("DLSS init failed — demoting %s → %s",
                     newMode == Mode::NRDDLSS ? "NRDDLSS" : "DLSSOnly",
                     newMode == Mode::NRDDLSS ? "NRDOnly" : "Native");
            if (newMode == Mode::NRDDLSS) {
                newMode = Mode::NRDOnly;
                renderW = m_width;
                renderH = m_height;
            } else {
                m_mode = Mode::Native;
                m_display = display;
                m_renderWidth = m_width;
                m_renderHeight = m_height;
                return false;
            }
        } else if (m_dlss) {
            uint32_t rw = 0, rh = 0;
            DLSSContext::QualityMode dq = DLSSContext::BALANCED;
            switch (m_dlssQuality) {
                case DLSSQuality::Performance: dq = DLSSContext::PERFORMANCE; break;
                case DLSSQuality::Balanced:    dq = DLSSContext::BALANCED; break;
                case DLSSQuality::Quality:     dq = DLSSContext::QUALITY; break;
                case DLSSQuality::DLAA:        dq = DLSSContext::DLAA; break;
            }
            if (m_dlss->getOptimalRenderResolution(m_width, m_height, dq, rw, rh)) {
                renderW = rw; renderH = rh;
            }
        }
    }

    const bool needsNrd = (newMode == Mode::NRDOnly || newMode == Mode::NRDDLSS);
    if (!initNrdPath(display, renderW, renderH, /*withNrd=*/needsNrd)) {
        LOG_WARN("NRD init failed — demoting to Native");
        shutdownDlssPath();
        shutdownNrdPath();
        m_mode = Mode::Native;
        m_display = display;
        return false;
    }

    // NRDDLSS mode: replace the LDR composite with a linear-HDR one (written
    // into the render-res intermediate), then create the tonemap pass which
    // samples the DLSS upscaled output and writes to the swapchain-sized
    // sampled image.
    if (newMode == Mode::NRDDLSS) {
        VkDevice dev = display->device();
        VkPhysicalDevice phys = display->physicalDevice();
        if (!initHdrIntermediates(dev, phys,
                renderW, renderH, m_width, m_height,
                m_hdrRenderPass,
                m_hdrRenderImage, m_hdrRenderMem, m_hdrRenderView, m_hdrRenderFb,
                m_hdrOutputImage, m_hdrOutputMem, m_hdrOutputView)) {
            LOG_WARN("HDR intermediate alloc failed — demoting NRDDLSS → NRDOnly");
            newMode = Mode::NRDOnly;
        } else {
            // Rebuild compositeRender against the HDR render pass (linear output).
            m_compositeRender.reset();
            m_compositeRender = std::make_unique<CompositePass>();
            if (!m_compositeRender->init(dev, m_hdrRenderPass,
                                         CompositePass::COMPOSITE_LINEAR_HDR,
                                         VK_FORMAT_R16G16B16A16_SFLOAT, "shaders")) {
                LOG_WARN("Composite (linear HDR) init failed — demoting to NRDOnly");
                newMode = Mode::NRDOnly;
            }
            // Tonemap pass renders into the swapchain-sized sampled image.
            if (newMode == Mode::NRDDLSS) {
                m_tonemap = std::make_unique<CompositePass>();
                if (!m_tonemap->init(dev, m_ldrRenderPass,
                                     CompositePass::TONEMAP_ONLY,
                                     display->sampledImageFormat(), "shaders")) {
                    LOG_WARN("Tonemap pass init failed — demoting to NRDOnly");
                    newMode = Mode::NRDOnly;
                    m_tonemap.reset();
                }
            }
            // If we demoted, restore COMPOSITE_TONEMAP into compositeRender.
            if (newMode == Mode::NRDOnly) {
                m_compositeRender.reset();
                m_compositeRender = std::make_unique<CompositePass>();
                if (!m_compositeRender->init(dev, m_ldrRenderPass,
                        CompositePass::COMPOSITE_TONEMAP,
                        display->sampledImageFormat(), "shaders")) {
                    LOG_ERROR("Unable to rebuild LDR composite after demotion");
                    shutdownNrdPath();
                    m_mode = Mode::Native;
                    m_display = display;
                    return false;
                }
            }
        }
    }

    // DLSSOnly: the path-trace kernel accumulates into m_accumBuffer at the
    // (sub-output) render resolution; we must resize it before the first
    // frame, otherwise the kernel walks past the buffer end.
    if (newMode == Mode::DLSSOnly && (renderW != m_width || renderH != m_height)) {
        m_accumBuffer.resize(renderW, renderH);
        m_auxBuffers.resize(renderW, renderH);
    }

    // DLSSOnly mode: path tracer writes HDR directly into m_sharedAux->hdrColor()
    // (a render-res shared image), DLSS upscales it into m_hdrOutputImage, then
    // m_tonemap maps to LDR sRGB on the swapchain-sized sampled image. No
    // composite-render pass (no diff/spec to combine), no NRD render-res HDR
    // intermediate (we use the shared interop image as DLSS input).
    if (newMode == Mode::DLSSOnly) {
        VkDevice dev = display->device();
        VkPhysicalDevice phys = display->physicalDevice();
        // Output-res HDR image only (DLSS write target). The shared
        // m_sharedAux->hdrColor() serves as DLSS input.
        if (!createHdrTarget(dev, phys, m_width, m_height,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                m_hdrOutputImage, m_hdrOutputMem, m_hdrOutputView)) {
            LOG_ERROR("DLSSOnly: HDR output image alloc failed — demoting to Native");
            shutdownDlssPath();
            shutdownNrdPath();
            m_mode = Mode::Native;
            m_display = display;
            return false;
        }
        m_tonemap = std::make_unique<CompositePass>();
        if (!m_tonemap->init(dev, m_ldrRenderPass,
                             CompositePass::TONEMAP_ONLY,
                             display->sampledImageFormat(), "shaders")) {
            LOG_ERROR("DLSSOnly: Tonemap pass init failed — demoting to Native");
            shutdownDlssPath();
            shutdownNrdPath();
            m_mode = Mode::Native;
            m_display = display;
            return false;
        }
    }

    m_renderWidth = renderW;
    m_renderHeight = renderH;
    m_mode = newMode;
    m_display = display;
    display->setPrePresentRecorder(&Renderer::prePresentTrampoline, this);
    resetAccumulation();
    return true;
#endif
}

void Renderer::shutdown() {
#ifdef PATHTRACER_NRD_DLSS_ENABLED
    if (m_display) m_display->setPrePresentRecorder(nullptr, nullptr);
    shutdownDlssPath();
    shutdownNrdPath();
#endif
    m_accumBuffer.free();
    m_auxBuffers.free();
    m_restir.free();
}

// ────────────────────────────────────────────────────────────────
// NRD + DLSS implementation (only compiled when the option is on)
// ────────────────────────────────────────────────────────────────
#ifdef PATHTRACER_NRD_DLSS_ENABLED

void Renderer::setDLSSQuality(DLSSQuality q) {
    if (q == m_dlssQuality) return;
    m_dlssQuality = q;
    // Re-init the active DLSS-using mode so DLSS picks the new quality's
    // render resolution. setMode() short-circuits on (mode==current); bounce
    // through Native to force a full teardown + rebuild. Cheap (few ms).
    if ((m_mode == Mode::NRDDLSS || m_mode == Mode::DLSSOnly) && m_display) {
        Mode prev = m_mode;
        setMode(Mode::Native, m_display);
        setMode(prev,         m_display);
    }
}

// Build a render pass whose single color attachment is the swapchain-sized
// LDR image (VulkanDisplay::sampledImage). Final layout is SHADER_READ_ONLY
// so the display's blit pipeline can sample it without extra transitions.
static VkRenderPass createLdrRenderPass(VkDevice device, VkFormat format) {
    VkAttachmentDescription att{};
    att.format         = format;
    att.samples        = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments = &ref;
    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1; rpci.pAttachments = &att;
    rpci.subpassCount = 1;    rpci.pSubpasses = &sp;
    rpci.dependencyCount = 2; rpci.pDependencies = deps;
    VkRenderPass rp = VK_NULL_HANDLE;
    vkCreateRenderPass(device, &rpci, nullptr, &rp);
    return rp;
}

bool Renderer::initNrdPath(VulkanDisplay* display, uint32_t renderW, uint32_t renderH,
                           bool withNrd)
{
    m_sharedAux = std::make_unique<VulkanSharedAuxBuffers>();
    if (!m_sharedAux->create(display->device(), display->physicalDevice(), renderW, renderH)) {
        LOG_ERROR("VulkanSharedAuxBuffers::create failed");
        m_sharedAux.reset();
        return false;
    }

    if (withNrd) {
        m_nrd = std::make_unique<NRDContext>();
        if (!m_nrd->init(*display, renderW, renderH)) {
            m_nrd.reset();
            m_sharedAux.reset();
            return false;
        }
    }

    // Composite render pass + framebuffer onto VulkanDisplay's sampled image.
    VkDevice dev = display->device();
    const VkFormat fmt = display->sampledImageFormat();
    m_ldrRenderPass = createLdrRenderPass(dev, fmt);
    if (!m_ldrRenderPass) {
        LOG_ERROR("Renderer: failed to create LDR render pass");
        m_nrd.reset(); m_sharedAux.reset();
        return false;
    }
    VkImageView viewAtt = display->sampledImageView();
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass      = m_ldrRenderPass;
    fbci.attachmentCount = 1;
    fbci.pAttachments    = &viewAtt;
    fbci.width           = display->width();
    fbci.height          = display->height();
    fbci.layers          = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &m_ldrFramebuffer) != VK_SUCCESS) {
        vkDestroyRenderPass(dev, m_ldrRenderPass, nullptr); m_ldrRenderPass = VK_NULL_HANDLE;
        m_nrd.reset(); m_sharedAux.reset();
        return false;
    }

    // For NRDOnly the composite combines (NRD diff + spec + alb + emis) and
    // tonemaps in one pass. For DLSSOnly that pass would be redundant — DLSS
    // already produced an HDR image we just need to tonemap. Skip it; the
    // DLSSOnly recordPrePresent() path uses `m_tonemap` (built later in
    // setMode()) instead.
    if (withNrd) {
        m_compositeRender = std::make_unique<CompositePass>();
        if (!m_compositeRender->init(dev, m_ldrRenderPass,
                                     CompositePass::COMPOSITE_TONEMAP, fmt, "shaders")) {
            LOG_ERROR("Renderer: CompositePass::init failed");
            vkDestroyFramebuffer(dev, m_ldrFramebuffer, nullptr); m_ldrFramebuffer = VK_NULL_HANDLE;
            vkDestroyRenderPass(dev, m_ldrRenderPass, nullptr); m_ldrRenderPass = VK_NULL_HANDLE;
            m_nrd.reset(); m_sharedAux.reset();
            return false;
        }
    }
    return true;
}

void Renderer::shutdownNrdPath() {
    // Wait for pending GPU work before touching anything NRD touched.
    if (m_display) vkDeviceWaitIdle(m_display->device());

    m_nrd.reset();
    m_compositeRender.reset();
    m_tonemap.reset();

    if (m_display) {
        VkDevice dev = m_display->device();
        if (m_ldrFramebuffer) { vkDestroyFramebuffer(dev, m_ldrFramebuffer, nullptr); m_ldrFramebuffer = VK_NULL_HANDLE; }
        if (m_ldrRenderPass)  { vkDestroyRenderPass(dev, m_ldrRenderPass, nullptr);  m_ldrRenderPass = VK_NULL_HANDLE; }
        if (m_hdrRenderFb)    vkDestroyFramebuffer(dev, m_hdrRenderFb, nullptr);
        if (m_hdrRenderView)  vkDestroyImageView(dev, m_hdrRenderView, nullptr);
        if (m_hdrRenderImage) vkDestroyImage(dev, m_hdrRenderImage, nullptr);
        if (m_hdrRenderMem)   vkFreeMemory(dev, m_hdrRenderMem, nullptr);
        if (m_hdrRenderPass)  vkDestroyRenderPass(dev, m_hdrRenderPass, nullptr);
        if (m_hdrOutputView)  vkDestroyImageView(dev, m_hdrOutputView, nullptr);
        if (m_hdrOutputImage) vkDestroyImage(dev, m_hdrOutputImage, nullptr);
        if (m_hdrOutputMem)   vkFreeMemory(dev, m_hdrOutputMem, nullptr);
    }
    m_hdrRenderFb = VK_NULL_HANDLE;
    m_hdrRenderView = VK_NULL_HANDLE;
    m_hdrRenderImage = VK_NULL_HANDLE;
    m_hdrRenderMem = VK_NULL_HANDLE;
    m_hdrRenderPass = VK_NULL_HANDLE;
    m_hdrOutputView = VK_NULL_HANDLE;
    m_hdrOutputImage = VK_NULL_HANDLE;
    m_hdrOutputMem = VK_NULL_HANDLE;
    m_sharedAux.reset();
}

// Allocate a plain Vulkan image + view + (optional) framebuffer for the
// NRDDLSS intermediate targets.
static bool createHdrTarget(VkDevice dev, VkPhysicalDevice phys,
                            uint32_t w, uint32_t h, VkImageUsageFlags usage,
                            VkImage& image, VkDeviceMemory& mem, VkImageView& view)
{
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format    = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent    = { w, h, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(dev, &ici, nullptr, &image) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(dev, image, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t idx = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            idx = i; break;
        }
    }
    if (idx == UINT32_MAX) return false;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = idx;
    if (vkAllocateMemory(dev, &mai, nullptr, &mem) != VK_SUCCESS) return false;
    if (vkBindImageMemory(dev, image, mem, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return vkCreateImageView(dev, &vci, nullptr, &view) == VK_SUCCESS;
}

static VkRenderPass createHdrRenderPass(VkDevice device) {
    VkAttachmentDescription att{};
    att.format         = VK_FORMAT_R16G16B16A16_SFLOAT;
    att.samples        = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments = &ref;
    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1; rpci.pAttachments = &att;
    rpci.subpassCount = 1;    rpci.pSubpasses = &sp;
    rpci.dependencyCount = 2; rpci.pDependencies = deps;
    VkRenderPass rp = VK_NULL_HANDLE;
    vkCreateRenderPass(device, &rpci, nullptr, &rp);
    return rp;
}

bool Renderer::initDlssPath(VulkanDisplay* display, uint32_t outputW, uint32_t outputH) {
    m_dlss = std::make_unique<DLSSContext>();
    if (!m_dlss->init(display->instance(), display->physicalDevice(), display->device())) {
        m_dlss.reset();
        return false;
    }
    (void)outputW; (void)outputH;
    // Feature creation happens inside the pre-present recorder, once we have
    // a recording command buffer.
    return true;
}

// Allocate all NRDDLSS-mode intermediate resources. Called from setMode() once
// the render resolution is known. Returns false on allocation failure; caller
// should then demote to NRDOnly / Native.
static bool initHdrIntermediates(
    VkDevice dev, VkPhysicalDevice phys,
    uint32_t renderW, uint32_t renderH,
    uint32_t outputW, uint32_t outputH,
    VkRenderPass& hdrRenderPass,
    VkImage& renderImg, VkDeviceMemory& renderMem, VkImageView& renderView, VkFramebuffer& renderFb,
    VkImage& outputImg, VkDeviceMemory& outputMem, VkImageView& outputView)
{
    hdrRenderPass = createHdrRenderPass(dev);
    if (!hdrRenderPass) return false;

    // Render-res intermediate: color attachment + sampled (consumed by DLSS).
    if (!createHdrTarget(dev, phys, renderW, renderH,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            renderImg, renderMem, renderView))
        return false;

    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass      = hdrRenderPass;
    fbci.attachmentCount = 1;
    fbci.pAttachments    = &renderView;
    fbci.width           = renderW;
    fbci.height          = renderH;
    fbci.layers          = 1;
    if (vkCreateFramebuffer(dev, &fbci, nullptr, &renderFb) != VK_SUCCESS) return false;

    // Output-res HDR target: DLSS storage write + sampled for tonemap pass.
    if (!createHdrTarget(dev, phys, outputW, outputH,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            outputImg, outputMem, outputView))
        return false;
    return true;
}

void Renderer::shutdownDlssPath() {
    m_dlss.reset();
}

void Renderer::prePresentTrampoline(VkCommandBuffer cmd, void* user) {
    static_cast<Renderer*>(user)->recordPrePresent(cmd);
}

void Renderer::recordPrePresent(VkCommandBuffer cmd) {
    if (!m_display || !m_sharedAux || !m_lastCameraValid) return;
    // NRDOnly / NRDDLSS need the NRD denoiser; DLSSOnly does not.
    const bool needsNrd = (m_mode == Mode::NRDOnly || m_mode == Mode::NRDDLSS);
    if (needsNrd && !m_nrd) return;

    const uint32_t rw = m_renderWidth;
    const uint32_t rh = m_renderHeight;

    // ── DLSSOnly fast path: skip NRD + composite. The path tracer wrote HDR
    // straight into m_sharedAux->hdrColor(); we just need DLSS upscale + tonemap.
    if (m_mode == Mode::DLSSOnly) {
        recordDlssOnlyPrePresent(cmd);
        return;
    }

    // Temporary crash-triage checkpoints. Remove once NRD is stable.
    static uint32_t s_ppFrame = 0;
    if (s_ppFrame < 3) {
        LOG_INFO("recordPrePresent[%u]: enter rw=%u rh=%u mode=%d",
                 s_ppFrame, rw, rh, (int)m_mode);
    }

    // Transition every shared aux image: UNDEFINED → GENERAL so NRD can
    // access them as STORAGE. They were written by CUDA surface writes, and
    // on first use Vulkan doesn't yet track their layout.
    auto toGeneral = [&](VkImage img) {
        SharedVulkanImage::transition(
            cmd, img,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    };
    toGeneral(m_sharedAux->diffuseRadianceHitDist().image());
    toGeneral(m_sharedAux->specularRadianceHitDist().image());
    toGeneral(m_sharedAux->normalRoughness().image());
    toGeneral(m_sharedAux->viewZ().image());
    toGeneral(m_sharedAux->motionVectors().image());
    toGeneral(m_sharedAux->albedo().image());
    toGeneral(m_sharedAux->emissive().image());
    toGeneral(m_sharedAux->ndcDepth().image());
    // NRD outputs: previous frame's toRead() left these in SHADER_READ_ONLY,
    // but we declared their initial state to NRI as GENERAL. NRI's generated
    // barrier at the top of Denoise() therefore uses an oldLayout that
    // doesn't match reality → validation warns and the driver may skip the
    // real layout transition, so NRD writes land in an image that samplers
    // downstream still see as SHADER_READ_ONLY content → output looks like
    // pure noise. Push them back to GENERAL ourselves (contents are
    // immediately overwritten by NRD anyway, so UNDEFINED→GENERAL is fine).
    toGeneral(m_nrd->outDiffuseImage());
    toGeneral(m_nrd->outSpecularImage());
    if (s_ppFrame < 3) LOG_INFO("recordPrePresent[%u]: aux→GENERAL done", s_ppFrame);

    // Build NRD common settings from the camera matrices.
    //
    // NRD requires COLUMN-MAJOR matrices (NRDSettings.h: "layout - column-major").
    // Our float4x4 stores rows contiguously (m[row][col]), so a raw memcpy
    // would hand NRD the transpose. Transpose here so world→clip reprojection
    // inside NRD matches the path-traced motion vectors, otherwise history
    // reprojection fails and temporal accumulation silently degrades to
    // "pass-through" — which looks exactly like "the denoiser is off".
    auto toColumnMajor = [](const float4x4& src, float dst[16]) {
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                dst[c * 4 + r] = src.m[r][c];
    };
    float viewToClip[16], viewToClipPrev[16], worldToView[16], worldToViewPrev[16];
    toColumnMajor(m_lastCamera.projMatrix,     viewToClip);
    toColumnMajor(m_lastCamera.prevProjMatrix, viewToClipPrev);
    toColumnMajor(m_lastCamera.viewMatrix,     worldToView);
    toColumnMajor(m_lastCamera.prevViewMatrix, worldToViewPrev);

    float jitter[2]     = { m_lastCamera.jitterOffset.x, m_lastCamera.jitterOffset.y };
    float jitterPrev[2] = { m_prevJitter[0], m_prevJitter[1] };
    // MVs are stored in pixel space; NRD wants UV-space post-scale → {1/w, 1/h}.
    float mvScale[2] = { 1.0f / (float)rw, 1.0f / (float)rh };

    // Temporal water-wave artifact hunt: the first ~6 frames should show
    // prev jitter equal to the previous frame's value (NOT zero, NOT the
    // current frame's value, NOT negated). If prev is 0 the denoiser thinks
    // the camera jittered by a full cameraJitter between frames, which causes
    // a sub-pixel reprojection bias that manifests as ripples.
    if (s_ppFrame < 6) {
        LOG_INFO("recordPrePresent[%u]: jitter curr=(%.4f,%.4f) prev=(%.4f,%.4f)",
                 s_ppFrame, jitter[0], jitter[1], jitterPrev[0], jitterPrev[1]);
    }

    const bool reset = (m_accumBuffer.getSampleCount() == 1);
    m_nrd->setCommonSettings(
        viewToClip, viewToClipPrev, worldToView, worldToViewPrev,
        jitter, jitterPrev, mvScale, rw, rh, m_frameIndex, reset,
        m_lastCamera.farPlane);

    if (s_ppFrame < 3) LOG_INFO("recordPrePresent[%u]: about to NRD denoise", s_ppFrame);
    // Dispatch the denoiser (NRD may leave its output images in GENERAL).
    m_nrd->denoise(cmd, *m_sharedAux);
    if (s_ppFrame < 3) LOG_INFO("recordPrePresent[%u]: NRD denoise recorded", s_ppFrame);

    // Transition NRD outputs + albedo + emissive to SHADER_READ_ONLY for sampling.
    auto toRead = [&](VkImage img) {
        SharedVulkanImage::transition(
            cmd, img,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    };
    toRead(m_nrd->outDiffuseImage());
    toRead(m_nrd->outSpecularImage());
    toRead(m_sharedAux->albedo().image());
    toRead(m_sharedAux->emissive().image());

    const int tmMode = (m_toneMappingMode == ToneMappingMode::Reinhard) ? 1
                     : (m_toneMappingMode == ToneMappingMode::ACES)     ? 2 : 0;

    if (m_mode == Mode::NRDOnly && m_compositeRender && m_ldrRenderPass && m_ldrFramebuffer) {
        m_compositeRender->setInputs(
            m_nrd->outDiffuseView(),
            m_nrd->outSpecularView(),
            m_sharedAux->albedo().view(),
            m_sharedAux->emissive().view());

        VkExtent2D ext{ m_display->width(), m_display->height() };
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = m_ldrRenderPass;
        rbi.framebuffer = m_ldrFramebuffer;
        rbi.renderArea.offset = {0, 0};
        rbi.renderArea.extent = ext;
        rbi.clearValueCount = 0;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        m_compositeRender->record(cmd, ext, m_exposure, tmMode);
        vkCmdEndRenderPass(cmd);
        // Render pass final layout is SHADER_READ_ONLY — ready for the
        // swapchain blit pipeline.
    }
    else if (m_mode == Mode::NRDDLSS && m_dlss && m_compositeRender && m_tonemap) {
        // Ensure DLSS feature is created (once, on first pre-present after mode change).
        if (!m_dlss->isValid()) {
            m_dlss->createFeature(cmd, m_renderWidth, m_renderHeight,
                                  m_width, m_height,
                                  DLSSContext::BALANCED /* will be retargeted below */,
                                  /*isHDR=*/true);
        }

        // (1) Linear HDR composite @ render res.
        m_compositeRender->setInputs(
            m_nrd->outDiffuseView(),
            m_nrd->outSpecularView(),
            m_sharedAux->albedo().view(),
            m_sharedAux->emissive().view());
        VkExtent2D rext{ m_renderWidth, m_renderHeight };
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = m_hdrRenderPass;
        rbi.framebuffer = m_hdrRenderFb;
        rbi.renderArea.offset = {0, 0};
        rbi.renderArea.extent = rext;
        rbi.clearValueCount = 0;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        m_compositeRender->record(cmd, rext, /*exposure=*/1.0f, /*tm=*/0);  // no tonemap yet
        vkCmdEndRenderPass(cmd);
        // m_hdrRenderImage is now SHADER_READ_ONLY_OPTIMAL (render pass final layout).

        // (2) DLSS upscale. DLSS expects the output image in GENERAL layout.
        SharedVulkanImage::transition(cmd, m_hdrOutputImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, VK_ACCESS_SHADER_WRITE_BIT);
        // It also expects the inputs (color + motion + depth) in SHADER_READ_ONLY.
        // hdrRenderImage already transitioned by the render pass finalLayout.
        // Motion / viewZ are in SHADER_READ_ONLY from the NRD composite path above
        // — but we never transitioned them to SHADER_READ_ONLY. Do so now.
        SharedVulkanImage::transition(cmd, m_sharedAux->motionVectors().image(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        SharedVulkanImage::transition(cmd, m_sharedAux->viewZ().image(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        SharedVulkanImage::transition(cmd, m_sharedAux->ndcDepth().image(),
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        m_dlss->evaluate(cmd,
            m_hdrRenderView, m_hdrRenderImage,
            m_hdrOutputView, m_hdrOutputImage,
            m_sharedAux->motionVectors().view(), m_sharedAux->motionVectors().image(),
            // DLSS Super-Resolution needs post-perspective NDC depth
            // (clip.z/clip.w, [0,1]); it does NOT support linear viewZ. See
            // NRD-Sample/DlssBefore.cs.hlsl.
            m_sharedAux->ndcDepth().view(),     m_sharedAux->ndcDepth().image(),
            VK_FORMAT_R16G16B16A16_SFLOAT,      // color
            VK_FORMAT_R16G16_SFLOAT,            // motion
            VK_FORMAT_R32_SFLOAT,               // depth (NDC z in [0,1])
            m_renderWidth, m_renderHeight,
            m_width, m_height,
            m_lastCamera.jitterOffset.x, m_lastCamera.jitterOffset.y,
            /*reset=*/m_accumBuffer.getSampleCount() == 1);

        // (3) Tonemap @ output res → sampledImage.
        SharedVulkanImage::transition(cmd, m_hdrOutputImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        m_tonemap->setInputs(m_hdrOutputView);
        VkExtent2D sext{ m_width, m_height };
        VkRenderPassBeginInfo rbi2{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi2.renderPass = m_ldrRenderPass;
        rbi2.framebuffer = m_ldrFramebuffer;
        rbi2.renderArea.offset = {0, 0};
        rbi2.renderArea.extent = sext;
        rbi2.clearValueCount = 0;
        vkCmdBeginRenderPass(cmd, &rbi2, VK_SUBPASS_CONTENTS_INLINE);
        m_tonemap->record(cmd, sext, m_exposure, tmMode);
        vkCmdEndRenderPass(cmd);
    }

    m_lastCameraMoved = false;
    // Stash this frame's jitter for NRD's `cameraJitterPrev` next frame.
    m_prevJitter[0] = m_lastCamera.jitterOffset.x;
    m_prevJitter[1] = m_lastCamera.jitterOffset.y;
    if (s_ppFrame < 3) LOG_INFO("recordPrePresent[%u]: done", s_ppFrame);
    ++s_ppFrame;
}

// ────────────────────────────────────────────────────────────────
// DLSSOnly pre-present: skip NRD entirely. The path tracer wrote HDR
// directly into m_sharedAux->hdrColor() (render res); we DLSS-upscale
// into m_hdrOutputImage (output res) then tonemap into the swapchain
// sampledImage. No diff/spec/normal/albedo/emissive consumed.
// ────────────────────────────────────────────────────────────────
void Renderer::recordDlssOnlyPrePresent(VkCommandBuffer cmd) {
    if (!m_display || !m_dlss || !m_tonemap || !m_sharedAux || !m_lastCameraValid) return;

    static uint32_t s_dlssOnlyFrame = 0;
    const bool trace = s_dlssOnlyFrame < 3;

    // Lazily create the DLSS feature on the first frame after mode change.
    if (!m_dlss->isValid()) {
        m_dlss->createFeature(cmd, m_renderWidth, m_renderHeight,
                              m_width, m_height,
                              DLSSContext::BALANCED,  // (quality is retargeted via setDLSSQuality)
                              /*isHDR=*/true);
        if (trace) LOG_INFO("DLSSOnly: feature created (%ux%u → %ux%u)",
                            m_renderWidth, m_renderHeight, m_width, m_height);
    }

    // (1) Bring shared interop images into the layout DLSS expects:
    //   - color (hdrColor): SHADER_READ_ONLY
    //   - motion vectors  : SHADER_READ_ONLY
    //   - viewZ           : SHADER_READ_ONLY
    // CUDA wrote them via surface objects with no Vulkan layout tracking, so
    // we transition from UNDEFINED (contents preserved on most drivers but
    // formally undefined-by-spec — same trick as the NRD path).
    auto toRead = [&](VkImage img) {
        SharedVulkanImage::transition(cmd, img,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, VK_ACCESS_SHADER_READ_BIT);
    };
    toRead(m_sharedAux->hdrColor().image());
    toRead(m_sharedAux->motionVectors().image());
    toRead(m_sharedAux->ndcDepth().image());

    // (2) DLSS expects the upscaled output image in GENERAL.
    SharedVulkanImage::transition(cmd, m_hdrOutputImage,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, VK_ACCESS_SHADER_WRITE_BIT);

    m_dlss->evaluate(cmd,
        // Inputs: color (render res from path tracer), output (output res, DLSS write).
        m_sharedAux->hdrColor().view(),  m_sharedAux->hdrColor().image(),
        m_hdrOutputView,                 m_hdrOutputImage,
        m_sharedAux->motionVectors().view(), m_sharedAux->motionVectors().image(),
        // DLSS Super-Resolution needs NDC depth (clip.z/clip.w), not linear viewZ.
        m_sharedAux->ndcDepth().view(),      m_sharedAux->ndcDepth().image(),
        VK_FORMAT_R16G16B16A16_SFLOAT,   // color
        VK_FORMAT_R16G16_SFLOAT,         // motion
        VK_FORMAT_R32_SFLOAT,            // depth (NDC z in [0,1])
        m_renderWidth, m_renderHeight,
        m_width,       m_height,
        m_lastCamera.jitterOffset.x, m_lastCamera.jitterOffset.y,
        /*reset=*/m_accumBuffer.getSampleCount() == 1);

    // (3) Tonemap @ output res → sampledImage.
    SharedVulkanImage::transition(cmd, m_hdrOutputImage,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    const int tmMode = (m_toneMappingMode == ToneMappingMode::Reinhard) ? 1
                     : (m_toneMappingMode == ToneMappingMode::ACES)     ? 2 : 0;
    m_tonemap->setInputs(m_hdrOutputView);
    VkExtent2D sext{ m_width, m_height };
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass  = m_ldrRenderPass;
    rbi.framebuffer = m_ldrFramebuffer;
    rbi.renderArea.offset = {0, 0};
    rbi.renderArea.extent = sext;
    rbi.clearValueCount = 0;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    m_tonemap->record(cmd, sext, m_exposure, tmMode);
    vkCmdEndRenderPass(cmd);

    m_lastCameraMoved = false;
    m_prevJitter[0] = m_lastCamera.jitterOffset.x;
    m_prevJitter[1] = m_lastCamera.jitterOffset.y;
    if (trace) LOG_INFO("DLSSOnly: pre-present[%u] done", s_dlssOnlyFrame);
    ++s_dlssOnlyFrame;
}

#endif // PATHTRACER_NRD_DLSS_ENABLED
