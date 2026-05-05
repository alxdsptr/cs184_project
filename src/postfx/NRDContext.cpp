#include "postfx/NRDContext.h"
#include "render/VulkanSharedAuxBuffers.h"
#include "display/VulkanDisplay.h"
#include "util/Log.h"

// ── NRI / NRD SDK glue ──────────────────────────────────────────
// NRD headers must come before NRIWrapperVK.h so the integration layer sees
// the SDK version macros. Integration/NRDIntegration.h requires NRD.h + NRI.h
// + NRIHelper.h already in scope.
#include <NRD.h>
#include <NRI.h>
#include <Extensions/NRIHelper.h>
// NRIWrapperVK uses AccelerationStructureBits from NRIRayTracing.h.
#include <Extensions/NRIRayTracing.h>
#include <Extensions/NRIWrapperVK.h>
#include <NRDIntegration.h>

// NRDIntegration.hpp is the implementation; include it exactly once in the
// whole program (this translation unit is that once).
#include <NRDIntegration.hpp>

#include <array>
#include <cstring>
#include <vector>
#include <memory>

// ────────────────────────────────────────────────────────────────
// Output image helper (pure Vulkan, no CUDA interop needed).
// NRD writes OUT_DIFF_RADIANCE_HITDIST / OUT_SPEC_RADIANCE_HITDIST into these.
// ────────────────────────────────────────────────────────────────
namespace {

struct OutputVkImage {
    VkDevice        device = VK_NULL_HANDLE;
    VkImage         image  = VK_NULL_HANDLE;
    VkDeviceMemory  memory = VK_NULL_HANDLE;
    VkImageView     view   = VK_NULL_HANDLE;

    bool create(VkDevice dev, VkPhysicalDevice phys,
                uint32_t w, uint32_t h, VkFormat format)
    {
        device = dev;

        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = format;
        ici.extent        = { w, h, 1 };
        ici.mipLevels     = 1;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                          | VK_IMAGE_USAGE_SAMPLED_BIT
                          | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ici, nullptr, &image) != VK_SUCCESS) return false;

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device, image, &req);
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        uint32_t memType = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                memType = i; break;
            }
        }
        if (memType == UINT32_MAX) return false;

        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = memType;
        if (vkAllocateMemory(device, &mai, nullptr, &memory) != VK_SUCCESS) return false;
        if (vkBindImageMemory(device, image, memory, 0) != VK_SUCCESS) return false;

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(device, &vci, nullptr, &view) != VK_SUCCESS) return false;
        return true;
    }

    void destroy() {
        if (view && device)   vkDestroyImageView(device, view, nullptr);
        if (image && device)  vkDestroyImage(device, image, nullptr);
        if (memory && device) vkFreeMemory(device, memory, nullptr);
        view = VK_NULL_HANDLE; image = VK_NULL_HANDLE; memory = VK_NULL_HANDLE;
    }
};

// NRD assigns an arbitrary Identifier per denoiser. We only have one (RELAX).
constexpr nrd::Identifier kRelaxId = 0;

// Wrap an already-existing VkImage into an NRI Texture.
nri::Texture* wrapVkImage(
    nri::Device* device,
    const nri::WrapperVKInterface& wrap,
    VkImage vkImg, VkFormat vkFmt, uint32_t w, uint32_t h,
    VkImageUsageFlags usage)
{
    nri::TextureVKDesc d{};
    d.vkImage           = reinterpret_cast<uint64_t>(vkImg);
    d.vkFormat          = (int32_t)vkFmt;
    d.vkImageType       = (int32_t)VK_IMAGE_TYPE_2D;
    d.vkImageUsageFlags = (uint32_t)usage;
    d.width             = (uint16_t)w;
    d.height            = (uint16_t)h;
    d.depth             = 1;
    d.mipNum            = 1;
    d.layerNum          = 1;
    d.sampleNum         = 1;
    nri::Texture* tex = nullptr;
    if (wrap.CreateTextureVK(*device, d, tex) != nri::Result::SUCCESS) return nullptr;
    return tex;
}

} // anonymous namespace

// ────────────────────────────────────────────────────────────────
// Impl — owns NRI device + NRD Integration + output Vk images
// ────────────────────────────────────────────────────────────────
struct NRDContext::Impl {
    VkDevice         vkDevice   = VK_NULL_HANDLE;
    VkPhysicalDevice vkPhys     = VK_NULL_HANDLE;
    VkInstance       vkInstance = VK_NULL_HANDLE;
    uint32_t         queueFamily = 0;
    VkQueue          queue      = VK_NULL_HANDLE;

    nri::Device*           nriDevice = nullptr;
    nri::CoreInterface     nriCore{};
    nri::HelperInterface   nriHelper{};
    nri::WrapperVKInterface nriWrapVk{};

    nrd::Integration       nrd;
    bool                   integrationAlive = false;

    // NRD frame settings carried across calls.
    nrd::CommonSettings    common{};

    // App-owned OUTPUT VkImages (RELAX writes into these).
    OutputVkImage          outDiff;
    OutputVkImage          outSpec;
    nri::Texture*          outDiffNri = nullptr;
    nri::Texture*          outSpecNri = nullptr;

    // NRI-wrapped input textures, re-created each frame if images change.
    // Cached by VkImage handle to avoid recreating every frame when the
    // shared aux buffers don't move.
    struct InputWraps {
        VkImage diff = VK_NULL_HANDLE, spec = VK_NULL_HANDLE;
        VkImage normRough = VK_NULL_HANDLE, viewZ = VK_NULL_HANDLE;
        VkImage mv = VK_NULL_HANDLE;
        nri::Texture* diffNri = nullptr;
        nri::Texture* specNri = nullptr;
        nri::Texture* normRoughNri = nullptr;
        nri::Texture* viewZNri = nullptr;
        nri::Texture* mvNri = nullptr;
    } inputs;

    uint32_t renderW = 0, renderH = 0;
    uint32_t frameCounter = 0;

    bool initNri(const VulkanDisplay& display) {
        vkInstance  = display.instance();
        vkPhys      = display.physicalDevice();
        vkDevice    = display.device();
        queue       = display.graphicsQueue();
        queueFamily = display.graphicsQueueFamily();

        const auto& instExts = display.enabledInstanceExtensions();
        const auto& devExts  = display.enabledDeviceExtensions();

        nri::DeviceCreationVKDesc d{};
        d.vkInstance       = (VKHandle)vkInstance;
        d.vkDevice         = (VKHandle)vkDevice;
        d.vkPhysicalDevice = (VKHandle)vkPhys;
        // NRI always queries core-name Vulkan 1.3 entry points (e.g.
        // vkCmdCopyBuffer2), so the underlying VkInstance/VkDevice *must*
        // be at least 1.3 — see VulkanDisplay::createInstance(). Passing 2
        // here makes NRI assume 1.2, its dispatch-table lookup fails
        // immediately with "Failed to get device function: 'vkCmdCopyBuffer2'".
        d.minorVersion     = 3;   // VK 1.3

        // Surface NRI's internal warnings/errors through our log — otherwise any
        // failure inside `DenoiseVK` is completely silent in Release builds.
        // We also MUST override AbortExecution: NRI's default implementation is
        // `DebugBreak()`, which terminates a Release process (no debugger attached)
        // *before* our MessageCallback output reaches disk. Replacing it with a
        // no-op (combined with fflush below) lets us actually read the error.
        d.callbackInterface.MessageCallback = [](nri::Message msgType, const char* file,
                                                 uint32_t line, const char* message, void*) {
            const char* tag = (msgType == nri::Message::ERROR)   ? "NRI-ERR"
                            : (msgType == nri::Message::WARNING) ? "NRI-WARN"
                                                                 : "NRI-INFO";
            LOG_INFO("[%s] %s:%u %s", tag, file ? file : "?", line, message ? message : "(null)");
            std::fflush(stdout);
            std::fflush(stderr);
        };
        d.callbackInterface.AbortExecution = [](void*) { /* swallow: don't DebugBreak */ };

        // Turn on NRI's embedded validation. If the SDK's layer is available,
        // also turn on Vulkan-level validation — that's what surfaces things like
        // "image used in layout X but expected Y" or "descriptor bound to wrong type".
        d.enableNRIValidation = true;

        // Exposed extensions (NRI needs to know which are actually enabled).
        d.vkExtensions.instanceExtensions    = instExts.data();
        d.vkExtensions.instanceExtensionNum  = (uint32_t)instExts.size();
        d.vkExtensions.deviceExtensions      = devExts.data();
        d.vkExtensions.deviceExtensionNum    = (uint32_t)devExts.size();

        // Queue family info (one GRAPHICS queue on `queueFamily`).
        nri::QueueFamilyVKDesc qfam{};
        qfam.queueNum    = 1;
        qfam.queueType   = nri::QueueType::GRAPHICS;
        qfam.familyIndex = queueFamily;
        d.queueFamilies   = &qfam;
        d.queueFamilyNum  = 1;

        if (nri::nriCreateDeviceFromVKDevice(d, nriDevice) != nri::Result::SUCCESS) {
            LOG_ERROR("NRI: nriCreateDeviceFromVKDevice failed");
            return false;
        }

        if (nri::nriGetInterface(*nriDevice, NRI_INTERFACE(nri::CoreInterface),
                                 &nriCore) != nri::Result::SUCCESS) return false;
        if (nri::nriGetInterface(*nriDevice, NRI_INTERFACE(nri::HelperInterface),
                                 &nriHelper) != nri::Result::SUCCESS) return false;
        if (nri::nriGetInterface(*nriDevice, NRI_INTERFACE(nri::WrapperVKInterface),
                                 &nriWrapVk) != nri::Result::SUCCESS) return false;
        return true;
    }

    bool createOutputs(uint32_t w, uint32_t h) {
        outDiff.destroy();
        outSpec.destroy();

        // RELAX outputs use RGBA16F per NRDDescs (we just take the pool default).
        const VkFormat kFmt = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (!outDiff.create(vkDevice, vkPhys, w, h, kFmt)) return false;
        if (!outSpec.create(vkDevice, vkPhys, w, h, kFmt)) return false;

        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (outDiffNri) nriCore.DestroyTexture(outDiffNri);
        if (outSpecNri) nriCore.DestroyTexture(outSpecNri);
        outDiffNri = wrapVkImage(nriDevice, nriWrapVk, outDiff.image, kFmt, w, h, usage);
        outSpecNri = wrapVkImage(nriDevice, nriWrapVk, outSpec.image, kFmt, w, h, usage);
        return outDiffNri && outSpecNri;
    }

    void destroyInputWraps() {
        if (inputs.diffNri)      nriCore.DestroyTexture(inputs.diffNri);
        if (inputs.specNri)      nriCore.DestroyTexture(inputs.specNri);
        if (inputs.normRoughNri) nriCore.DestroyTexture(inputs.normRoughNri);
        if (inputs.viewZNri)     nriCore.DestroyTexture(inputs.viewZNri);
        if (inputs.mvNri)        nriCore.DestroyTexture(inputs.mvNri);
        inputs = InputWraps{};
    }

    bool ensureInputWraps(const VulkanSharedAuxBuffers& aux) {
        // Re-wrap only if underlying VkImages changed (e.g., after a resize).
        VkImage d = aux.diffuseRadianceHitDist().image();
        VkImage s = aux.specularRadianceHitDist().image();
        VkImage nr = aux.normalRoughness().image();
        VkImage vz = aux.viewZ().image();
        VkImage mv = aux.motionVectors().image();
        if (d == inputs.diff && s == inputs.spec && nr == inputs.normRough &&
            vz == inputs.viewZ && mv == inputs.mv && inputs.diffNri) {
            return true;
        }
        destroyInputWraps();
        inputs.diff = d; inputs.spec = s;
        inputs.normRough = nr; inputs.viewZ = vz; inputs.mv = mv;
        const uint32_t w = aux.width(), h = aux.height();
        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        inputs.diffNri = wrapVkImage(nriDevice, nriWrapVk, d,
            VK_FORMAT_R16G16B16A16_SFLOAT, w, h, usage);
        inputs.specNri = wrapVkImage(nriDevice, nriWrapVk, s,
            VK_FORMAT_R16G16B16A16_SFLOAT, w, h, usage);
        inputs.normRoughNri = wrapVkImage(nriDevice, nriWrapVk, nr,
            VK_FORMAT_R8G8B8A8_UNORM, w, h, usage);
        inputs.viewZNri = wrapVkImage(nriDevice, nriWrapVk, vz,
            VK_FORMAT_R32_SFLOAT, w, h, usage);
        inputs.mvNri = wrapVkImage(nriDevice, nriWrapVk, mv,
            VK_FORMAT_R16G16_SFLOAT, w, h, usage);
        return inputs.diffNri && inputs.specNri && inputs.normRoughNri &&
               inputs.viewZNri && inputs.mvNri;
    }

    bool recreateIntegration(uint32_t w, uint32_t h) {
        // NRD settings: one denoiser (RELAX_DIFFUSE_SPECULAR).
        nrd::DenoiserDesc denoiser{};
        denoiser.identifier = kRelaxId;
        denoiser.denoiser   = nrd::Denoiser::RELAX_DIFFUSE_SPECULAR;

        nrd::InstanceCreationDesc instDesc{};
        instDesc.denoisers     = &denoiser;
        instDesc.denoisersNum  = 1;

        nrd::IntegrationCreationDesc integDesc{};
        std::strncpy(integDesc.name, "PathtracerNRD", sizeof(integDesc.name) - 1);
        integDesc.resourceWidth   = (uint16_t)w;
        integDesc.resourceHeight  = (uint16_t)h;
        integDesc.queuedFrameNum  = 2;
        integDesc.autoWaitForIdle = true;
        integDesc.residencyPriority = 0.0f;
        integDesc.demoteFloat32to16 = false;
        integDesc.promoteFloat16to32 = false;
        integDesc.enableWholeLifetimeDescriptorCaching = false;

        if (integrationAlive) {
            nrd.Destroy();
            integrationAlive = false;
        }
        if (nrd.Recreate(integDesc, instDesc, nriDevice) != nrd::Result::SUCCESS) {
            LOG_ERROR("NRD Integration::Recreate failed");
            return false;
        }
        integrationAlive = true;
        renderW = w; renderH = h;
        return true;
    }
};

// ────────────────────────────────────────────────────────────────
// NRDContext public API
// ────────────────────────────────────────────────────────────────
NRDContext::NRDContext() : m_impl(new Impl()) {}
NRDContext::~NRDContext() { shutdown(); delete m_impl; m_impl = nullptr; }

bool NRDContext::init(const VulkanDisplay& display, uint32_t renderW, uint32_t renderH) {
    if (!m_impl->initNri(display)) return false;
    if (!m_impl->recreateIntegration(renderW, renderH)) return false;
    if (!m_impl->createOutputs(renderW, renderH)) return false;
    return true;
}

bool NRDContext::resize(uint32_t renderW, uint32_t renderH) {
    if (!m_impl || !m_impl->nriDevice) return false;
    if (renderW == m_impl->renderW && renderH == m_impl->renderH) return true;
    // Output VkImages + NRD internal pool must be rebuilt.
    if (!m_impl->recreateIntegration(renderW, renderH)) return false;
    if (!m_impl->createOutputs(renderW, renderH)) return false;
    // Input wraps will be rebuilt lazily when they're next needed.
    m_impl->destroyInputWraps();
    // Re-trigger one-shot RelaxSettings apply on next denoise() — the new
    // integration starts fresh with default RELAX settings.
    m_impl->frameCounter = 0;
    return true;
}

void NRDContext::setCommonSettings(
    const float viewToClip[16], const float viewToClipPrev[16],
    const float worldToView[16], const float worldToViewPrev[16],
    float cameraJitter[2], float cameraJitterPrev[2],
    float motionVectorScalePx[2],
    uint32_t w, uint32_t h,
    uint32_t frameIndex, bool reset,
    float denoisingRange)
{
    auto& c = m_impl->common;
    std::memcpy(c.viewToClipMatrix,      viewToClip,      sizeof(float)*16);
    std::memcpy(c.viewToClipMatrixPrev,  viewToClipPrev,  sizeof(float)*16);
    std::memcpy(c.worldToViewMatrix,     worldToView,     sizeof(float)*16);
    std::memcpy(c.worldToViewMatrixPrev, worldToViewPrev, sizeof(float)*16);

    // NRD asserts that jitter is in [-0.5, 0.5]. Our Halton sequence is
    // theoretically inside that range, but clamp defensively in case a caller
    // passes jitter already scaled to e.g. pixel/UV units by mistake.
    auto clamp05 = [](float v) { return v < -0.5f ? -0.5f : (v > 0.5f ? 0.5f : v); };
    c.cameraJitter[0]     = clamp05(cameraJitter[0]);
    c.cameraJitter[1]     = clamp05(cameraJitter[1]);
    c.cameraJitterPrev[0] = clamp05(cameraJitterPrev[0]);
    c.cameraJitterPrev[1] = clamp05(cameraJitterPrev[1]);
    c.motionVectorScale[0] = motionVectorScalePx[0];
    c.motionVectorScale[1] = motionVectorScalePx[1];
    c.motionVectorScale[2] = 0.0f;
    c.resourceSize[0]     = (uint16_t)w;
    c.resourceSize[1]     = (uint16_t)h;
    c.resourceSizePrev[0] = (uint16_t)w;
    c.resourceSizePrev[1] = (uint16_t)h;
    c.rectSize[0]         = (uint16_t)w;
    c.rectSize[1]         = (uint16_t)h;
    c.rectSizePrev[0]     = (uint16_t)w;
    c.rectSizePrev[1]     = (uint16_t)h;
    c.frameIndex          = frameIndex;
    c.accumulationMode    = reset ? nrd::AccumulationMode::RESTART
                                  : nrd::AccumulationMode::CONTINUE;
    c.isMotionVectorInWorldSpace = false;
    // viewZScale stays at 1.0 (default): we write linear meters into IN_VIEWZ.
    // denoisingRange clipped to a sane minimum so a degenerate camera
    // farPlane (0/NaN) doesn't disable denoising entirely.
    c.denoisingRange = (denoisingRange > 1.0f) ? denoisingRange : 1000.0f;

    // One-shot snapshot of every field NRD validates, so the next failure
    // (if any) is instantly diagnosable from log.txt.
    static uint32_t s_dumped = 0;
    if (s_dumped < 2) {
        LOG_INFO("NRD.setCommonSettings[%u]: jitter=(%.4f,%.4f) prev=(%.4f,%.4f)",
                 s_dumped, c.cameraJitter[0], c.cameraJitter[1],
                 c.cameraJitterPrev[0], c.cameraJitterPrev[1]);
        LOG_INFO("  mvScale=(%.6f,%.6f,%.6f) rw/rh=(%u,%u) frame=%u accum=%d",
                 c.motionVectorScale[0], c.motionVectorScale[1], c.motionVectorScale[2],
                 (unsigned)c.resourceSize[0], (unsigned)c.resourceSize[1],
                 c.frameIndex, (int)c.accumulationMode);
        LOG_INFO("  viewZScale=%.4f denoisingRange=%.1f disoccT=%.4f",
                 c.viewZScale, c.denoisingRange, c.disocclusionThreshold);
        ++s_dumped;
    }
}

void NRDContext::denoise(VkCommandBuffer cmd, const VulkanSharedAuxBuffers& aux) {
    if (!m_impl->integrationAlive) return;

    const bool trace = m_impl->frameCounter < 3;
    if (trace) LOG_INFO("NRD.denoise[%u]: NewFrame", m_impl->frameCounter);
    m_impl->nrd.NewFrame();
    if (trace) LOG_INFO("NRD.denoise[%u]: SetCommonSettings", m_impl->frameCounter);
    m_impl->nrd.SetCommonSettings(m_impl->common);

    // RELAX denoiser settings — one-shot (settings are sticky across frames in
    // the integration). Re-apply if the integration was recreated (resize).
    //
    // Tuning mirrors RTXPT's NrdConfig::getDefaultRELAXSettings(): they ship a
    // path tracer with the same probabilistic-bucket pattern we use, and their
    // tuning kills shimmer that the NRD defaults leave in.  Notable choices:
    //   - hitDistRecon = OFF (NOT AREA_3X3). With AREA_3X3 the per-pixel hitT
    //     gets averaged across the 3×3 spatial neighborhood every frame,
    //     coupling neighbor luminance into the local hitT estimate. On a static
    //     bucket-flickering input that injects neighbor-correlated jitter that
    //     reads as "shifting noise patches" — exactly what the user reported.
    //   - prepassBlurRadius = 0. The prepass spatial reuse helps under heavy
    //     bucket-flicker but its kernel size depends on hitT, which itself is
    //     flickering — so it visibly modulates each frame. RTXPT's comment is
    //     blunt: "using prepass blur causes more issues than it solves".
    //   - specularLobeAngleSlack = 0.2 hides noisy secondary bounces (per
    //     RTXPT comment) — good for our 4–6 bounce paths.
    //   - higher specular accumulation (40 frames) since specular signal is
    //     intrinsically lower-frequency.
    if (m_impl->frameCounter == 0) {
        nrd::RelaxSettings relax{};
        relax.enableAntiFirefly = true;
        relax.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::OFF;
        relax.diffusePrepassBlurRadius      = 0.0f;
        relax.specularPrepassBlurRadius     = 0.0f;
        relax.atrousIterationNum            = 5;
        relax.lobeAngleFraction             = 0.7f;
        relax.specularLobeAngleSlack        = 0.2f;
        relax.depthThreshold                = 0.004f;
        relax.diffuseMaxAccumulatedFrameNum     = 25;
        relax.specularMaxAccumulatedFrameNum    = 40;
        relax.diffuseMaxFastAccumulatedFrameNum = 5;
        relax.specularMaxFastAccumulatedFrameNum = 6;
        relax.antilagSettings.accelerationAmount = 0.55f;
        relax.antilagSettings.spatialSigmaScale  = 2.5f;
        relax.antilagSettings.temporalSigmaScale = 0.3f;
        relax.antilagSettings.resetAmount        = 0.5f;
        if (m_impl->nrd.SetDenoiserSettings(kRelaxId, &relax) != nrd::Result::SUCCESS) {
            LOG_ERROR("NRD.denoise: SetDenoiserSettings(RELAX) failed");
        } else {
            LOG_INFO("NRD.denoise: RELAX settings applied "
                     "(RTXPT-tuned: hitDistRecon=OFF, prepassBlur=0, accum=25/40)");
        }
    }

    // We own the NRI device (initNri), so we go through the plain `Denoise()`
    // path with NRI-wrapped textures — NOT `DenoiseVK`, which is reserved for
    // the `RecreateVK`-owned-device flow and asserts on `m_Wrapped == VK`.
    if (trace) LOG_INFO("NRD.denoise[%u]: ensureInputWraps", m_impl->frameCounter);
    if (!m_impl->ensureInputWraps(aux)) {
        LOG_ERROR("NRD.denoise[%u]: ensureInputWraps failed", m_impl->frameCounter);
        return;
    }

    nri::AccessLayoutStage inputState{};
    inputState.access = nri::AccessBits::SHADER_RESOURCE_STORAGE;
    inputState.layout = nri::Layout::GENERAL;
    inputState.stages = nri::StageBits::COMPUTE_SHADER;

    auto makeRes = [&](nri::Texture* tex) {
        nrd::Resource r{};
        r.nri.texture = tex;
        r.state       = inputState;
        return r;
    };
    nrd::Resource rDiff  = makeRes(m_impl->inputs.diffNri);
    nrd::Resource rSpec  = makeRes(m_impl->inputs.specNri);
    nrd::Resource rNorm  = makeRes(m_impl->inputs.normRoughNri);
    nrd::Resource rViewZ = makeRes(m_impl->inputs.viewZNri);
    nrd::Resource rMv    = makeRes(m_impl->inputs.mvNri);
    nrd::Resource rODiff = makeRes(m_impl->outDiffNri);
    nrd::Resource rOSpec = makeRes(m_impl->outSpecNri);

    nrd::ResourceSnapshot snap;
    snap.SetResource(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST,  rDiff);
    snap.SetResource(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST,  rSpec);
    snap.SetResource(nrd::ResourceType::IN_NORMAL_ROUGHNESS,       rNorm);
    snap.SetResource(nrd::ResourceType::IN_VIEWZ,                  rViewZ);
    snap.SetResource(nrd::ResourceType::IN_MV,                     rMv);
    snap.SetResource(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, rODiff);
    snap.SetResource(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, rOSpec);
    snap.restoreInitialState = true;

    // Wrap the raw VkCommandBuffer into an NRI CommandBuffer object so
    // `Denoise()` can record into it. Must be destroyed after the call —
    // the wrapper is non-owning and cheap to create.
    nri::CommandBufferVKDesc cbDesc{};
    cbDesc.vkCommandBuffer = (VKHandle)cmd;
    cbDesc.queueType       = nri::QueueType::GRAPHICS;
    nri::CommandBuffer* nriCmd = nullptr;
    nri::Result cbRes = m_impl->nriWrapVk.CreateCommandBufferVK(
        *m_impl->nriDevice, cbDesc, nriCmd);
    if (cbRes != nri::Result::SUCCESS || !nriCmd) {
        LOG_ERROR("NRD.denoise[%u]: CreateCommandBufferVK failed (%d)",
                  m_impl->frameCounter, (int)cbRes);
        return;
    }

    const nrd::Identifier denoisers[] = { kRelaxId };
    if (trace) { LOG_INFO("NRD.denoise[%u]: calling Denoise", m_impl->frameCounter); std::fflush(stdout); }
    m_impl->nrd.Denoise(denoisers, 1, *nriCmd, snap);
    if (trace) { LOG_INFO("NRD.denoise[%u]: Denoise returned", m_impl->frameCounter); std::fflush(stdout); }

    m_impl->nriCore.DestroyCommandBuffer(nriCmd);

    m_impl->frameCounter++;
}

VkImage     NRDContext::outDiffuseImage()  const { return m_impl ? m_impl->outDiff.image  : VK_NULL_HANDLE; }
VkImageView NRDContext::outDiffuseView()   const { return m_impl ? m_impl->outDiff.view   : VK_NULL_HANDLE; }
VkImage     NRDContext::outSpecularImage() const { return m_impl ? m_impl->outSpec.image  : VK_NULL_HANDLE; }
VkImageView NRDContext::outSpecularView()  const { return m_impl ? m_impl->outSpec.view   : VK_NULL_HANDLE; }
bool        NRDContext::isValid() const { return m_impl && m_impl->integrationAlive; }

void NRDContext::shutdown() {
    if (!m_impl) return;
    if (m_impl->integrationAlive) {
        m_impl->nrd.Destroy();
        m_impl->integrationAlive = false;
    }
    m_impl->destroyInputWraps();
    if (m_impl->outDiffNri) m_impl->nriCore.DestroyTexture(m_impl->outDiffNri);
    if (m_impl->outSpecNri) m_impl->nriCore.DestroyTexture(m_impl->outSpecNri);
    m_impl->outDiffNri = nullptr;
    m_impl->outSpecNri = nullptr;
    m_impl->outDiff.destroy();
    m_impl->outSpec.destroy();
    if (m_impl->nriDevice) {
        nri::nriDestroyDevice(m_impl->nriDevice);
        m_impl->nriDevice = nullptr;
    }
}
