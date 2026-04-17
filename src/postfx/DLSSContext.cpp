#include "postfx/DLSSContext.h"
#include "util/Log.h"

// NGX Vulkan headers.
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

#include <cstring>
#include <cstdlib>

namespace {

// NVDA requires a project ID. For non-production apps the SDK accepts an
// arbitrary UUID. Keep it stable across runs so NGX caches per-project
// telemetry consistently.
constexpr const char* kProjectId = "pathtracer-cs184-0f8dbc24";

NVSDK_NGX_PerfQuality_Value toNgxQuality(DLSSContext::QualityMode q) {
    switch (q) {
        case DLSSContext::PERFORMANCE: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        case DLSSContext::BALANCED:    return NVSDK_NGX_PerfQuality_Value_Balanced;
        case DLSSContext::QUALITY:     return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case DLSSContext::DLAA:        return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    return NVSDK_NGX_PerfQuality_Value_Balanced;
}

} // namespace

struct DLSSContext::Impl {
    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         device   = VK_NULL_HANDLE;

    bool                 ngxInitialized = false;
    NVSDK_NGX_Parameter* params         = nullptr;
    NVSDK_NGX_Handle*    dlss           = nullptr;

    // Cached configuration to detect when we need to recreate the feature.
    uint32_t   lastRenderW = 0, lastRenderH = 0;
    uint32_t   lastOutputW = 0, lastOutputH = 0;
    QualityMode lastQuality = BALANCED;

    ~Impl() {}
};

DLSSContext::DLSSContext() : m_impl(new Impl()) {}
DLSSContext::~DLSSContext() { shutdown(); delete m_impl; m_impl = nullptr; }

// C++-linkage free function exposed to VulkanDisplay.cpp so it can ask NGX
// for required extensions *before* we create VkInstance/VkDevice — without
// dragging NGX headers into VulkanDisplay's translation unit.
bool DLSSContext_QueryRequiredExts(
    std::vector<const char*>& instanceExts,
    std::vector<const char*>& deviceExts)
{
    return DLSSContext::queryRequiredExtensions(instanceExts, deviceExts);
}

bool DLSSContext::queryRequiredExtensions(
    std::vector<const char*>& instanceExts,
    std::vector<const char*>& deviceExts)
{
    unsigned int instCnt = 0;
    const char** instList = nullptr;
    unsigned int devCnt = 0;
    const char** devList = nullptr;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_RequiredExtensions(
        &instCnt, &instList, &devCnt, &devList);
    if (NVSDK_NGX_FAILED(r)) {
        LOG_WARN("DLSS: NVSDK_NGX_VULKAN_RequiredExtensions failed (0x%x)", (unsigned)r);
        return false;
    }
    for (unsigned i = 0; i < instCnt; ++i) instanceExts.push_back(instList[i]);
    for (unsigned i = 0; i < devCnt;  ++i) deviceExts.push_back(devList[i]);
    return true;
}

bool DLSSContext::init(VkInstance instance, VkPhysicalDevice phys, VkDevice device) {
    m_impl->instance = instance;
    m_impl->phys     = phys;
    m_impl->device   = device;

    // nvngx_dlss.dll is shipped next to the exe by our CMake POST_BUILD step.
    // Leave application data path null → NGX writes logs next to the DLL.
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId,
        NVSDK_NGX_ENGINE_TYPE_CUSTOM,
        "1.0",
        nullptr,      // application data path
        instance, phys, device);
    if (NVSDK_NGX_FAILED(r)) {
        LOG_WARN("DLSS: VULKAN_Init_with_ProjectID failed (0x%x)", (unsigned)r);
        return false;
    }
    m_impl->ngxInitialized = true;

    // Capability parameters (driver / hardware supports DLSS?).
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_impl->params))) {
        LOG_WARN("DLSS: GetCapabilityParameters failed");
        return false;
    }
    int dlssAvail = 0;
    NVSDK_NGX_Parameter_GetI(m_impl->params,
        NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvail);
    if (!dlssAvail) {
        LOG_WARN("DLSS: not available on this GPU / driver");
        return false;
    }
    return true;
}

bool DLSSContext::getOptimalRenderResolution(
    uint32_t outputW, uint32_t outputH, QualityMode quality,
    uint32_t& renderW, uint32_t& renderH)
{
    if (!m_impl || !m_impl->params) return false;
    unsigned int optRenderW = 0, optRenderH = 0;
    unsigned int renderWMin = 0, renderHMin = 0;
    unsigned int renderWMax = 0, renderHMax = 0;
    float sharpness = 0.0f;
    NVSDK_NGX_Result r = NGX_DLSS_GET_OPTIMAL_SETTINGS(
        m_impl->params, outputW, outputH, toNgxQuality(quality),
        &optRenderW, &optRenderH,
        &renderWMax, &renderHMax,
        &renderWMin, &renderHMin,
        &sharpness);
    if (NVSDK_NGX_FAILED(r) || optRenderW == 0) return false;
    renderW = optRenderW;
    renderH = optRenderH;
    return true;
}

bool DLSSContext::createFeature(
    VkCommandBuffer cmd,
    uint32_t renderW, uint32_t renderH,
    uint32_t outputW, uint32_t outputH,
    QualityMode quality,
    bool isHDR)
{
    if (!m_impl || !m_impl->ngxInitialized || !m_impl->params) return false;

    // Release previous feature if present.
    if (m_impl->dlss) {
        NVSDK_NGX_VULKAN_ReleaseFeature(m_impl->dlss);
        m_impl->dlss = nullptr;
    }

    NVSDK_NGX_DLSS_Create_Params createParams{};
    createParams.Feature.InWidth        = renderW;
    createParams.Feature.InHeight       = renderH;
    createParams.Feature.InTargetWidth  = outputW;
    createParams.Feature.InTargetHeight = outputH;
    createParams.Feature.InPerfQualityValue = toNgxQuality(quality);

    int flags = 0;
    flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR *      (isHDR ? 1 : 0);
    flags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;    // our MVs are at render (low) res
    flags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted * 0; // we use linear view-space Z, not inverted clip
    // Do NOT request jittered MVs or auto-exposure.
    createParams.InFeatureCreateFlags = flags;

    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(
        cmd, /*creationNodeMask=*/1, /*visibilityNodeMask=*/1,
        &m_impl->dlss, m_impl->params, &createParams);
    if (NVSDK_NGX_FAILED(r)) {
        LOG_WARN("DLSS: CREATE_DLSS_EXT failed (0x%x)", (unsigned)r);
        return false;
    }
    m_impl->lastRenderW = renderW;
    m_impl->lastRenderH = renderH;
    m_impl->lastOutputW = outputW;
    m_impl->lastOutputH = outputH;
    m_impl->lastQuality = quality;
    return true;
}

void DLSSContext::evaluate(
    VkCommandBuffer cmd,
    VkImageView  inColor,  VkImage inColorImage,
    VkImageView  outColor, VkImage outColorImage,
    VkImageView  motion,   VkImage motionImage,
    VkImageView  depth,    VkImage depthImage,
    VkFormat     colorFormat, VkFormat motionFormat, VkFormat depthFormat,
    uint32_t     renderW, uint32_t renderH,
    uint32_t     outputW, uint32_t outputH,
    float        jitterX, float jitterY,
    bool         reset)
{
    if (!m_impl || !m_impl->dlss) return;

    VkImageSubresourceRange srr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    NVSDK_NGX_Resource_VK rIn     = NVSDK_NGX_Create_ImageView_Resource_VK(inColor,     inColorImage,  srr, colorFormat,  renderW, renderH, /*readWrite=*/false);
    NVSDK_NGX_Resource_VK rOut    = NVSDK_NGX_Create_ImageView_Resource_VK(outColor,    outColorImage, srr, colorFormat,  outputW, outputH, /*readWrite=*/true);
    NVSDK_NGX_Resource_VK rMotion = NVSDK_NGX_Create_ImageView_Resource_VK(motion,      motionImage,   srr, motionFormat, renderW, renderH, /*readWrite=*/false);
    NVSDK_NGX_Resource_VK rDepth  = NVSDK_NGX_Create_ImageView_Resource_VK(depth,       depthImage,    srr, depthFormat,  renderW, renderH, /*readWrite=*/false);

    NVSDK_NGX_VK_DLSS_Eval_Params eval{};
    eval.Feature.pInColor  = &rIn;
    eval.Feature.pInOutput = &rOut;
    eval.Feature.InSharpness = 0.0f;
    eval.pInDepth          = &rDepth;
    eval.pInMotionVectors  = &rMotion;
    eval.InJitterOffsetX   = jitterX;
    eval.InJitterOffsetY   = jitterY;
    eval.InRenderSubrectDimensions = { renderW, renderH };
    eval.InReset           = reset ? 1 : 0;
    eval.InMVScaleX        = 1.0f;   // our MVs are in render-pixel space
    eval.InMVScaleY        = 1.0f;
    eval.InPreExposure     = 1.0f;

    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, m_impl->dlss, m_impl->params, &eval);
    if (NVSDK_NGX_FAILED(r)) {
        LOG_WARN("DLSS: EVALUATE failed (0x%x)", (unsigned)r);
    }
}

bool DLSSContext::isValid() const {
    return m_impl && m_impl->ngxInitialized && m_impl->dlss != nullptr;
}

void DLSSContext::shutdown() {
    if (!m_impl) return;
    if (m_impl->dlss) {
        NVSDK_NGX_VULKAN_ReleaseFeature(m_impl->dlss);
        m_impl->dlss = nullptr;
    }
    if (m_impl->params) {
        NVSDK_NGX_VULKAN_DestroyParameters(m_impl->params);
        m_impl->params = nullptr;
    }
    if (m_impl->ngxInitialized) {
        NVSDK_NGX_VULKAN_Shutdown1(m_impl->device);
        m_impl->ngxInitialized = false;
    }
    m_impl->device = VK_NULL_HANDLE;
    m_impl->phys = VK_NULL_HANDLE;
    m_impl->instance = VK_NULL_HANDLE;
}
