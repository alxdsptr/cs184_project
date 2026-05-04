#include "postfx/DLSSDContext.h"
#include "util/Log.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>

#include <cstring>
#include <string>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

namespace {

// Same project ID as DLSSContext (NGX requires GUID format). Sharing the
// project ID across DLSS-SR / DLSS-RR keeps NGX's per-project caches coherent.
constexpr const char* kProjectId = "0f8dbc24-c5c8-4f9b-9e4a-2d1b3a7c6e5f";

NVSDK_NGX_PerfQuality_Value toNgxQuality(DLSSDContext::QualityMode q) {
    switch (q) {
        case DLSSDContext::PERFORMANCE: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        case DLSSDContext::BALANCED:    return NVSDK_NGX_PerfQuality_Value_Balanced;
        case DLSSDContext::QUALITY:     return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case DLSSDContext::DLAA:        return NVSDK_NGX_PerfQuality_Value_DLAA;
    }
    return NVSDK_NGX_PerfQuality_Value_Balanced;
}

} // namespace

struct DLSSDContext::Impl {
    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         device   = VK_NULL_HANDLE;

    bool                 ngxInitialized = false;
    NVSDK_NGX_Parameter* params         = nullptr;
    NVSDK_NGX_Handle*    feature        = nullptr;
};

DLSSDContext::DLSSDContext() : m_impl(new Impl()) {}
DLSSDContext::~DLSSDContext() { shutdown(); delete m_impl; m_impl = nullptr; }

bool DLSSDContext::init(VkInstance instance, VkPhysicalDevice phys, VkDevice device) {
    m_impl->instance = instance;
    m_impl->phys     = phys;
    m_impl->device   = device;

    // NGX writable AppData path — pick the exe directory, same fallback used
    // by DLSSContext to avoid the LOCALAPPDATA pitfall.
    std::wstring appDataPath;
#ifdef _WIN32
    wchar_t exePath[MAX_PATH] = {};
    DWORD n = GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        std::wstring p(exePath, n);
        auto slash = p.find_last_of(L"\\/");
        if (slash != std::wstring::npos) appDataPath = p.substr(0, slash);
    }
#endif
    const wchar_t* appDataArg = appDataPath.empty() ? nullptr : appDataPath.c_str();

    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId,
        NVSDK_NGX_ENGINE_TYPE_CUSTOM,
        "1.0",
        appDataArg,
        instance, phys, device);
    if (NVSDK_NGX_FAILED(r)) {
        LOG_WARN("DLSS-RR: VULKAN_Init_with_ProjectID failed (0x%x)", (unsigned)r);
        return false;
    }
    m_impl->ngxInitialized = true;

    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_impl->params))) {
        LOG_WARN("DLSS-RR: GetCapabilityParameters failed");
        return false;
    }

    // RR has its own capability flag (separate from DLSS-SR availability).
    int rrAvail = 0;
    NVSDK_NGX_Parameter_GetI(m_impl->params,
        NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &rrAvail);
    if (!rrAvail) {
        LOG_WARN("DLSS-RR: not available on this GPU / driver");
        return false;
    }
    return true;
}

bool DLSSDContext::getOptimalRenderResolution(
    uint32_t outputW, uint32_t outputH, QualityMode quality,
    uint32_t& renderW, uint32_t& renderH)
{
    if (!m_impl || !m_impl->params) return false;
    unsigned int optW = 0, optH = 0;
    unsigned int wMax = 0, hMax = 0, wMin = 0, hMin = 0;
    float sharpness = 0.0f;
    NVSDK_NGX_Result r = NGX_DLSSD_GET_OPTIMAL_SETTINGS(
        m_impl->params, outputW, outputH, toNgxQuality(quality),
        &optW, &optH, &wMax, &hMax, &wMin, &hMin, &sharpness);
    if (NVSDK_NGX_FAILED(r) || optW == 0) return false;
    renderW = optW;
    renderH = optH;
    return true;
}

bool DLSSDContext::createFeature(
    VkCommandBuffer cmd,
    uint32_t renderW, uint32_t renderH,
    uint32_t outputW, uint32_t outputH,
    QualityMode quality)
{
    if (!m_impl || !m_impl->ngxInitialized || !m_impl->params) return false;

    if (m_impl->feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(m_impl->feature);
        m_impl->feature = nullptr;
    }

    NVSDK_NGX_DLSSD_Create_Params cp{};
    cp.InDenoiseMode    = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    // We pack roughness into worldNormalRoughness.w — RR §3.4.4.2 says set
    // RoughnessMode to Packed in that case.
    cp.InRoughnessMode  = NVSDK_NGX_DLSS_Roughness_Mode_Packed;
    // Linear viewZ vs HW depth — we feed NDC clip.z/clip.w via specHitT
    // matrices and treat the depth buffer as linear (set to Linear, since
    // it's not actually a HW-format depth buffer).
    cp.InUseHWDepth     = NVSDK_NGX_DLSS_Depth_Type_Linear;
    cp.InWidth          = renderW;
    cp.InHeight         = renderH;
    cp.InTargetWidth    = outputW;
    cp.InTargetHeight   = outputH;
    cp.InPerfQualityValue = toNgxQuality(quality);

    int flags = 0;
    flags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    flags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes; // motion vectors at render res
    // DLSS-RR §3.7: Auto-Exposure is unsupported. Do NOT set the flag.
    cp.InFeatureCreateFlags = flags;

    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSSD_EXT1(
        m_impl->device, cmd,
        /*creationNodeMask=*/1, /*visibilityNodeMask=*/1,
        &m_impl->feature, m_impl->params, &cp);
    if (NVSDK_NGX_FAILED(r)) {
        LOG_WARN("DLSS-RR: CREATE_DLSSD_EXT1 failed (0x%x)", (unsigned)r);
        return false;
    }
    LOG_INFO("DLSS-RR: feature created %ux%u -> %ux%u (quality=%d)",
             renderW, renderH, outputW, outputH, (int)quality);
    return true;
}

void DLSSDContext::evaluate(
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
    bool     reset)
{
    if (!m_impl || !m_impl->feature) return;

    VkImageSubresourceRange srr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    NVSDK_NGX_Resource_VK rIn   = NVSDK_NGX_Create_ImageView_Resource_VK(
        inColor,  inColorImage,  srr, colorFormat,  renderW, renderH, /*rw=*/false);
    NVSDK_NGX_Resource_VK rOut  = NVSDK_NGX_Create_ImageView_Resource_VK(
        outColor, outColorImage, srr, VK_FORMAT_R16G16B16A16_SFLOAT, outputW, outputH, /*rw=*/true);
    NVSDK_NGX_Resource_VK rMV   = NVSDK_NGX_Create_ImageView_Resource_VK(
        motion,   motionImage,   srr, motionFormat, renderW, renderH, false);
    NVSDK_NGX_Resource_VK rDep  = NVSDK_NGX_Create_ImageView_Resource_VK(
        depth,    depthImage,    srr, depthFormat,  renderW, renderH, false);
    NVSDK_NGX_Resource_VK rDA   = NVSDK_NGX_Create_ImageView_Resource_VK(
        diffAlbedo, diffAlbedoImage, srr, diffAlbedoFormat, renderW, renderH, false);
    NVSDK_NGX_Resource_VK rSA   = NVSDK_NGX_Create_ImageView_Resource_VK(
        specAlbedo, specAlbedoImage, srr, VK_FORMAT_R16G16B16A16_SFLOAT, renderW, renderH, false);
    NVSDK_NGX_Resource_VK rNR   = NVSDK_NGX_Create_ImageView_Resource_VK(
        normalRoughness, normalRoughnessImage, srr, VK_FORMAT_R16G16B16A16_SFLOAT, renderW, renderH, false);
    NVSDK_NGX_Resource_VK rHit  = NVSDK_NGX_Create_ImageView_Resource_VK(
        specHitT, specHitTImage, srr, VK_FORMAT_R32_SFLOAT, renderW, renderH, false);

    NVSDK_NGX_VK_DLSSD_Eval_Params p{};
    p.pInColor          = &rIn;
    p.pInOutput         = &rOut;
    p.pInMotionVectors  = &rMV;
    p.pInDepth          = &rDep;
    p.pInDiffuseAlbedo  = &rDA;
    p.pInSpecularAlbedo = &rSA;
    // Roughness is packed into normals.w → set both pInNormals and pInRoughness
    // to the same image (helper sets both NGX params; the RR runtime obeys the
    // Roughness_Mode_Packed flag and reads .w for roughness).
    p.pInNormals        = &rNR;
    p.pInRoughness      = &rNR;
    p.pInSpecularHitDistance = &rHit;
    // We provide hitT + matrices so RR derives specular MVs internally
    // (§3.4.9). Matrices: row-major, left-multiply (matches our float4x4).
    p.pInWorldToViewMatrix = worldToView;
    p.pInViewToClipMatrix  = viewToClip;

    p.InJitterOffsetX   = jitterX;
    p.InJitterOffsetY   = jitterY;
    p.InRenderSubrectDimensions = { renderW, renderH };
    p.InReset           = reset ? 1 : 0;
    p.InMVScaleX        = 1.0f;
    p.InMVScaleY        = 1.0f;
    p.InPreExposure     = 1.0f;
    p.InExposureScale   = 1.0f;

    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSSD_EXT(
        cmd, m_impl->feature, m_impl->params, &p);
    if (NVSDK_NGX_FAILED(r)) {
        LOG_WARN("DLSS-RR: EVALUATE_DLSSD_EXT failed (0x%x)", (unsigned)r);
    }
}

bool DLSSDContext::isValid() const {
    return m_impl && m_impl->ngxInitialized && m_impl->feature != nullptr;
}

void DLSSDContext::shutdown() {
    if (!m_impl) return;
    if (m_impl->feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(m_impl->feature);
        m_impl->feature = nullptr;
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
