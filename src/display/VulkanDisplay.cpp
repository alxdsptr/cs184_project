#include "display/VulkanDisplay.h"
#include "util/ImageWriter.h"
#include "util/Log.h"
#include "util/CudaCheck.h"

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <windows.h>
  #define VK_USE_PLATFORM_WIN32_KHR
  #include <vulkan/vulkan_win32.h>
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>

#define VK_CHECK(x) do { \
    VkResult _r = (x); \
    if (_r != VK_SUCCESS) { \
        LOG_ERROR("Vulkan error %d at %s:%d", (int)_r, __FILE__, __LINE__); \
        throw std::runtime_error("Vulkan error"); \
    } \
} while (0)

static const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// ── Debug callback ───────────────────────────────────────────
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCb(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LOG_WARN("Vulkan: %s", data->pMessage);
    }
    return VK_FALSE;
}

// ── File IO ──────────────────────────────────────────────────
static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) {
        LOG_ERROR("Cannot open shader: %s", path.c_str());
        return {};
    }
    size_t sz = (size_t)f.tellg();
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), sz);
    return buf;
}

// ── Win32 handle helpers for CUDA interop ────────────────────
#ifdef _WIN32
static PFN_vkGetMemoryWin32HandleKHR       g_vkGetMemoryWin32HandleKHR       = nullptr;
static PFN_vkGetSemaphoreWin32HandleKHR    g_vkGetSemaphoreWin32HandleKHR    = nullptr;
#endif

void VulkanDisplay::setWindow(GLFWwindow* window) {
    m_window = window;
}

// ─────────────────────────────────────────────────────────────
// Instance
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::createInstance() {
#ifndef NDEBUG
    m_validationEnabled = true;
#endif

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName   = "CUDA Path Tracer";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName        = "pathtracer";
    app.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    // NRI's VK backend always queries promoted-to-core names for KHR
    // functions (vkCmdCopyBuffer2, vkCmdPipelineBarrier2, etc.) and won't
    // fall back to the *KHR aliases — those only exist in 1.3 core. So we
    // request 1.3 whenever NRD/DLSS is compiled in.
#ifdef PATHTRACER_NRD_DLSS_ENABLED
    app.apiVersion         = VK_API_VERSION_1_3;
#else
    app.apiVersion         = VK_API_VERSION_1_2;
#endif

    // Extensions required by GLFW + debug + external memory capabilities
    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    // Merge in NGX's required instance extensions. De-dup naively (O(n²) but
    // n is tiny). If the query fails, we continue — DLSSContext::init will
    // later detect the failure and demote to Native / NRDOnly.
    {
        std::vector<const char*> ngxInst, ngxDev;  // dev unused here
        // We need a function-scope include; avoid pulling the NGX header
        // into this header by using a small extern helper defined in
        // DLSSContext.cpp.
        extern bool DLSSContext_QueryRequiredExts(std::vector<const char*>&,
                                                  std::vector<const char*>&);
        if (DLSSContext_QueryRequiredExts(ngxInst, ngxDev)) {
            auto already = [&](const char* s) {
                for (const char* e : extensions) if (!strcmp(e, s)) return true;
                return false;
            };
            for (const char* e : ngxInst) if (!already(e)) extensions.push_back(e);
        }
    }
#endif

    std::vector<const char*> layers;
    if (m_validationEnabled) {
        // Probe layer availability
        uint32_t count = 0;
        vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> avail(count);
        vkEnumerateInstanceLayerProperties(&count, avail.data());
        bool found = false;
        for (auto& l : avail) if (strcmp(l.layerName, kValidationLayer) == 0) { found = true; break; }
        if (found) {
            layers.push_back(kValidationLayer);
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        } else {
            m_validationEnabled = false;
            LOG_WARN("Vulkan validation layer not available, continuing without it");
        }
    }

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = (uint32_t)extensions.size();
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount       = (uint32_t)layers.size();
    ci.ppEnabledLayerNames     = layers.data();

    VK_CHECK(vkCreateInstance(&ci, nullptr, &m_instance));

    // Record for postfx (NRD/DLSS). Safe because these strings are all static.
    m_enabledInstanceExts = extensions;

    if (m_validationEnabled) {
        VkDebugUtilsMessengerCreateInfoEXT dci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dci.pfnUserCallback = debugCb;
        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
        if (fn) fn(m_instance, &dci, nullptr, &m_debugMessenger);
    }
}

// ─────────────────────────────────────────────────────────────
// Surface
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::createSurface() {
    VK_CHECK(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));
}

// ─────────────────────────────────────────────────────────────
// Physical device — pick one that (a) supports swapchain + graphics,
// (b) matches CUDA's device UUID so interop is legal.
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::pickPhysicalDevice() {
    // Query CUDA device UUID for match
    int cudaDev = 0;
    cudaGetDevice(&cudaDev);
    cudaDeviceProp cudaProp{};
    cudaGetDeviceProperties(&cudaProp, cudaDev);

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("No Vulkan GPUs");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    auto pickFamily = [&](VkPhysicalDevice dev) -> int {
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> q(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, q.data());
        for (uint32_t i = 0; i < qcount; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, m_surface, &present);
            if ((q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) return (int)i;
        }
        return -1;
    };

    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    int chosenFamily = -1;
    for (auto d : devices) {
        VkPhysicalDeviceIDProperties idProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(d, &props2);

        int fam = pickFamily(d);
        if (fam < 0) continue;

        // Match CUDA UUID if available
        if (memcmp(idProps.deviceUUID, cudaProp.uuid.bytes, VK_UUID_SIZE) == 0) {
            chosen = d;
            chosenFamily = fam;
            memcpy(m_deviceUUID, idProps.deviceUUID, VK_UUID_SIZE);
            LOG_INFO("Vulkan device: %s (matched CUDA UUID)", props2.properties.deviceName);
            break;
        }
        // Fallback: first discrete GPU
        if (chosen == VK_NULL_HANDLE && props2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            chosen = d;
            chosenFamily = fam;
            memcpy(m_deviceUUID, idProps.deviceUUID, VK_UUID_SIZE);
        }
    }

    if (chosen == VK_NULL_HANDLE) throw std::runtime_error("No suitable Vulkan GPU");
    m_physicalDevice = chosen;
    m_graphicsQueueFamily = (uint32_t)chosenFamily;
}

// ─────────────────────────────────────────────────────────────
// Device + queue, with external memory / semaphore extensions
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::createDevice() {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = m_graphicsQueueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    std::vector<const char*> exts = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
#endif
    };

    // Probe available device extensions once; used by both NRD/DLSS and NRI setup.
    uint32_t availCnt = 0;
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &availCnt, nullptr);
    std::vector<VkExtensionProperties> avail(availCnt);
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &availCnt, avail.data());
    auto isExtSupported = [&](const char* s) {
        for (auto& p : avail) if (!strcmp(p.extensionName, s)) return true;
        return false;
    };
    auto alreadyEnabled = [&](const char* s) {
        for (const char* e : exts) if (!strcmp(e, s)) return true;
        return false;
    };

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    {
        std::vector<const char*> ngxInst, ngxDev;
        extern bool DLSSContext_QueryRequiredExts(std::vector<const char*>&,
                                                  std::vector<const char*>&);
        if (DLSSContext_QueryRequiredExts(ngxInst, ngxDev)) {
            for (const char* e : ngxDev) {
                if (!alreadyEnabled(e) && isExtSupported(e)) exts.push_back(e);
            }
        }
    }

    // NRI (backing NRD) requires dynamicRendering, synchronization2,
    // extendedDynamicState and push-descriptor. The first three are core in
    // Vulkan 1.3 (promoted from these KHR/EXT extensions); we still list
    // them so drivers/loader don't complain on edge cases, but the real
    // feature bits come from VkPhysicalDeviceVulkan13Features below. The
    // last one (push_descriptor) was NOT promoted to 1.3 core, so it must
    // be enabled as an extension for NRI's `vkCmdPushDescriptorSet` lookup
    // to succeed.
    const char* nriRequired[] = {
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    };
    for (const char* e : nriRequired) {
        if (!alreadyEnabled(e) && isExtSupported(e)) exts.push_back(e);
    }
#endif

    VkPhysicalDeviceFeatures feats{};
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount    = 1;
    ci.pQueueCreateInfos       = &qci;
    ci.enabledExtensionCount   = (uint32_t)exts.size();
    ci.ppEnabledExtensionNames = exts.data();
    ci.pEnabledFeatures        = &feats;

#ifdef PATHTRACER_NRD_DLSS_ENABLED
    // NRI's VK backend uses (but doesn't itself enable) several feature bits.
    // Vulkan validation rightly complains if the bits aren't on, and the
    // downstream behaviour — `vkGetBufferDeviceAddress` silently returning
    // 0, partially-bound descriptor sets being malformed — produces exactly
    // the kind of "silently crashes deep inside DenoiseVK" symptom we just
    // fought through. Enable everything NRI pokes at.
    VkPhysicalDeviceDynamicRenderingFeatures dynRenderFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
    VkPhysicalDeviceSynchronization2Features sync2Feat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    VkPhysicalDeviceExtendedDynamicStateFeaturesEXT edsFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
    // bufferDeviceAddress: NRI allocates with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    //                      and calls vkGetBufferDeviceAddress on the result.
    // descriptorBindingPartiallyBound: NRD's pipeline layouts use this on every
    //                      descriptor set layout binding range.
    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceDescriptorIndexingFeatures diFeat{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    dynRenderFeat.dynamicRendering       = VK_TRUE;
    sync2Feat.synchronization2           = VK_TRUE;
    edsFeat.extendedDynamicState         = VK_TRUE;
    bdaFeat.bufferDeviceAddress          = VK_TRUE;
    diFeat.descriptorBindingPartiallyBound = VK_TRUE;
    // Other descriptor-indexing bits NRI may use under the hood; cheap to turn on.
    diFeat.shaderSampledImageArrayNonUniformIndexing   = VK_TRUE;
    diFeat.shaderStorageImageArrayNonUniformIndexing   = VK_TRUE;
    diFeat.runtimeDescriptorArray                      = VK_TRUE;
    // UpdateAfterBind + UpdateUnusedWhilePending: without these, NRI creates
    // its descriptor set layouts with VkDescriptorBindingFlags(0). Then when
    // NewFrame() calls vkUpdateDescriptorSets on a set still referenced by
    // an in-flight command buffer, validation fires 03047 and the shader
    // reads undefined descriptors — which shows up as "all-snow" output from
    // the denoiser even though it's technically "running".
    diFeat.descriptorBindingSampledImageUpdateAfterBind  = VK_TRUE;
    diFeat.descriptorBindingStorageImageUpdateAfterBind  = VK_TRUE;
    diFeat.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    diFeat.descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
    diFeat.descriptorBindingUpdateUnusedWhilePending     = VK_TRUE;

    dynRenderFeat.pNext = &sync2Feat;
    sync2Feat.pNext     = &edsFeat;
    edsFeat.pNext       = &bdaFeat;
    bdaFeat.pNext       = &diFeat;
    ci.pNext            = &dynRenderFeat;
#endif

    VK_CHECK(vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device));
    vkGetDeviceQueue(m_device, m_graphicsQueueFamily, 0, &m_graphicsQueue);

    // Record for postfx (NRD/DLSS).
    m_enabledDeviceExts = exts;

#ifdef _WIN32
    g_vkGetMemoryWin32HandleKHR = (PFN_vkGetMemoryWin32HandleKHR)
        vkGetDeviceProcAddr(m_device, "vkGetMemoryWin32HandleKHR");
    g_vkGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)
        vkGetDeviceProcAddr(m_device, "vkGetSemaphoreWin32HandleKHR");
#endif
}

// ─────────────────────────────────────────────────────────────
// Swapchain
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &caps);

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, m_surface, &fmtCount, formats.data());

    // We write tonemap-ed, sRGB-encoded 8-bit bytes from CUDA. Picking an
    // sRGB swapchain format would make the hardware apply a *second* linear->
    // sRGB conversion when the fragment shader writes into the attachment,
    // blowing out the image. Force a UNORM format.
    VkSurfaceFormatKHR surfaceFmt = formats[0];
    bool found = false;
    for (auto& f : formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM)
            && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surfaceFmt = f;
            found = true;
            break;
        }
    }
    if (!found) {
        LOG_WARN("No UNORM swapchain format; falling back to %d (may look over-bright)",
                 (int)surfaceFmt.format);
    }
    LOG_INFO("Swapchain format: %d", (int)surfaceFmt.format);
    m_swapchainFormat = surfaceFmt.format;

    // Extent. The swapchain MUST be created at caps.currentExtent (when not
    // sentinel) — Vulkan ignores anything else. We track that separately as
    // m_swapchainExtent. Render-side resources (m_sampledImage, m_interopBuffer,
    // screenshot staging) stay at the caller-requested m_width/m_height so a
    // headless 1920x1080 replay produces 1920x1080 PNGs even when Windows
    // clamps the swapchain to 1920x1055 because of the taskbar. The
    // fullscreen-triangle blit in present() samples m_sampledImage with
    // normalized UVs, so any size mismatch resolves as automatic scaling
    // when the swapchain image is presented to a visible window.
    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        extent.width  = std::clamp(m_width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        extent.height = std::clamp(m_height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    m_swapchainExtent = extent;
    if (extent.width != m_width || extent.height != m_height) {
        LOG_INFO("Swapchain extent %ux%u differs from render extent %ux%u - "
                 "render-side buffers stay at request, swapchain blit will scale.",
                 extent.width, extent.height, m_width, m_height);
    }

    // Prefer MAILBOX (tear-free, no frame-rate cap) over FIFO (hard vsync cap).
    // MAILBOX is optional, so query and fall back to FIFO (always supported).
    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
    uint32_t pmCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &pmCount, nullptr);
    std::vector<VkPresentModeKHR> pms(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &pmCount, pms.data());
    for (auto pm : pms) {
        if (pm == VK_PRESENT_MODE_MAILBOX_KHR) { presentMode = pm; break; }
    }
    LOG_INFO("Swapchain present mode: %s",
             presentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : "FIFO");

    // MAILBOX needs ≥3 images to actually decouple render rate from refresh rate.
    m_minImageCount = caps.minImageCount + 1;
    if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR && m_minImageCount < 3) {
        m_minImageCount = 3;
    }
    if (caps.maxImageCount > 0 && m_minImageCount > caps.maxImageCount) {
        m_minImageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface          = m_surface;
    sci.minImageCount    = m_minImageCount;
    sci.imageFormat      = surfaceFmt.format;
    sci.imageColorSpace  = surfaceFmt.colorSpace;
    sci.imageExtent      = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform     = caps.currentTransform;
    sci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode      = presentMode;
    sci.clipped          = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(m_device, &sci, nullptr, &m_swapchain));

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &count, nullptr);
    m_swapchainImages.resize(count);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &count, m_swapchainImages.data());

    m_swapchainImageViews.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = m_swapchainImages[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = surfaceFmt.format;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(m_device, &ivci, nullptr, &m_swapchainImageViews[i]));
    }
}

void VulkanDisplay::destroySwapchain() {
    for (auto fb : m_framebuffers) if (fb) vkDestroyFramebuffer(m_device, fb, nullptr);
    m_framebuffers.clear();
    for (auto v : m_swapchainImageViews) if (v) vkDestroyImageView(m_device, v, nullptr);
    m_swapchainImageViews.clear();
    m_swapchainImages.clear();
    if (m_swapchain) { vkDestroySwapchainKHR(m_device, m_swapchain, nullptr); m_swapchain = VK_NULL_HANDLE; }
}

// ─────────────────────────────────────────────────────────────
// Render pass + framebuffers
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::createRenderPass() {
    VkAttachmentDescription color{};
    color.format = m_swapchainFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &color;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &sub;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;
    VK_CHECK(vkCreateRenderPass(m_device, &rpci, nullptr, &m_renderPass));
}

void VulkanDisplay::createFramebuffers() {
    m_framebuffers.resize(m_swapchainImageViews.size());
    for (size_t i = 0; i < m_swapchainImageViews.size(); i++) {
        VkImageView att = m_swapchainImageViews[i];
        VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fci.renderPass      = m_renderPass;
        fci.attachmentCount = 1;
        fci.pAttachments    = &att;
        fci.width  = m_swapchainExtent.width;
        fci.height = m_swapchainExtent.height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(m_device, &fci, nullptr, &m_framebuffers[i]));
    }
}

// ─────────────────────────────────────────────────────────────
// Sync: per-frame binary semaphores + fence,
//       plus CUDA-interop timeline semaphores
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::createSyncObjects() {
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t i = 0; i < kFramesInFlight; i++) {
        VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &m_imageAvailable[i]));
        VK_CHECK(vkCreateFence(m_device, &fci, nullptr, &m_inFlight[i]));
    }
    // One renderFinished semaphore per swapchain image — indexing by frame-in-flight
    // is unsafe when swapchain image count != kFramesInFlight (signal may alias a
    // semaphore still owned by a pending presentation).
    m_renderFinished.resize(m_swapchainImages.size(), VK_NULL_HANDLE);
    for (auto& s : m_renderFinished) {
        VK_CHECK(vkCreateSemaphore(m_device, &sci, nullptr, &s));
    }

    // For the CUDA-Vulkan interop we use a single binary semaphore pair,
    // exported and imported into CUDA. For simplicity (and because CUDA's
    // external semaphore support on Windows prefers binary D3D12 fence
    // semaphores otherwise), we actually just rely on queue idle in present()
    // — CUDA's default stream completes before we submit copy. Marker kept
    // for future async interop.
    m_cudaReadySem = VK_NULL_HANDLE;
    m_vulkanReadySem = VK_NULL_HANDLE;
}

void VulkanDisplay::destroySyncObjects() {
    for (uint32_t i = 0; i < kFramesInFlight; i++) {
        if (m_imageAvailable[i]) vkDestroySemaphore(m_device, m_imageAvailable[i], nullptr);
        if (m_inFlight[i])       vkDestroyFence(m_device, m_inFlight[i], nullptr);
        m_imageAvailable[i] = VK_NULL_HANDLE;
        m_inFlight[i] = VK_NULL_HANDLE;
    }
    for (auto& s : m_renderFinished) {
        if (s) vkDestroySemaphore(m_device, s, nullptr);
    }
    m_renderFinished.clear();
}

void VulkanDisplay::createCommandPool() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.queueFamilyIndex = m_graphicsQueueFamily;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(m_device, &ci, nullptr, &m_commandPool));

    VkCommandBufferAllocateInfo aci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    aci.commandPool = m_commandPool;
    aci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    aci.commandBufferCount = kFramesInFlight;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &aci, m_commandBuffers));
}

// ─────────────────────────────────────────────────────────────
// Descriptor set layout + pool for the fullscreen blit
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 1;
    ci.pBindings = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_descriptorSetLayout));
}

void VulkanDisplay::createDescriptorPool() {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = 1;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(m_device, &ci, nullptr, &m_descriptorPool));

    VkDescriptorSetAllocateInfo asi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    asi.descriptorPool = m_descriptorPool;
    asi.descriptorSetCount = 1;
    asi.pSetLayouts = &m_descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(m_device, &asi, &m_descriptorSet));
}

void VulkanDisplay::createImGuiDescriptorPool() {
    // ImGui needs a pool big enough for its own allocations.
    VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER,                1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       1000},
    };
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = 1000;
    ci.poolSizeCount = (uint32_t)std::size(sizes);
    ci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(m_device, &ci, nullptr, &m_imguiDescriptorPool));
}

// ─────────────────────────────────────────────────────────────
// Graphics pipeline — fullscreen triangle + textured frag
// ─────────────────────────────────────────────────────────────
VkShaderModule VulkanDisplay::loadShaderModule(const std::string& path) const {
    auto bytes = readFile(path);
    if (bytes.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(m_device, &ci, nullptr, &mod));
    return mod;
}

void VulkanDisplay::createPipeline() {
    // Resolve SPIR-V paths relative to the executable (CMake copies them to
    // $<TARGET_FILE_DIR:pathtracer>/shaders/). Fall back to CWD variants so
    // developers running from arbitrary working directories still find them.
    namespace fs = std::filesystem;
    fs::path exeDir;
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0) exeDir = fs::path(std::wstring(buf, n)).parent_path();
#endif

    const char* kVert = "fullscreen_quad_vk.vert.spv";
    const char* kFrag = "fullscreen_quad_vk.frag.spv";
    fs::path candidates[] = {
        exeDir / "shaders" / kVert,
        fs::path("shaders") / kVert,
        fs::path("../shaders") / kVert,
        fs::path("Release/shaders") / kVert,
    };
    fs::path vertPath, fragPath;
    for (auto& c : candidates) {
        if (fs::exists(c)) {
            vertPath = c;
            fragPath = c.parent_path() / kFrag;
            break;
        }
    }
    if (vertPath.empty()) {
        LOG_ERROR("SPIR-V shaders not found near exe or CWD; rebuild the 'shaders' target");
        throw std::runtime_error("SPIR-V shader not found");
    }
    VkShaderModule vs = loadShaderModule(vertPath.string());
    VkShaderModule fs_mod = loadShaderModule(fragPath.string());
    if (!vs || !fs_mod) throw std::runtime_error("SPIR-V shader load failed");

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_mod;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments    = &blend;

    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2;
    ds.pDynamicStates    = dyn;

    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 1;
    lci.pSetLayouts    = &m_descriptorSetLayout;
    VK_CHECK(vkCreatePipelineLayout(m_device, &lci, nullptr, &m_pipelineLayout));

    VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState   = &ms;
    gpci.pColorBlendState    = &cb;
    gpci.pDynamicState       = &ds;
    gpci.layout              = m_pipelineLayout;
    gpci.renderPass          = m_renderPass;
    gpci.subpass             = 0;
    VK_CHECK(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &gpci, nullptr, &m_pipeline));

    vkDestroyShaderModule(m_device, vs, nullptr);
    vkDestroyShaderModule(m_device, fs_mod, nullptr);

    // Sampler used for the blit
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(m_device, &sci, nullptr, &m_sampler));
}

// ─────────────────────────────────────────────────────────────
// Interop: buffer exported to CUDA
// ─────────────────────────────────────────────────────────────
uint32_t VulkanDisplay::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    throw std::runtime_error("No matching memory type");
}

void VulkanDisplay::createInteropBuffer() {
    m_interopSize = (VkDeviceSize)m_width * m_height * 4;

    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
#ifdef _WIN32
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &ext;
    bci.size  = m_interopSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &bci, nullptr, &m_interopBuffer));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(m_device, m_interopBuffer, &req);

    VkExportMemoryAllocateInfo expInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
#ifdef _WIN32
    expInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    expInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &expInfo;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &mai, nullptr, &m_interopMemory));
    VK_CHECK(vkBindBufferMemory(m_device, m_interopBuffer, m_interopMemory, 0));

    // Export handle and import into CUDA.
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR ghi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    ghi.memory = m_interopMemory;
    ghi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE handle = nullptr;
    VK_CHECK(g_vkGetMemoryWin32HandleKHR(m_device, &ghi, &handle));

    cudaExternalMemoryHandleDesc desc{};
    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    desc.size = req.size;
    cudaExternalMemory_t extMem = nullptr;
    CUDA_CHECK(cudaImportExternalMemory(&extMem, &desc));
    m_cudaExtMem = (void*)extMem;

    cudaExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size   = req.size;
    bd.flags  = 0;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&m_cudaDevPtr, extMem, &bd));

    // cudaImportExternalMemory duplicates the handle; we own the original.
    CloseHandle(handle);
#else
    // (Linux path omitted — project targets Windows.)
#endif
}

void VulkanDisplay::destroyInteropBuffer() {
    if (m_cudaDevPtr) {
        cudaFree(m_cudaDevPtr);  // safe on mapped buffer
        m_cudaDevPtr = nullptr;
    }
    if (m_cudaExtMem) {
        cudaDestroyExternalMemory((cudaExternalMemory_t)m_cudaExtMem);
        m_cudaExtMem = nullptr;
    }
    if (m_interopBuffer) { vkDestroyBuffer(m_device, m_interopBuffer, nullptr); m_interopBuffer = VK_NULL_HANDLE; }
    if (m_interopMemory) { vkFreeMemory(m_device, m_interopMemory, nullptr);    m_interopMemory = VK_NULL_HANDLE; }
}

// ─────────────────────────────────────────────────────────────
// Sampled image that the CUDA buffer is copied into each frame
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::createSampledImage() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format    = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent    = {m_width, m_height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling  = VK_IMAGE_TILING_OPTIMAL;
    ici.usage   = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // post-NRD composite writes here
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(m_device, &ici, nullptr, &m_sampledImage));

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(m_device, m_sampledImage, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(m_device, &mai, nullptr, &m_sampledImageMemory));
    VK_CHECK(vkBindImageMemory(m_device, m_sampledImage, m_sampledImageMemory, 0));

    VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivci.image = m_sampledImage;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(m_device, &ivci, nullptr, &m_sampledImageView));

    m_sampledImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanDisplay::destroySampledImage() {
    if (m_sampledImageView)   { vkDestroyImageView(m_device, m_sampledImageView, nullptr);   m_sampledImageView = VK_NULL_HANDLE; }
    if (m_sampledImage)       { vkDestroyImage(m_device, m_sampledImage, nullptr);           m_sampledImage = VK_NULL_HANDLE; }
    if (m_sampledImageMemory) { vkFreeMemory(m_device, m_sampledImageMemory, nullptr);       m_sampledImageMemory = VK_NULL_HANDLE; }
}

void VulkanDisplay::createScreenshotResources() {
    // Staging buffer sized for the full RGBA8 framebuffer, persistently
    // mapped so saveToPNG can memcpy out without per-call map/unmap.
    const VkDeviceSize size = (VkDeviceSize)m_width * (VkDeviceSize)m_height * 4u;
    m_screenshotStagingSize = size;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = size;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(m_device, &bci, nullptr, &m_screenshotStagingBuf));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(m_device, m_screenshotStagingBuf, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = req.size;
    // Prefer HOST_CACHED for the screenshot staging — we only ever READ this
    // buffer on the CPU, and reading from uncached host-coherent memory is
    // PCIe-bound at ~150-250 MB/s, which dominated saveToPNG. With cached
    // memory the memcpy runs at L2/main-memory bandwidth (~5+ GB/s).
    //
    // Try HOST_VISIBLE | HOST_CACHED | HOST_COHERENT first (most common on
    // PC GPUs — no explicit invalidate needed), then HOST_VISIBLE | HOST_CACHED
    // (needs vkInvalidateMappedMemoryRanges before each read), then fall back
    // to plain HOST_VISIBLE | HOST_COHERENT for portability.
    auto tryFindMem = [&](VkMemoryPropertyFlags flags) -> int32_t {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & flags) == flags) {
                return (int32_t)i;
            }
        }
        return -1;
    };
    const VkMemoryPropertyFlags hostCachedCoherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT  |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags hostCached =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const VkMemoryPropertyFlags hostCoherent =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int32_t typeIdx = tryFindMem(hostCachedCoherent);
    if (typeIdx >= 0) {
        m_screenshotStagingNeedsInvalidate = false;
        LOG_INFO("Screenshot staging: HOST_CACHED|HOST_COHERENT");
    } else if ((typeIdx = tryFindMem(hostCached)) >= 0) {
        m_screenshotStagingNeedsInvalidate = true;
        LOG_INFO("Screenshot staging: HOST_CACHED (explicit invalidate)");
    } else {
        typeIdx = tryFindMem(hostCoherent);
        if (typeIdx < 0) throw std::runtime_error("No host-visible memory type for screenshot staging");
        m_screenshotStagingNeedsInvalidate = false;
        LOG_WARN("Screenshot staging: HOST_COHERENT only (slow PCIe readback)");
    }
    mai.memoryTypeIndex = (uint32_t)typeIdx;
    VK_CHECK(vkAllocateMemory(m_device, &mai, nullptr, &m_screenshotStagingMem));
    VK_CHECK(vkBindBufferMemory(m_device, m_screenshotStagingBuf, m_screenshotStagingMem, 0));
    VK_CHECK(vkMapMemory(m_device, m_screenshotStagingMem, 0, size, 0,
                         &m_screenshotStagingMap));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool        = m_commandPool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &cbai, &m_screenshotCmd));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(m_device, &fci, nullptr, &m_screenshotFence));
}

void VulkanDisplay::destroyScreenshotResources() {
    if (m_screenshotFence) {
        vkDestroyFence(m_device, m_screenshotFence, nullptr);
        m_screenshotFence = VK_NULL_HANDLE;
    }
    if (m_screenshotCmd) {
        vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_screenshotCmd);
        m_screenshotCmd = VK_NULL_HANDLE;
    }
    if (m_screenshotStagingMap) {
        vkUnmapMemory(m_device, m_screenshotStagingMem);
        m_screenshotStagingMap = nullptr;
    }
    if (m_screenshotStagingMem) {
        vkFreeMemory(m_device, m_screenshotStagingMem, nullptr);
        m_screenshotStagingMem = VK_NULL_HANDLE;
    }
    if (m_screenshotStagingBuf) {
        vkDestroyBuffer(m_device, m_screenshotStagingBuf, nullptr);
        m_screenshotStagingBuf = VK_NULL_HANDLE;
    }
    m_screenshotStagingSize = 0;
}

void VulkanDisplay::updateDescriptorSet() {
    VkDescriptorImageInfo di{};
    di.sampler     = m_sampler;
    di.imageView   = m_sampledImageView;
    di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = m_descriptorSet;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &di;
    vkUpdateDescriptorSets(m_device, 1, &w, 0, nullptr);
}

void VulkanDisplay::transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                          VkImageLayout oldLayout, VkImageLayout newLayout,
                                          VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                                          VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) const {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// ─────────────────────────────────────────────────────────────
// Public lifecycle
// ─────────────────────────────────────────────────────────────
void VulkanDisplay::init(uint32_t width, uint32_t height) {
    if (!m_window) throw std::runtime_error("VulkanDisplay::setWindow must be called before init");
    m_width = width;
    m_height = height;

    createInstance();
    createSurface();
    pickPhysicalDevice();
    createDevice();
    createSwapchain();
    createRenderPass();
    createFramebuffers();
    createCommandPool();
    createSyncObjects();
    createDescriptorSetLayout();
    createDescriptorPool();
    createPipeline();
    createImGuiDescriptorPool();

    createInteropBuffer();
    createSampledImage();
    createScreenshotResources();
    updateDescriptorSet();

    LOG_INFO("VulkanDisplay initialized (%ux%u)", m_width, m_height);
}

void VulkanDisplay::resize(uint32_t width, uint32_t height) {
    if (width == m_width && height == m_height) return;
    if (width == 0 || height == 0) return;

    vkDeviceWaitIdle(m_device);
    destroyScreenshotResources();
    destroySampledImage();
    destroyInteropBuffer();
    destroySwapchain();

    m_width = width;
    m_height = height;
    createSwapchain();
    createFramebuffers();
    createInteropBuffer();
    createSampledImage();
    createScreenshotResources();
    updateDescriptorSet();
}

void* VulkanDisplay::mapForCUDA() {
    return m_cudaDevPtr;
}

void VulkanDisplay::unmapFromCUDA() {
    // We rely on cudaDeviceSynchronize before the Vulkan copy. The path tracer
    // uses the default stream; Renderer::renderFrame is synchronous by the time
    // present() runs because subsequent CPU work reads results. For stricter
    // ordering we'd import an external semaphore pair; keeping it simple here.
    CUDA_CHECK(cudaStreamSynchronize(0));
}

void VulkanDisplay::present() {
    uint32_t frame = m_frameIndex;

    vkWaitForFences(m_device, 1, &m_inFlight[frame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX,
                                         m_imageAvailable[frame], VK_NULL_HANDLE, &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        // Swapchain needs recreate; caller will handle via framebuffer size check.
        return;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) VK_CHECK(acq);

    vkResetFences(m_device, 1, &m_inFlight[frame]);

    VkCommandBuffer cmd = m_commandBuffers[frame];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    if (m_prePresentRecorder) {
        // Non-Native path: callback is responsible for populating m_sampledImage
        // and must leave it in SHADER_READ_ONLY_OPTIMAL for the swapchain blit
        // pipeline below. It gets the command buffer mid-recording.
        m_prePresentRecorder(cmd, m_prePresentUser);
        m_sampledImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    } else {
        // ── Native path: classic CUDA interop buffer → sampled image copy.
        transitionImageLayout(cmd, m_sampledImage,
            m_sampledImageLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {m_width, m_height, 1};
        vkCmdCopyBufferToImage(cmd, m_interopBuffer, m_sampledImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        transitionImageLayout(cmd, m_sampledImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        m_sampledImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // 4. Begin render pass
    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass = m_renderPass;
    rbi.framebuffer = m_framebuffers[imageIndex];
    rbi.renderArea.offset = {0, 0};
    rbi.renderArea.extent = m_swapchainExtent;
    rbi.clearValueCount = 1;
    rbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{0, 0, (float)m_swapchainExtent.width, (float)m_swapchainExtent.height, 0, 1};
    VkRect2D sc{{0, 0}, m_swapchainExtent};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_descriptorSet, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // 5. ImGui draws on top
    if (m_imguiRecorder) {
        m_imguiRecorder(cmd, m_imguiUser);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores    = &m_imageAvailable[frame];
    si.pWaitDstStageMask  = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &m_renderFinished[imageIndex];
    VK_CHECK(vkQueueSubmit(m_graphicsQueue, 1, &si, m_inFlight[frame]));
    // Latch this fence as the "most recent present" so saveToPNG can wait on
    // exactly the work that wrote m_sampledImage instead of draining the
    // entire device with vkDeviceWaitIdle.
    m_lastPresentFenceIdx = frame;

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &m_renderFinished[imageIndex];
    pi.swapchainCount = 1;
    pi.pSwapchains    = &m_swapchain;
    pi.pImageIndices  = &imageIndex;
    VkResult pr = vkQueuePresentKHR(m_graphicsQueue, &pi);
    if (pr != VK_SUCCESS && pr != VK_ERROR_OUT_OF_DATE_KHR && pr != VK_SUBOPTIMAL_KHR) {
        VK_CHECK(pr);
    }

    m_frameIndex = (m_frameIndex + 1) % kFramesInFlight;
}

void VulkanDisplay::waitIdle() {
    if (m_device) vkDeviceWaitIdle(m_device);
}

void VulkanDisplay::shutdown() {
    if (!m_device) return;

    // Drain any pending image-encoder jobs before tearing down. The worker
    // thread doesn't touch Vulkan directly, but if the caller forgot to flush
    // we still want completed PNGs on disk before exit.
    if (m_imageWriter) {
        m_imageWriter->flush();
        m_imageWriter.reset();
    }

    vkDeviceWaitIdle(m_device);

    destroyScreenshotResources();
    destroySampledImage();
    destroyInteropBuffer();
    if (m_sampler) { vkDestroySampler(m_device, m_sampler, nullptr); m_sampler = VK_NULL_HANDLE; }

    if (m_pipeline)       vkDestroyPipeline(m_device, m_pipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    if (m_descriptorPool) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    if (m_imguiDescriptorPool) vkDestroyDescriptorPool(m_device, m_imguiDescriptorPool, nullptr);
    if (m_descriptorSetLayout) vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    if (m_commandPool)    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    destroySyncObjects();
    if (m_renderPass)     vkDestroyRenderPass(m_device, m_renderPass, nullptr);

    destroySwapchain();

    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;

    if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_debugMessenger) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(m_instance, m_debugMessenger, nullptr);
    }
    if (m_instance) vkDestroyInstance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
}

void VulkanDisplay::setImGuiRecorder(void (*fn)(VkCommandBuffer, void*), void* user) {
    m_imguiRecorder = fn;
    m_imguiUser = user;
}

void VulkanDisplay::setPrePresentRecorder(void (*fn)(VkCommandBuffer, void*), void* user) {
    m_prePresentRecorder = fn;
    m_prePresentUser = user;
}

VulkanDisplay::VulkanDisplay() = default;
VulkanDisplay::~VulkanDisplay() = default;

// ─────────────────────────────────────────────────────────────
// Screenshot: read the displayed image back to host and hand it to the
// background ImageWriter for encoding (PNG by extension, raw RGBA8 otherwise).
//
// Source must be `m_sampledImage` (RGBA8_UNORM), NOT `m_cudaDevPtr`. In
// non-Native modes (NRDOnly / NRDDLSS / DLSSOnly) the CUDA interop buffer is
// never written by the kernel — the path-trace output goes through the
// pre-present recorder (NRD denoise + composite + DLSS) and is rendered
// directly into `m_sampledImage`. Reading `m_cudaDevPtr` in those modes
// returns whatever uninitialized GPU memory was allocated to the interop
// buffer (typically 0xFF on Win32 OPAQUE_WIN32 → pure-white screenshot
// regardless of the actual on-screen output).
//
// We pump the GPU to idle first so that the most recently presented frame is
// fully resolved into `m_sampledImage`, then run the pooled image→buffer
// copy command and wait on a fence (cheaper than vkQueueWaitIdle). The host
// memcpy + encode is offloaded so the caller can move on to the next pose.
// ─────────────────────────────────────────────────────────────
bool VulkanDisplay::saveToPNG(const std::string& path) {
    if (m_width == 0 || m_height == 0 || !m_sampledImage) return false;
    if (!m_screenshotStagingBuf || !m_screenshotCmd || !m_screenshotFence) return false;

    // Wait for ONLY the most recent present submission to finish — that's the
    // submit that wrote m_sampledImage. vkDeviceWaitIdle would also drain
    // unrelated GPU work and (worse) couldn't pipeline against the encoder
    // workers; on a 1080p replay this saves ~10 ms/frame.
    //
    // Safety: the fence at m_lastPresentFenceIdx is signaled-on-completion by
    // present()'s vkQueueSubmit and only reset on a *future* present() call
    // that reuses the same slot (kFramesInFlight=2, cyclic). Replay calls
    // saveToPNG immediately after present() with no further present in
    // between, so the fence is in {signaled, signal-pending} when we wait.
    vkWaitForFences(m_device, 1, &m_inFlight[m_lastPresentFenceIdx],
                    VK_TRUE, UINT64_MAX);

    const VkDeviceSize size = m_screenshotStagingSize;

    VK_CHECK(vkResetCommandBuffer(m_screenshotCmd, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_screenshotCmd, &bi);

    // m_sampledImage was last left in SHADER_READ_ONLY_OPTIMAL by present().
    // Transition to TRANSFER_SRC for the copy, then back so subsequent frames
    // observe the layout the present-loop expects.
    transitionImageLayout(m_screenshotCmd, m_sampledImage,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {m_width, m_height, 1};
    vkCmdCopyImageToBuffer(m_screenshotCmd, m_sampledImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_screenshotStagingBuf, 1, &region);

    transitionImageLayout(m_screenshotCmd, m_sampledImage,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    vkEndCommandBuffer(m_screenshotCmd);

    VK_CHECK(vkResetFences(m_device, 1, &m_screenshotFence));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &m_screenshotCmd;
    VK_CHECK(vkQueueSubmit(m_graphicsQueue, 1, &si, m_screenshotFence));
    VK_CHECK(vkWaitForFences(m_device, 1, &m_screenshotFence, VK_TRUE, UINT64_MAX));

    // If we picked HOST_CACHED memory without HOST_COHERENT, the GPU's writes
    // may still be sitting in some intermediate buffer not yet visible to a
    // CPU cache load. Invalidate makes them visible. With HOST_COHERENT this
    // is implicit and the call is unnecessary.
    if (m_screenshotStagingNeedsInvalidate) {
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = m_screenshotStagingMem;
        range.offset = 0;
        range.size   = VK_WHOLE_SIZE;
        VK_CHECK(vkInvalidateMappedMemoryRanges(m_device, 1, &range));
    }

    // Memcpy out of the persistently-mapped staging buffer into a fresh
    // owned vector that we hand off to the encoder thread. The vector
    // ownership transfer is the cheapest way to pipeline encode against the
    // next frame's render — no shared state, no extra synchronization.
    std::vector<unsigned char> pixels((size_t)size);
    std::memcpy(pixels.data(), m_screenshotStagingMap, (size_t)size);

    if (!m_imageWriter) {
        m_imageWriter = std::make_unique<ImageWriter>(m_imageWriterFailFast);
    }
    return m_imageWriter->submit(std::move(pixels), m_width, m_height, path);
}

void VulkanDisplay::flushImageWriter() {
    if (m_imageWriter) m_imageWriter->flush();
}

size_t VulkanDisplay::imageWriterFailureCount() const {
    return m_imageWriter ? m_imageWriter->failureCount() : 0;
}

std::string VulkanDisplay::imageWriterFirstFailurePath() const {
    return m_imageWriter ? m_imageWriter->firstFailurePath() : std::string{};
}
