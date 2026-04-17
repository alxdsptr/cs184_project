#ifdef _WIN32
#  define NOMINMAX
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "interop/VulkanImageInterop.h"  // brings in <vulkan/vulkan.h>
#include "util/CudaCheck.h"

#ifdef _WIN32
#  define VK_USE_PLATFORM_WIN32_KHR
#  include <vulkan/vulkan_win32.h>
#endif

#include <stdexcept>
#include <utility>

namespace {

// Locally resolved once per process — VulkanDisplay resolves its own copy, but
// this utility is independent of that translation unit.
PFN_vkGetMemoryWin32HandleKHR getMemHandleFn(VkDevice device) {
    static PFN_vkGetMemoryWin32HandleKHR fn = nullptr;
    if (!fn) {
        fn = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
            device, "vkGetMemoryWin32HandleKHR");
    }
    return fn;
}

uint32_t findMemType(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    throw std::runtime_error("VulkanImageInterop: no matching memory type");
}

} // namespace

SharedVulkanImage::~SharedVulkanImage() {
    destroy();
}

SharedVulkanImage& SharedVulkanImage::operator=(SharedVulkanImage&& other) noexcept {
    if (this != &other) {
        destroy();
        m_device   = other.m_device;
        m_image    = other.m_image;
        m_memory   = other.m_memory;
        m_view     = other.m_view;
        m_format   = other.m_format;
        m_width    = other.m_width;
        m_height   = other.m_height;
        m_extMem   = other.m_extMem;
        m_mipArray = other.m_mipArray;
        m_array    = other.m_array;
        m_surface  = other.m_surface;
        other.m_device   = VK_NULL_HANDLE;
        other.m_image    = VK_NULL_HANDLE;
        other.m_memory   = VK_NULL_HANDLE;
        other.m_view     = VK_NULL_HANDLE;
        other.m_extMem   = nullptr;
        other.m_mipArray = nullptr;
        other.m_array    = nullptr;
        other.m_surface  = 0;
    }
    return *this;
}

bool SharedVulkanImage::create(VkDevice device, VkPhysicalDevice phys,
                               uint32_t width, uint32_t height,
                               VkFormat format,
                               VkImageUsageFlags usage,
                               const cudaChannelFormatDesc& cudaChanDesc)
{
    m_device = device;
    m_format = format;
    m_width  = width;
    m_height = height;

    // ── VkImage (exportable, exclusive, 2D, 1 mip, 1 layer) ─────────
    VkExternalMemoryImageCreateInfo extImg{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
#ifdef _WIN32
    extImg.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    extImg.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.pNext         = &extImg;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = { width, height, 1 };
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // NRD needs at least STORAGE | SAMPLED; composite pass needs SAMPLED;
    // CUDA surface write requires STORAGE; we add both unconditionally here.
    ici.usage         = usage | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(m_device, &ici, nullptr, &m_image) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(m_device, m_image, &req);

    VkExportMemoryAllocateInfo expInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
#ifdef _WIN32
    expInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    expInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    // Tells the driver to back the allocation with a dedicated VkImage alias
    // — required on many drivers for external image memory to work.
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.image = m_image;
    expInfo.pNext = &dedicated;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext           = &expInfo;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = findMemType(phys, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(m_device, &mai, nullptr, &m_memory) != VK_SUCCESS) {
        vkDestroyImage(m_device, m_image, nullptr);
        m_image = VK_NULL_HANDLE;
        return false;
    }
    if (vkBindImageMemory(m_device, m_image, m_memory, 0) != VK_SUCCESS) {
        destroy();
        return false;
    }

    // ── Import into CUDA as mipmapped array → surface object ────────
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR ghi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    ghi.memory     = m_memory;
    ghi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE handle = nullptr;
    auto getFn = getMemHandleFn(m_device);
    if (!getFn || getFn(m_device, &ghi, &handle) != VK_SUCCESS) {
        destroy();
        return false;
    }

    cudaExternalMemoryHandleDesc extDesc{};
    extDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    extDesc.handle.win32.handle = handle;
    extDesc.size                = req.size;
    // Dedicated allocation → flag required for CUDA to accept it.
    extDesc.flags               = cudaExternalMemoryDedicated;
    CUDA_CHECK(cudaImportExternalMemory(&m_extMem, &extDesc));
    CloseHandle(handle); // CUDA duplicates internally

    cudaExternalMemoryMipmappedArrayDesc mmDesc{};
    mmDesc.offset              = 0;
    mmDesc.formatDesc          = cudaChanDesc;
    mmDesc.extent              = make_cudaExtent(width, height, 0);
    // cudaArraySurfaceLoadStore → allow surface writes (what we need).
    mmDesc.flags               = cudaArraySurfaceLoadStore;
    mmDesc.numLevels           = 1;
    CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&m_mipArray, m_extMem, &mmDesc));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&m_array, m_mipArray, 0));

    cudaResourceDesc surfRes{};
    surfRes.resType         = cudaResourceTypeArray;
    surfRes.res.array.array = m_array;
    CUDA_CHECK(cudaCreateSurfaceObject(&m_surface, &surfRes));
#endif

    // ── VkImageView — uses the same format the caller specified ─────
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image    = m_image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(m_device, &vci, nullptr, &m_view) != VK_SUCCESS) {
        destroy();
        return false;
    }
    return true;
}

void SharedVulkanImage::destroy() {
    if (m_surface) {
        cudaDestroySurfaceObject(m_surface);
        m_surface = 0;
    }
    // m_array is a view into m_mipArray; freeing the mipmapped array releases it.
    m_array = nullptr;
    if (m_mipArray) {
        cudaFreeMipmappedArray(m_mipArray);
        m_mipArray = nullptr;
    }
    if (m_extMem) {
        cudaDestroyExternalMemory(m_extMem);
        m_extMem = nullptr;
    }
    if (m_view && m_device) {
        vkDestroyImageView(m_device, m_view, nullptr);
        m_view = VK_NULL_HANDLE;
    }
    if (m_image && m_device) {
        vkDestroyImage(m_device, m_image, nullptr);
        m_image = VK_NULL_HANDLE;
    }
    if (m_memory && m_device) {
        vkFreeMemory(m_device, m_memory, nullptr);
        m_memory = VK_NULL_HANDLE;
    }
    m_device = VK_NULL_HANDLE;
    m_width = m_height = 0;
    m_format = VK_FORMAT_UNDEFINED;
}

void SharedVulkanImage::transition(VkCommandBuffer cmd, VkImage image,
                                   VkImageLayout oldLayout, VkImageLayout newLayout,
                                   VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                                   VkAccessFlags srcAccess, VkAccessFlags dstAccess)
{
    if (oldLayout == newLayout) return;
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.srcAccessMask       = srcAccess;
    bar.dstAccessMask       = dstAccess;
    bar.oldLayout           = oldLayout;
    bar.newLayout           = newLayout;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image               = image;
    bar.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0,
                         0, nullptr, 0, nullptr, 1, &bar);
}
