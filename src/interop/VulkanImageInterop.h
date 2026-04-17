#pragma once

// Allocates a VkImage whose backing memory is exported via Win32 handle and
// imported into CUDA as a cudaMipmappedArray. The resulting cudaSurfaceObject_t
// lets CUDA kernels write straight into storage that NRD (Vulkan compute) can
// subsequently read — zero-copy interop.
//
// Deliberately Windows-only; the host project does not target Linux/POSIX FD
// interop. Mirrors VulkanDisplay::createInteropBuffer() but for images.

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>
#include <cstdint>

class SharedVulkanImage {
public:
    SharedVulkanImage() = default;
    ~SharedVulkanImage();
    SharedVulkanImage(const SharedVulkanImage&) = delete;
    SharedVulkanImage& operator=(const SharedVulkanImage&) = delete;
    SharedVulkanImage(SharedVulkanImage&& other) noexcept { *this = std::move(other); }
    SharedVulkanImage& operator=(SharedVulkanImage&& other) noexcept;

    // `cudaChannelFormat` must match `format` (e.g. VK_FORMAT_R16G16B16A16_SFLOAT
    // ↔ cudaChannelFormatKindFloat with 4×16 bits). Caller is responsible for
    // consistency; a mismatch will produce a garbled surface.
    bool create(VkDevice device, VkPhysicalDevice phys,
                uint32_t width, uint32_t height,
                VkFormat format,
                VkImageUsageFlags usage,
                const cudaChannelFormatDesc& cudaChanDesc);

    void destroy();

    VkImage        image()      const { return m_image; }
    VkImageView    view()       const { return m_view; }
    VkDeviceMemory memory()     const { return m_memory; }
    VkFormat       format()     const { return m_format; }
    uint32_t       width()      const { return m_width; }
    uint32_t       height()     const { return m_height; }
    // Valid for CUDA kernels to write to (surf2Dwrite). Null after destroy().
    cudaSurfaceObject_t surface() const { return m_surface; }

    // One-shot layout transition helper (no-op when oldLayout == newLayout).
    static void transition(VkCommandBuffer cmd, VkImage image,
                           VkImageLayout oldLayout, VkImageLayout newLayout,
                           VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                           VkAccessFlags srcAccess, VkAccessFlags dstAccess);

private:
    VkDevice               m_device       = VK_NULL_HANDLE;
    VkImage                m_image        = VK_NULL_HANDLE;
    VkDeviceMemory         m_memory       = VK_NULL_HANDLE;
    VkImageView            m_view         = VK_NULL_HANDLE;
    VkFormat               m_format       = VK_FORMAT_UNDEFINED;
    uint32_t               m_width        = 0;
    uint32_t               m_height       = 0;

    cudaExternalMemory_t   m_extMem       = nullptr;
    cudaMipmappedArray_t   m_mipArray     = nullptr;
    cudaArray_t            m_array        = nullptr; // level 0 of m_mipArray
    cudaSurfaceObject_t    m_surface      = 0;
};
