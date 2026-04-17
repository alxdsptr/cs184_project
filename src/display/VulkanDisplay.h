#pragma once
#include "display/DisplayBackend.h"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

struct GLFWwindow;

// CUDA <-> Vulkan interop display backend.
//
// Strategy:
//   - CUDA path tracer writes into a device-side buffer (uchar4 RGBA).
//   - The buffer memory is allocated via Vulkan with external-memory export
//     (opaque Win32 handle), then imported into CUDA as `cudaExternalMemory_t`.
//     That gives us a zero-copy pointer both sides can write/read.
//   - Each frame we record: vkCmdCopyBufferToImage into the sampled image,
//     blit ImGui on top via imgui_impl_vulkan, present.
//   - Synchronization across the CUDA/Vulkan boundary uses a timeline
//     semaphore exported from Vulkan and imported into CUDA.
class VulkanDisplay : public DisplayBackend {
public:
    // Vulkan requires a window handle up front (surface creation). The
    // backend must therefore be wired to GLFW before init().
    void setWindow(GLFWwindow* window);

    void init(uint32_t width, uint32_t height) override;
    void resize(uint32_t width, uint32_t height) override;

    // Returns a CUDA device pointer (uchar4*) that survives for the lifetime
    // of the current size. Kernels may write directly; no actual "mapping"
    // call is needed since the buffer is persistently imported.
    void* mapForCUDA() override;
    void  unmapFromCUDA() override;

    // Acquires swapchain image, records blit + ImGui draw (via callback),
    // presents. `recordImGui` is invoked with the VkCommandBuffer after the
    // fullscreen blit and before end-render-pass.
    void  present() override;

    // Blocks until all submitted work completes. Call before tearing down
    // anything that the GPU might still be using (ImGui resources, etc.).
    void waitIdle();

    void shutdown() override;

    // Called by Application each frame — the GUI layer records its draw
    // data into the active command buffer. Set to nullptr for headless.
    void setImGuiRecorder(void (*fn)(VkCommandBuffer, void*), void* user);

    // For GUI init: these getters expose the objects imgui_impl_vulkan needs.
    VkInstance       instance()       const { return m_instance; }
    VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    VkDevice         device()         const { return m_device; }
    VkQueue          graphicsQueue()  const { return m_graphicsQueue; }
    uint32_t         graphicsQueueFamily() const { return m_graphicsQueueFamily; }
    VkRenderPass     renderPass()     const { return m_renderPass; }
    VkDescriptorPool descriptorPool() const { return m_imguiDescriptorPool; }
    uint32_t         swapchainImageCount() const { return (uint32_t)m_swapchainImages.size(); }
    uint32_t         minImageCount()  const { return m_minImageCount; }

    bool saveToPNG(const std::string& path) const;

private:
    // ── Init sub-steps ───────────────────────────────────────────
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createDevice();
    void createSwapchain();
    void destroySwapchain();
    void createRenderPass();
    void createFramebuffers();
    void createSyncObjects();
    void destroySyncObjects();
    void createCommandPool();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createPipeline();
    void createImGuiDescriptorPool();

    void createInteropBuffer();
    void destroyInteropBuffer();
    void createSampledImage();
    void destroySampledImage();
    void updateDescriptorSet();

    // Helpers
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    VkShaderModule loadShaderModule(const std::string& path) const;
    void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                               VkImageLayout oldLayout, VkImageLayout newLayout,
                               VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                               VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage) const;

    // ── State ────────────────────────────────────────────────────
    GLFWwindow* m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    VkInstance       m_instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR     m_surface        = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice         m_device         = VK_NULL_HANDLE;
    VkQueue          m_graphicsQueue  = VK_NULL_HANDLE;
    uint32_t         m_graphicsQueueFamily = UINT32_MAX;
    uint8_t          m_deviceUUID[VK_UUID_SIZE] = {};

    VkSwapchainKHR   m_swapchain = VK_NULL_HANDLE;
    VkFormat         m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D       m_swapchainExtent = {};
    std::vector<VkImage>       m_swapchainImages;
    std::vector<VkImageView>   m_swapchainImageViews;
    std::vector<VkFramebuffer> m_framebuffers;
    uint32_t         m_minImageCount = 2;

    VkRenderPass m_renderPass = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_descriptorPool      = VK_NULL_HANDLE;
    VkDescriptorSet       m_descriptorSet       = VK_NULL_HANDLE;

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_pipeline       = VK_NULL_HANDLE;

    VkCommandPool    m_commandPool = VK_NULL_HANDLE;

    // Per-frame-in-flight (command recording, fences, image-available)
    static constexpr uint32_t kFramesInFlight = 2;
    VkCommandBuffer m_commandBuffers[kFramesInFlight] = {};
    VkSemaphore     m_imageAvailable[kFramesInFlight] = {};
    VkFence         m_inFlight[kFramesInFlight]       = {};
    uint32_t        m_frameIndex = 0;
    // Per-swapchain-image: render-finished signals the image is ready to
    // present. Keyed on image index so reuse only happens after re-acquire.
    std::vector<VkSemaphore> m_renderFinished;

    // Sampled image that CUDA data gets copied into
    VkImage        m_sampledImage       = VK_NULL_HANDLE;
    VkDeviceMemory m_sampledImageMemory = VK_NULL_HANDLE;
    VkImageView    m_sampledImageView   = VK_NULL_HANDLE;
    VkSampler      m_sampler            = VK_NULL_HANDLE;
    VkImageLayout  m_sampledImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Exported buffer shared with CUDA
    VkBuffer       m_interopBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory m_interopMemory       = VK_NULL_HANDLE;
    VkDeviceSize   m_interopSize         = 0;
    void*          m_cudaExtMem          = nullptr;  // cudaExternalMemory_t (void* to avoid cuda header)
    void*          m_cudaDevPtr          = nullptr;

    // CUDA <-> Vulkan semaphore (timeline)
    VkSemaphore    m_cudaReadySem        = VK_NULL_HANDLE;
    VkSemaphore    m_vulkanReadySem      = VK_NULL_HANDLE;
    void*          m_cudaReadyExtSem     = nullptr;
    void*          m_vulkanReadyExtSem   = nullptr;
    uint64_t       m_timelineValue       = 0;

    // ImGui render callback
    void (*m_imguiRecorder)(VkCommandBuffer, void*) = nullptr;
    void* m_imguiUser = nullptr;

    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;

    bool m_validationEnabled = false;
};
