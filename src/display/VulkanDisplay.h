#pragma once
#include "display/DisplayBackend.h"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class ImageWriter;

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
    VulkanDisplay();
    ~VulkanDisplay() override;

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

    // Optional per-frame recorder that replaces the default CUDA interop
    // buffer-to-image copy. When set, the callback is invoked in the middle
    // of present() — it must, by its end, leave `sampledImage()` in the
    // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL layout containing the LDR
    // swapchain-sized output. Used by Renderer in non-Native modes to
    // redirect the pipeline through NRD + composite + DLSS.
    void setPrePresentRecorder(void (*fn)(VkCommandBuffer, void*), void* user);

    // Getters used by the pre-present recorder.
    VkImage     sampledImage()  const { return m_sampledImage; }
    VkImageView sampledImageView() const { return m_sampledImageView; }
    VkFormat    sampledImageFormat() const { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t    width()  const { return m_width; }
    uint32_t    height() const { return m_height; }

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

    // Asynchronously persists the most recently presented frame to disk.
    // The GPU→host readback is synchronous (we can't return until the next
    // frame may overwrite m_sampledImage), but the encode + file IO runs on
    // a background ImageWriter thread. Format is chosen by the file
    // extension: `.png` → fpng, anything else → raw RGBA8 dump.
    //
    // Returns false if the writer is shutting down or — in fail-fast mode —
    // a prior encode already failed. Callers should treat false as a signal
    // to abort the capture loop. True only means the readback queued; the
    // actual file write may still fail later, in which case it is recorded
    // via imageWriterFailureCount() / imageWriterFirstFailurePath().
    bool saveToPNG(const std::string& path);

    // Blocks until all queued ImageWriter jobs have finished. Call before
    // returning from a headless replay loop so we don't tear down before the
    // last frames hit disk.
    void flushImageWriter();

    // Configure the ImageWriter's failure policy. Must be called before the
    // first saveToPNG() since the writer is constructed lazily on first use.
    // Default is non-fail-fast (suitable for ad-hoc GUI screenshots).
    void setImageWriterFailFast(bool enabled) { m_imageWriterFailFast = enabled; }

    // Cumulative count of background encode failures since the writer was
    // created. Returns 0 before the writer is constructed.
    size_t imageWriterFailureCount() const;
    // Path of the first failed encode (empty if none). For end-of-run logs.
    std::string imageWriterFirstFailurePath() const;

    // Enabled extensions recorded at instance/device creation. Read by the
    // NRD/DLSS postfx layer to construct an NRI DeviceCreationVKDesc. Native
    // mode never touches these; they're inert.
    const std::vector<const char*>& enabledInstanceExtensions() const { return m_enabledInstanceExts; }
    const std::vector<const char*>& enabledDeviceExtensions()   const { return m_enabledDeviceExts; }

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
    void createScreenshotResources();
    void destroyScreenshotResources();
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
    // Index into m_inFlight of the most recently submitted present's fence,
    // used by saveToPNG to wait on exactly that work (not the whole device).
    // Default 0 is harmless: m_inFlight[0] is created VK_FENCE_CREATE_SIGNALED,
    // so a wait before the first present returns immediately.
    uint32_t        m_lastPresentFenceIdx = 0;
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

    // Pre-present recorder (replaces buffer-to-image copy when non-null).
    void (*m_prePresentRecorder)(VkCommandBuffer, void*) = nullptr;
    void* m_prePresentUser = nullptr;

    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;

    // Pooled resources for saveToPNG. Sized to match m_width*m_height*4
    // and rebuilt on every resize alongside m_sampledImage. Avoids per-call
    // vkAllocateMemory / command-buffer alloc churn that dominated replay.
    VkBuffer        m_screenshotStagingBuf  = VK_NULL_HANDLE;
    VkDeviceMemory  m_screenshotStagingMem  = VK_NULL_HANDLE;
    void*           m_screenshotStagingMap  = nullptr;  // persistently mapped
    VkDeviceSize    m_screenshotStagingSize = 0;
    bool            m_screenshotStagingNeedsInvalidate = false;  // true if HOST_CACHED w/o HOST_COHERENT
    VkCommandBuffer m_screenshotCmd         = VK_NULL_HANDLE;
    VkFence         m_screenshotFence       = VK_NULL_HANDLE;

    // Background image encoder. Allocated lazily on first saveToPNG so
    // headless / non-capture runs pay nothing for the worker thread.
    std::unique_ptr<ImageWriter> m_imageWriter;
    bool                         m_imageWriterFailFast = false;

    bool m_validationEnabled = false;

    // Recorded (not populated outside init) — const ptrs borrowed from the
    // same allocation sites as createInstance/createDevice, safe for the life
    // of the display backend.
    std::vector<const char*> m_enabledInstanceExts;
    std::vector<const char*> m_enabledDeviceExts;
};
