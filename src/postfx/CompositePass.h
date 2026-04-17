#pragma once

// Thin Vulkan helper wrapping the three fullscreen-quad shader permutations
// used by the NRD/DLSS pipeline:
//
//   COMPOSITE_TONEMAP    — NRD-only mode. 4 inputs (diff/spec/albedo/emissive),
//                          output is sRGB LDR written straight to the swapchain.
//   COMPOSITE_LINEAR_HDR — NRD+DLSS mode. Same 4 inputs, output is linear HDR
//                          RGBA16F (consumed by DLSS).
//   TONEMAP_ONLY         — NRD+DLSS mode, post-DLSS. 1 input (HDR at output res),
//                          output is sRGB LDR on the swapchain.

#include <vulkan/vulkan.h>
#include <cstdint>

class CompositePass {
public:
    enum Mode {
        COMPOSITE_TONEMAP,
        COMPOSITE_LINEAR_HDR,
        TONEMAP_ONLY,
    };

    CompositePass() = default;
    ~CompositePass();
    CompositePass(const CompositePass&) = delete;
    CompositePass& operator=(const CompositePass&) = delete;

    // `outputFormat` must match the render-pass attachment format of the
    // framebuffer the caller will bind when recording.
    bool init(VkDevice device, VkRenderPass renderPass, Mode mode,
              VkFormat outputFormat, const char* spirvDir);

    // Update the sampled inputs the next record() call will consume. For the
    // 4-input modes pass all 4 views; for TONEMAP_ONLY only `a` is used.
    void setInputs(VkImageView a, VkImageView b = VK_NULL_HANDLE,
                   VkImageView c = VK_NULL_HANDLE, VkImageView d = VK_NULL_HANDLE);

    // Record a fullscreen draw into the active render pass. Caller is
    // responsible for vkCmdBeginRenderPass / vkCmdEndRenderPass.
    void record(VkCommandBuffer cmd, VkExtent2D extent,
                float exposure, int tonemapMode);

    void shutdown();
    bool isValid() const { return m_pipeline != VK_NULL_HANDLE; }

private:
    VkShaderModule loadSpv(const char* path);
    void           writeDescriptorSet();
    void           createSampler();

    VkDevice              m_device   = VK_NULL_HANDLE;
    VkRenderPass          m_renderPass = VK_NULL_HANDLE;
    Mode                  m_mode     = COMPOSITE_TONEMAP;
    uint32_t              m_inputCount = 0;
    bool                  m_usesPush  = false;

    VkSampler             m_sampler  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool     = VK_NULL_HANDLE;
    VkDescriptorSet       m_set      = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            m_pipeline = VK_NULL_HANDLE;

    VkImageView           m_inputs[4] = {};
    bool                  m_setDirty = true;
};
